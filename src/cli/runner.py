"""CLI 运行器：交互式输入 + Agent 执行 + 进度展示 + 持久化。

CLI 交互流程（main）：
1. 初始化日志并打印 Banner；校验 LLM_API_KEY，缺失则提示并以退出码 1 退出。
2. 提示用户输入数据文件路径（回车使用默认 datasets/sales.csv）与分析需求
   （为空则退出码 1）；预读数据集做可读性校验，失败退出码 1。
3. 创建 PersistenceService（PostgreSQL + Redis，不可用时内部优雅降级），
   建立 session/task/run 记录并打印 ID。
4. 调用 run_agent() 流式执行 LangGraph，逐节点打印进度；
   Ctrl+C 中断退出码 130，其他异常标记 run 失败并退出码 1。
5. 结束后根据 status 打印报告保存路径或失败告警（退出条件）。
"""

from __future__ import annotations

import sys
from typing import Any

from src.agent.graph import build_graph, make_initial_state
from src.config.settings import get_settings
from src.db.service import PersistenceService, tool_type_of
from src.logging_config import setup_logging
from src.tools.file_tools import load_dataframe

# 启动时打印的欢迎横幅
BANNER = """========================================
       AI Data Analyst Agent
========================================"""

# 节点 → 进度标签：图每产出一个节点更新，终端就打印对应中文提示
NODE_LABELS = {
    "task_understanding": "[任务理解] 正在理解分析任务...",
    "skill_selection": "[Skill] 正在选择分析技能...",
    "profiler": "[数据画像] 正在读取数据...",
    "planner": "[规划] 正在制定分析计划...",
    "agent": "[Agent] 正在决策...",
    "tools": "[工具]",
    "insight": "[洞察] 正在提炼关键洞察...",
    "report": "[报告] 正在生成最终报告...",
}

# 节点 → 持久化状态（写入 Redis session 的 current_status）
NODE_STATUS = {
    "task_understanding": "understanding",
    "skill_selection": "skill_selection",
    "profiler": "profiling",
    "planner": "planning",
    "agent": "tool_calling",
    "tools": "observing",
    "insight": "insight",
    "report": "report",
}


def _print_tool_names(update: dict[str, Any]) -> None:
    """从 agent 节点的消息中提取并打印将要调用的工具名。

    参数：
        update: LangGraph 节点产出的状态片段（含 messages）。
    返回值：
        None；副作用是向终端打印一行 "→ 调用工具: ..."。
    """
    messages = update.get("messages") or []
    names = []
    # AIMessage.tool_calls 是本轮决策出的工具调用列表（含 name/args）
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            names.append(tc.get("name", ""))
    if names:
        print("  → 调用工具: " + ", ".join(names))


def _print_tool_results(update: dict[str, Any]) -> None:
    """打印工具执行结果摘要（成功 ✓ / 失败 ✗，结果截断到 200 字）。

    参数：
        update: tools 节点产出的状态片段（含 tool_results）。
    返回值：
        None；副作用是逐行打印工具状态与结果摘要。
    """
    results = update.get("tool_results") or []
    for r in results:
        status = r.get("status", "?")
        name = r.get("name", "?")
        result = str(r.get("result", ""))
        # 折叠换行并截断，避免长输出打乱进度展示
        short = result[:200].replace("\n", " ").strip()
        mark = "✓" if status == "success" else "✗"
        print(f"  {mark} [{name}] {short}")


def _persist_update(persistence: PersistenceService, node_name: str, update: dict[str, Any]) -> None:
    """把节点更新写入持久化（task/run/tool_call/result/report/session）。

    参数：
        persistence: 持久化服务实例（PostgreSQL + Redis，内部已处理降级）。
        node_name: 本轮产出更新的图节点名。
        update: 节点产出的状态片段。
    返回值：
        None；副作用是把运行状态、计划、工具调用、洞察、报告落库。
    """
    # 先更新 session 当前状态与步数（状态码用 NODE_STATUS 映射）
    persistence.update_status(NODE_STATUS.get(node_name, node_name), step=update.get("step_count"))

    if node_name == "tools":
        # 工具节点：调用与结果按下标一一对应，逐条入库便于事后审计
        tool_calls = update.get("tool_calls", [])
        tool_results = update.get("tool_results", [])
        for tc, tr in zip(tool_calls, tool_results):
            name = tc.get("name", "")
            persistence.record_tool_call(
                name=name,
                tool_type=tool_type_of(name),
                arguments=tc.get("args"),
                result=tr.get("result"),
                status=tr.get("status"),
            )
    elif node_name == "skill_selection":
        # Skill 选择阶段：分析计划尚为空，仅记录选中的技能列表
        persistence.record_plan(None, update.get("selected_skills"))
    elif node_name == "planner":
        # 规划阶段：写入正式分析计划
        persistence.record_plan(update.get("analysis_plan"))
    elif node_name == "insight":
        # 洞察阶段：把提炼出的关键结论作为阶段结果保存
        persistence.record_result("insight", {"insights": update.get("insights", [])})
    elif "final_report" in update:
        # 报告节点（兜底分支）：保存报告路径与全文
        persistence.record_report(update.get("report_path"), update.get("final_report"))


def run_agent(
    dataset_path: str,
    request: str,
    persistence: PersistenceService | None = None,
) -> dict[str, Any]:
    """构建并流式执行 Agent 图，返回最终结果。

    参数：
        dataset_path: 数据文件路径。
        request: 用户的自然语言分析需求。
        persistence: 可选持久化服务；传 None 时只打印不落库。
    返回值：
        dict: {final_report 报告全文, report_path 报告路径, status 完成状态}；
        图未产出最终报告时 status 保持初始值 "error"。
    """
    graph = build_graph()
    initial = make_initial_state(dataset_path, request)

    # 默认按失败初始化，只有 report 节点真正产出 final_report 才标记完成
    final: dict[str, Any] = {"final_report": "", "report_path": "", "status": "error"}
    final_step_count = 0

    # stream_mode="updates"：每个节点执行完就 yield 一次 {节点名: 状态增量}
    for chunk in graph.stream(initial, stream_mode="updates"):
        for node_name, update in chunk.items():
            label = NODE_LABELS.get(node_name, f"[{node_name}]")
            print(f"\n{label}")

            # 针对关键节点打印专属进度明细
            if node_name == "agent":
                _print_tool_names(update)
            elif node_name == "tools":
                _print_tool_results(update)
            elif node_name == "skill_selection":
                selected = update.get("selected_skills") or []
                print(f"  已选择 Skills: {', '.join(selected) if selected else '（无）'}")
            elif node_name == "profiler":
                # 画像节点只展示第一条 observation 的首行，避免刷屏
                obs = (update.get("observations") or [""])[0]
                print(f"  {obs.splitlines()[0] if obs else ''}")

            # 记录最新步数（节点可能多轮循环递增）
            if "step_count" in update:
                final_step_count = update["step_count"]

            if persistence:
                _persist_update(persistence, node_name, update)

            # 报告节点产出：捕获最终结果，作为循环结束后的返回依据
            if "final_report" in update:
                final["final_report"] = update["final_report"]
                final["report_path"] = update.get("report_path", "")
                final["status"] = update.get("status", "done")

    if persistence:
        # 图执行完毕（正常或提前结束）：关闭 run，按最终状态写 completed/failed
        done = final["status"] == "done"
        persistence.complete(
            "completed" if done else "failed",
            step_count=final_step_count,
            error=None if done else "Agent 未正常完成",
        )

    return final


def main() -> None:
    """CLI 入口：交互式收集输入并驱动一次完整的 Agent 分析运行。

    返回值：
        None；通过 sys.exit() 设置进程退出码：
        0 正常完成；1 参数/数据/运行出错；130 用户 Ctrl+C 中断。
    """
    setup_logging()
    print(BANNER)
    print()

    settings = get_settings()

    # 前置校验：没有 LLM API Key 后续所有节点都无法运行，直接退出
    if not settings.llm_api_key:
        print("⚠ 未检测到 LLM_API_KEY。")
        print("  请复制 .env.example 为 .env 并填写 API Key 后重试。")
        sys.exit(1)

    # 交互输入 1/2：数据集路径（回车用默认样例数据）
    dataset_path = input("请输入数据文件路径：\n> ").strip()
    if not dataset_path:
        dataset_path = "datasets/sales.csv"
        print(f"(使用默认数据集: {dataset_path})")

    # 交互输入 2/2：分析需求；为空无法继续，属于明确的退出条件
    request = input("\n请输入你的分析需求：\n> ").strip()
    if not request:
        print("分析需求不能为空。")
        sys.exit(1)

    # 校验数据集可读：提前失败，避免 Agent 跑到工具节点才报错
    try:
        df = load_dataframe(dataset_path)
    except Exception as e:
        print(f"✗ 无法读取数据集: {e}")
        sys.exit(1)

    print(f"\n已加载数据集: {dataset_path}（{len(df)} 行 × {df.shape[1]} 列）")
    print("-" * 60)

    # 启动持久化（PostgreSQL + Redis，不可用时优雅降级）
    persistence = PersistenceService()
    # 建立本次会话/任务/运行记录，并把数据集元信息一并存档
    ids = persistence.start_run(
        dataset_path,
        request,
        dataset_info={"row_count": len(df), "column_count": df.shape[1], "columns": list(df.columns)},
    )
    print(f"[Session] {ids['session_id']}")
    if ids["task_id"] is not None:
        print(f"[Task] {ids['task_id']}")
    if ids["run_id"] is not None:
        print(f"[Agent Run] {ids['run_id']}")
    print("-" * 60)

    try:
        # 核心执行：流式跑图，节点进度在 run_agent 内部实时打印
        result = run_agent(dataset_path, request, persistence=persistence)
    except KeyboardInterrupt:
        # 用户主动中断：约定使用 130（128 + SIGINT 信号编号 2）
        print("\n已中断。")
        sys.exit(130)
    except Exception as e:
        # 运行期异常：标记 run 失败后以退出码 1 结束
        print(f"\n✗ Agent 运行失败: {type(e).__name__}: {e}")
        persistence.complete("failed", error=str(e))
        sys.exit(1)

    # 收尾：按最终状态给出成功/失败提示（CLI 循环的退出分支）
    print("\n" + "=" * 60)
    if result["status"] == "done" and result["report_path"]:
        print(f"✅ 分析完成，报告已保存到：{result['report_path']}")
        if ids["task_id"] is not None:
            print("Task saved / Report saved / Analysis results saved")
    else:
        print("⚠ 分析未正常完成，请检查上方日志。")
    print("=" * 60)
