"""CLI 运行器：交互式输入 + Agent 执行 + 进度展示 + 持久化 + 连续对话。

CLI 交互流程（main）：
1. 初始化日志并打印 Banner；校验 LLM_API_KEY，缺失则提示并以退出码 1 退出。
2. 提示用户输入数据文件路径（回车使用默认 datasets/sales.csv）与分析需求
   （为空则退出码 1）；预读数据集做可读性校验，失败退出码 1。
3. 创建 PersistenceService（PostgreSQL + Redis，不可用时内部优雅降级），
   建立 session/task/run 记录并打印 ID。
4. 调用 run_agent() 流式执行 LangGraph，逐节点打印进度；
   Ctrl+C 中断退出码 130，其他异常标记 run 失败并退出码 1。
5. 结束一轮后进入**连续追问**：直接输入新问题即可复用同一数据集与上下文，
   输入 /new 更换数据集，输入 /exit（或空行）退出。

连续对话通过 ConversationMemory 实现：只注入同一数据集的最近若干轮结论摘要，
换数据集自动视为全新任务，避免历史污染当前分析。
"""

from __future__ import annotations

import sys
from typing import Any

from src.agent.graph import build_graph, make_initial_state
from src.agent.memory import ConversationMemory
from src.config.settings import get_settings
from src.db.service import PersistenceService, tool_type_of
from src.logging_config import setup_logging
from src.tools.file_tools import load_dataframe

# 启动时打印的欢迎横幅
BANNER = """========================================
       AI Data Analyst Agent
========================================"""

# 追问阶段的退出/切换指令（同时支持中英文，减少记忆负担）
EXIT_COMMANDS = {"/exit", "/quit", "/q", "exit", "quit"}
NEW_DATASET_COMMANDS = {"/new", "/dataset"}

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
    # 耗时来自同批次的 tool_calls 台账（两者按下标一一对应）
    calls = update.get("tool_calls") or []
    durations = [int(c.get("duration_ms") or 0) for c in calls]
    for i, r in enumerate(results):
        status = r.get("status", "?")
        name = r.get("name", "?")
        result = str(r.get("result", ""))
        # 折叠换行并截断，避免长输出打乱进度展示
        short = result[:200].replace("\n", " ").strip()
        mark = "✓" if status == "success" else "✗"
        # 展示耗时：慢工具一眼可见，便于定位性能问题
        cost = f" ({durations[i]}ms)" if i < len(durations) and durations[i] else ""
        print(f"  {mark} [{name}]{cost} {short}")


def _observation_summary(update: dict[str, Any]) -> str:
    """取出节点 observation 的正文首行（跳过 "[标签]" 形式的标题行）。

    节点的观察文本统一是「[标签]\\n正文」，直接取第一行只会打印标签，
    对用户没有信息量。

    :param update: 节点产出的状态片段
    :return: 适合打印的一行摘要；取不到时返回空串
    """
    obs = (update.get("observations") or [""])[0]
    lines = [ln.strip() for ln in str(obs).splitlines() if ln.strip()]
    # 首行是 "[xxx]" 纯标签时跳过它
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        lines = lines[1:]
    return lines[0] if lines else ""


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
            status = tr.get("status")
            persistence.record_tool_call(
                name=name,
                tool_type=tool_type_of(name),
                arguments=tc.get("args"),
                result=tr.get("result"),
                # 失败时把错误摘要单独入 error_message 字段，便于按错误检索
                error=None if status == "success" else str(tr.get("result", ""))[:500],
                status=status,
                duration_ms=int(tc.get("duration_ms") or 0),
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
    conversation_context: str = "",
) -> dict[str, Any]:
    """构建并流式执行 Agent 图，返回最终结果。

    参数：
        dataset_path: 数据文件路径。
        request: 用户的自然语言分析需求。
        persistence: 可选持久化服务；传 None 时只打印不落库。
        conversation_context: 连续对话的历史上下文（由 ConversationMemory 生成）；
            为空串表示全新任务。
    返回值：
        dict: {final_report 报告全文, report_path 报告路径, status 完成状态,
        insights 本轮洞察, degraded 降级原因, llm 用量汇总}；
        图未产出最终报告时 status 保持初始值 "error"。
    """
    graph = build_graph()
    initial = make_initial_state(dataset_path, request, conversation_context)

    # 默认按失败初始化，只有 report 节点真正产出 final_report 才标记完成
    final: dict[str, Any] = {
        "final_report": "",
        "report_path": "",
        "status": "error",
        "insights": [],
        "degraded": [],
    }
    final_step_count = 0
    # LLM 用量汇总：由 record_llm_calls 逐批累加，最终写入 agent_runs
    llm_summary: dict[str, Any] = {
        "calls": 0, "failed": 0, "total_tokens": None, "duration_ms": 0,
    }

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
                # 画像观察文本形如 "[数据画像]\n<摘要>"，首行只是标签，
                # 这里取正文首行展示，避免终端只打出一个没有信息量的标题
                print(f"  {_observation_summary(update)}")

            # 记录最新步数（节点可能多轮循环递增）
            if "step_count" in update:
                final_step_count = update["step_count"]

            # 累加本节点产生的 LLM 调用记录与降级原因
            final["insights"].extend(update.get("insights") or [])
            final["degraded"].extend(update.get("degraded") or [])

            if persistence:
                _persist_update(persistence, node_name, update)
                # LLM 用量单独汇总：既要落表，也要写回运行级统计
                sum_part = persistence.record_llm_calls(update.get("llm_calls"))
                llm_summary["calls"] += sum_part["calls"]
                llm_summary["failed"] += sum_part["failed"]
                llm_summary["duration_ms"] += sum_part["duration_ms"]
                if sum_part["total_tokens"] is not None:
                    llm_summary["total_tokens"] = (
                        llm_summary["total_tokens"] or 0
                    ) + sum_part["total_tokens"]

            # 报告节点产出：捕获最终结果，作为循环结束后的返回依据
            if "final_report" in update:
                final["final_report"] = update["final_report"]
                final["report_path"] = update.get("report_path", "")
                final["status"] = update.get("status", "done")

    final["llm"] = llm_summary
    # 降级/失败时把原因打印出来，避免用户以为「报告正常 = 分析完整」
    if final["degraded"]:
        print("\n⚠ 本次分析存在降级：")
        for d in final["degraded"]:
            print(f"  - {d}")
    if llm_summary["calls"]:
        tokens = llm_summary["total_tokens"]
        # 接口未返回用量时显示「未知」，而不是显示 0（避免误读为零消耗）
        token_text = f"{tokens} tokens" if tokens is not None else "token 用量未返回"
        print(
            f"[LLM] 调用 {llm_summary['calls']} 次"
            f"（失败 {llm_summary['failed']} 次），{token_text}"
        )

    if persistence:
        # 图执行完毕（正常或提前结束）：关闭 run，按最终状态写 completed/failed
        done = final["status"] == "done"
        persistence.complete(
            "completed" if done else "failed",
            step_count=final_step_count,
            error=None if done else "Agent 未正常完成",
            llm_calls=llm_summary,
        )

    return final


def _read_dataset_or_exit(dataset_path: str):
    """读取数据集做前置校验；失败时打印原因并以退出码 1 结束进程。

    提前失败可以避免 Agent 跑到工具节点才报错——那时用户已经等了一轮
    LLM 调用，体验更差。

    :param dataset_path: 用户输入的数据集路径
    :return: 加载好的 DataFrame
    """
    try:
        df = load_dataframe(dataset_path)
    except Exception as e:
        print(f"✗ 无法读取数据集: {e}")
        sys.exit(1)
    print(f"\n已加载数据集: {dataset_path}（{len(df)} 行 × {df.shape[1]} 列）")
    print("-" * 60)
    return df


def _ask_dataset_path() -> str:
    """交互式询问数据集路径（回车使用默认样例数据）。"""
    dataset_path = input("请输入数据文件路径：\n> ").strip()
    if not dataset_path:
        dataset_path = "datasets/sales.csv"
        print(f"(使用默认数据集: {dataset_path})")
    return dataset_path


def _run_turn(
    dataset_path: str,
    request: str,
    persistence: PersistenceService,
    session_id: str,
    df,
    memory: ConversationMemory,
) -> dict[str, Any]:
    """执行一轮完整的分析：建 run → 跑图 → 记录对话历史。

    抽成独立函数是因为连续对话下这段逻辑会被反复执行；
    每轮都会新建一条 task/run 记录，但共用同一个 session_id，
    因此数据库里既能看到「同一会话的多次追问」，又能单独追溯每一次运行。

    :param dataset_path: 数据文件路径
    :param request: 本轮的用户需求
    :param persistence: 持久化服务（内部已处理降级）
    :param session_id: 会话 ID（多轮共用）
    :param df: 已加载的数据集（用于登记行列数）
    :param memory: 对话记忆（本轮结束后写入）
    :return: run_agent 的结果字典
    """
    # 建立本轮的 task/run 记录；session_id 保持不变以延续会话
    ids = persistence.start_run(
        dataset_path,
        request,
        session_id=session_id,
        dataset_info={
            "row_count": len(df),
            "column_count": df.shape[1],
            "columns": list(df.columns),
        },
    )
    if ids["task_id"] is not None:
        print(f"[Task] {ids['task_id']}  [Agent Run] {ids['run_id']}")
    print("-" * 60)

    # 连续对话：取同一数据集的历史结论摘要，让「为什么？」能接上上一轮
    context = memory.build_context(request, dataset_path)
    if context:
        print("[记忆] 已注入上一轮分析结论作为上下文")

    try:
        # 核心执行：流式跑图，节点进度在 run_agent 内部实时打印
        result = run_agent(
            dataset_path, request, persistence=persistence, conversation_context=context
        )
    except KeyboardInterrupt:
        # 用户主动中断：约定使用 130（128 + SIGINT 信号编号 2）
        print("\n已中断。")
        sys.exit(130)
    except Exception as e:
        # 运行期异常：标记 run 失败后以退出码 1 结束
        print(f"\n✗ Agent 运行失败: {type(e).__name__}: {e}")
        persistence.complete("failed", error=str(e))
        sys.exit(1)

    # 记录本轮问答供后续追问使用（只存结论摘要，不存完整报告）
    memory.remember(
        request, dataset_path, result.get("insights"), result.get("report_path", "")
    )
    return result


def main() -> None:
    """CLI 入口：交互式收集输入并驱动 Agent 分析，支持连续追问。

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

    dataset_path = _ask_dataset_path()

    # 交互输入 2/2：分析需求；为空无法继续，属于明确的退出条件
    request = input("\n请输入你的分析需求：\n> ").strip()
    if not request:
        print("分析需求不能为空。")
        sys.exit(1)

    # 校验数据集可读：提前失败，避免 Agent 跑到工具节点才报错
    df = _read_dataset_or_exit(dataset_path)

    # 启动持久化（PostgreSQL + Redis，不可用时优雅降级）
    persistence = PersistenceService()
    # 会话 ID 在本次进程内固定，多轮追问共用；Redis 不可用时记忆退回进程内
    session_id = persistence.start_run(
        dataset_path,
        request,
        dataset_info={"row_count": len(df), "column_count": df.shape[1], "columns": list(df.columns)},
    )["session_id"]
    print(f"[Session] {session_id}")
    # 新建会话后清空可能残留的历史轮次，避免复用旧 session_id 时串入无关上下文
    memory = ConversationMemory(session_id)
    memory.clear()

    result = _run_turn(dataset_path, request, persistence, session_id, df, memory)
    _print_turn_result(result, dataset_path)

    # ---- 连续追问循环 ----
    while True:
        try:
            follow_up = input(
                "\n继续追问（直接输入新问题；/new 更换数据集；/exit 退出）：\n> "
            ).strip()
        except EOFError:
            # 非交互环境（管道输入结束）：正常退出
            break
        except KeyboardInterrupt:
            print("\n已中断。")
            sys.exit(130)

        if not follow_up or follow_up.lower() in EXIT_COMMANDS:
            break

        if follow_up.lower() in NEW_DATASET_COMMANDS:
            # 更换数据集：历史记忆按数据集隔离，无需手工清理，直接重问路径即可
            dataset_path = _ask_dataset_path()
            df = _read_dataset_or_exit(dataset_path)
            print("(已切换数据集；历史上下文按数据集隔离，不会串入本轮分析)")
            continue

        result = _run_turn(dataset_path, follow_up, persistence, session_id, df, memory)
        _print_turn_result(result, dataset_path)

    print("\n再见。")


def _print_turn_result(result: dict[str, Any], dataset_path: str) -> None:
    """打印单轮分析的收尾提示（成功路径 + 失败告警）。

    :param result: run_agent 的结果字典
    :param dataset_path: 本轮数据集路径（失败提示里回显，便于确认作用对象）
    """
    print("\n" + "=" * 60)
    if result["status"] == "done" and result["report_path"]:
        print(f"✅ 分析完成，报告已保存到：{result['report_path']}")
    else:
        print(f"⚠ 分析未正常完成（数据集：{dataset_path}），请检查上方日志。")
    print("=" * 60)
