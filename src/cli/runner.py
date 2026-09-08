"""CLI 运行器：交互式输入 + Agent 执行 + 进度展示 + 持久化。"""

from __future__ import annotations

import sys
from typing import Any

from src.agent.graph import build_graph, make_initial_state
from src.config.settings import get_settings
from src.db.service import PersistenceService, tool_type_of
from src.logging_config import setup_logging
from src.tools.file_tools import load_dataframe

BANNER = """========================================
       AI Data Analyst Agent
========================================"""

# 节点 → 进度标签
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
    """从 agent 节点的消息中提取并打印将要调用的工具名。"""
    messages = update.get("messages") or []
    names = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            names.append(tc.get("name", ""))
    if names:
        print("  → 调用工具: " + ", ".join(names))


def _print_tool_results(update: dict[str, Any]) -> None:
    """打印工具执行结果摘要。"""
    results = update.get("tool_results") or []
    for r in results:
        status = r.get("status", "?")
        name = r.get("name", "?")
        result = str(r.get("result", ""))
        short = result[:200].replace("\n", " ").strip()
        mark = "✓" if status == "success" else "✗"
        print(f"  {mark} [{name}] {short}")


def _persist_update(persistence: PersistenceService, node_name: str, update: dict[str, Any]) -> None:
    """把节点更新写入持久化（task/run/tool_call/result/report/session）。"""
    persistence.update_status(NODE_STATUS.get(node_name, node_name), step=update.get("step_count"))

    if node_name == "tools":
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
        persistence.record_plan(None, update.get("selected_skills"))
    elif node_name == "planner":
        persistence.record_plan(update.get("analysis_plan"))
    elif node_name == "insight":
        persistence.record_result("insight", {"insights": update.get("insights", [])})
    elif "final_report" in update:
        persistence.record_report(update.get("report_path"), update.get("final_report"))


def run_agent(
    dataset_path: str,
    request: str,
    persistence: PersistenceService | None = None,
) -> dict[str, Any]:
    """执行 Agent，返回最终结果（final_report / report_path / status）。"""
    graph = build_graph()
    initial = make_initial_state(dataset_path, request)

    final: dict[str, Any] = {"final_report": "", "report_path": "", "status": "error"}
    final_step_count = 0

    for chunk in graph.stream(initial, stream_mode="updates"):
        for node_name, update in chunk.items():
            label = NODE_LABELS.get(node_name, f"[{node_name}]")
            print(f"\n{label}")

            if node_name == "agent":
                _print_tool_names(update)
            elif node_name == "tools":
                _print_tool_results(update)
            elif node_name == "skill_selection":
                selected = update.get("selected_skills") or []
                print(f"  已选择 Skills: {', '.join(selected) if selected else '（无）'}")
            elif node_name == "profiler":
                obs = (update.get("observations") or [""])[0]
                print(f"  {obs.splitlines()[0] if obs else ''}")

            if "step_count" in update:
                final_step_count = update["step_count"]

            if persistence:
                _persist_update(persistence, node_name, update)

            if "final_report" in update:
                final["final_report"] = update["final_report"]
                final["report_path"] = update.get("report_path", "")
                final["status"] = update.get("status", "done")

    if persistence:
        done = final["status"] == "done"
        persistence.complete(
            "completed" if done else "failed",
            step_count=final_step_count,
            error=None if done else "Agent 未正常完成",
        )

    return final


def main() -> None:
    """CLI 入口。"""
    setup_logging()
    print(BANNER)
    print()

    settings = get_settings()

    if not settings.llm_api_key:
        print("⚠ 未检测到 LLM_API_KEY。")
        print("  请复制 .env.example 为 .env 并填写 API Key 后重试。")
        sys.exit(1)

    dataset_path = input("请输入数据文件路径：\n> ").strip()
    if not dataset_path:
        dataset_path = "datasets/sales.csv"
        print(f"(使用默认数据集: {dataset_path})")

    request = input("\n请输入你的分析需求：\n> ").strip()
    if not request:
        print("分析需求不能为空。")
        sys.exit(1)

    # 校验数据集可读
    try:
        df = load_dataframe(dataset_path)
    except Exception as e:
        print(f"✗ 无法读取数据集: {e}")
        sys.exit(1)

    print(f"\n已加载数据集: {dataset_path}（{len(df)} 行 × {df.shape[1]} 列）")
    print("-" * 60)

    # 启动持久化（PostgreSQL + Redis，不可用时优雅降级）
    persistence = PersistenceService()
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
        result = run_agent(dataset_path, request, persistence=persistence)
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Agent 运行失败: {type(e).__name__}: {e}")
        persistence.complete("failed", error=str(e))
        sys.exit(1)

    print("\n" + "=" * 60)
    if result["status"] == "done" and result["report_path"]:
        print(f"✅ 分析完成，报告已保存到：{result['report_path']}")
        if ids["task_id"] is not None:
            print("Task saved / Report saved / Analysis results saved")
    else:
        print("⚠ 分析未正常完成，请检查上方日志。")
    print("=" * 60)
