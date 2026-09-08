"""PersistenceService 测试：持久化编排 + 优雅降级 + 完整 Agent 集成。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state
from src.cli.runner import run_agent
from src.db.service import PersistenceService, result_type_of, tool_type_of
from tests.conftest import SALES_CSV

_DATASET_INFO = {"row_count": 2160, "column_count": 12, "columns": ["sales", "region"]}


def test_tool_type_of() -> None:
    assert tool_type_of("read_dataset") == "local"
    assert tool_type_of("mcp__read_dataset") == "mcp"


def test_result_type_of() -> None:
    assert result_type_of("detect_outliers") == "outlier"
    assert result_type_of("mcp__detect_outliers") == "outlier"
    assert result_type_of("calculate_correlation") == "correlation"
    assert result_type_of("execute_python") == "other"


def test_start_run_and_complete(sqlite_db, fake_redis) -> None:
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析销售额", dataset_info=_DATASET_INFO)
    assert ids["session_id"]
    assert ids["task_id"] is not None
    assert ids["run_id"] is not None
    assert svc.repo.get_task(ids["task_id"]).status == "running"

    svc.complete("completed", step_count=5)
    assert svc.repo.get_task(ids["task_id"]).status == "completed"
    assert svc.repo.get_run(ids["run_id"]).step_count == 5


def test_record_tool_call_local_and_mcp(sqlite_db, fake_redis) -> None:
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)
    svc.record_tool_call("read_dataset", "local", {"path": "x"}, '{"num_rows":2160}', "success")
    svc.record_tool_call("mcp__detect_outliers", "mcp", {"path": "x"}, '{"outlier_count":47}', "success")

    calls = svc.repo.list_tool_calls(ids["run_id"])
    assert [c.tool_type for c in calls] == ["local", "mcp"]
    # detect_outliers 归类为 outlier 结果
    assert "outlier" in [r.result_type for r in svc.repo.list_results(ids["task_id"])]


def test_record_report(sqlite_db, fake_redis) -> None:
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)
    svc.record_report("reports/a.md", "# 报告")
    assert svc.repo.get_report(ids["task_id"]).report_content == "# 报告"


def test_db_unavailable_degradation(monkeypatch, fake_redis) -> None:
    monkeypatch.setattr("src.db.service.is_db_available", lambda: False)
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析")
    assert ids["task_id"] is None  # DB 不可用，不创建 task/run
    # 以下调用不应崩溃
    svc.record_tool_call("x", "local", {}, "r", "success")
    svc.record_result("outlier", {})
    svc.record_report("p", "c")
    svc.complete("completed")


class FakeChatModel:
    def __init__(self) -> None:
        self.agent_calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        first = messages[0].content if messages else ""
        if "技能选择器" in first:
            return AIMessage(content="[]")
        if "自主的数据分析智能体" in first:
            return self._agent_response()
        if "规划者" in first:
            return AIMessage(content='[{"step":1,"goal":"检测异常","tool":"detect_outliers","note":"zscore"}]')
        if "洞察专家" in first:
            return AIMessage(content="洞察1：检测到异常值。")
        if "报告撰写" in first:
            return AIMessage(content="# 报告\n\n## 图表说明\n本次分析未生成图表。")
        if "理解" in first:
            return AIMessage(content="目标：检测异常值。")
        return AIMessage(content="ok")

    def _agent_response(self) -> AIMessage:
        self.agent_calls += 1
        if self.agent_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "detect_outliers",
                        "args": {"path": SALES_CSV, "column": "sales", "method": "zscore"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="结论：检测到异常值。")


def _patch_all_llm(monkeypatch, fake) -> None:
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_full_agent_with_persistence(sqlite_db, fake_redis, monkeypatch) -> None:
    fake = FakeChatModel()
    _patch_all_llm(monkeypatch, fake)

    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "检测异常值", dataset_info=_DATASET_INFO)

    result = run_agent(SALES_CSV, "检测异常值", persistence=svc)

    assert result["status"] == "done"

    # 持久化结果
    assert svc.repo.get_task(ids["task_id"]).status == "completed"
    calls = svc.repo.list_tool_calls(ids["run_id"])
    assert [c.tool_name for c in calls] == ["detect_outliers"]
    assert [c.tool_type for c in calls] == ["local"]
    assert "outlier" in [r.result_type for r in svc.repo.list_results(ids["task_id"])]
    assert svc.repo.get_report(ids["task_id"]) is not None

    Path(result["report_path"]).unlink(missing_ok=True)
