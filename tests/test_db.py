"""数据库 ORM / Repository 测试（SQLite 测试库）。"""

from __future__ import annotations

from sqlalchemy import inspect

from src.db.database import get_engine, is_db_available
from src.db.repository import Repository

EXPECTED_TABLES = {
    "datasets",
    "analysis_tasks",
    "agent_runs",
    "tool_calls",
    "analysis_results",
    "reports",
}


def test_db_available(sqlite_db) -> None:
    assert is_db_available() is True


def test_tables_created(sqlite_db) -> None:
    tables = set(inspect(get_engine()).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_create_dataset_task_run(sqlite_db) -> None:
    r = Repository()
    did = r.create_dataset("sales.csv", "datasets/sales.csv", "csv", 2160, 12)
    tid = r.create_task(did, "分析销售额", "running")
    rid = r.create_run(tid, "sess-1")
    assert did > 0 and tid > 0 and rid > 0
    assert r.get_task(tid).user_request == "分析销售额"
    assert r.get_run(rid).session_id == "sess-1"


def test_dataset_dedup_by_path(sqlite_db) -> None:
    r = Repository()
    a = r.create_dataset("sales.csv", "datasets/sales.csv", "csv", 2160, 12)
    b = r.create_dataset("sales.csv", "datasets/sales.csv", "csv", 2160, 12)
    assert a == b  # 相同路径复用


def test_task_persistence_status(sqlite_db) -> None:
    r = Repository()
    tid = r.create_task(None, "分析", "pending")
    r.update_task(tid, status="completed")
    task = r.get_task(tid)
    assert task.status == "completed"
    assert task.completed_at is not None


def test_run_persistence(sqlite_db) -> None:
    r = Repository()
    rid = r.create_run(None, "sess-2")
    r.update_run(rid, status="failed", step_count=7, error_message="boom")
    run = r.get_run(rid)
    assert run.status == "failed"
    assert run.step_count == 7
    assert run.error_message == "boom"


def test_tool_call_local_and_mcp(sqlite_db) -> None:
    r = Repository()
    rid = r.create_run(None, "sess-3")
    r.add_tool_call(rid, "read_dataset", "local", {"path": "x"}, '{"num_rows":1}', "success")
    r.add_tool_call(rid, "mcp__detect_outliers", "mcp", {"path": "x"}, '{"outlier_count":47}', "success")
    calls = r.list_tool_calls(rid)
    assert [c.tool_name for c in calls] == ["read_dataset", "mcp__detect_outliers"]
    assert [c.tool_type for c in calls] == ["local", "mcp"]


def test_analysis_result_persistence(sqlite_db) -> None:
    r = Repository()
    tid = r.create_task(None, "分析", "running")
    r.add_result(tid, "outlier", {"count": 47})
    r.add_result(tid, "chart", {"path": "x.png"})
    assert [x.result_type for x in r.list_results(tid)] == ["outlier", "chart"]


def test_report_persistence(sqlite_db) -> None:
    r = Repository()
    tid = r.create_task(None, "分析", "running")
    r.add_report(tid, "reports/a.md", "# 报告")
    rep = r.get_report(tid)
    assert rep.report_content == "# 报告"
    assert rep.report_path == "reports/a.md"


def test_get_run_by_session(sqlite_db) -> None:
    r = Repository()
    r.create_run(None, "sess-x")
    run = r.get_run_by_session("sess-x")
    assert run is not None
    assert run.session_id == "sess-x"
