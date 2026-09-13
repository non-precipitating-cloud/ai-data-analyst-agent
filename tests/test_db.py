"""数据库 ORM / Repository 数据访问层测试（src.db，使用 SQLite 测试库）。

借助 conftest 的 sqlite_db 夹具（每个用例独立的临时 SQLite 库）覆盖：
- 引擎可用性判定与建表完整性；
- 数据集 / 分析任务 / Agent 运行记录的创建与回读；
- 数据集按路径去重、任务与运行的状态更新（含完成时间、错误信息）；
- 工具调用（本地 + MCP）、分析结果、报告的持久化；
- 按 session_id 反查运行记录。
"""

from __future__ import annotations

from sqlalchemy import inspect

from src.db.database import get_engine, is_db_available
from src.db.repository import Repository

# 业务期望存在的 6 张核心表
EXPECTED_TABLES = {
    "datasets",
    "analysis_tasks",
    "agent_runs",
    "tool_calls",
    "analysis_results",
    "reports",
}


def test_db_available(sqlite_db) -> None:
    """配置好 SQLite 引擎后，数据库可用性检查应返回 True。"""
    assert is_db_available() is True


def test_tables_created(sqlite_db) -> None:
    """init_db 建表后，6 张核心表应全部存在。"""
    # 通过 SQLAlchemy 反射出实际建好的表名集合
    tables = set(inspect(get_engine()).get_table_names())
    assert EXPECTED_TABLES <= tables


def test_create_dataset_task_run(sqlite_db) -> None:
    """级联创建 数据集→任务→运行记录，返回主键均为正且关联字段可正确回读。"""
    r = Repository()
    # 依次登记数据集、创建分析任务、开启一次 Agent 运行
    did = r.create_dataset("sales.csv", "datasets/sales.csv", "csv", 2160, 12)
    tid = r.create_task(did, "分析销售额", "running")
    rid = r.create_run(tid, "sess-1")
    assert did > 0 and tid > 0 and rid > 0
    # 回读任务与运行，验证用户需求和会话 ID 等关联字段
    assert r.get_task(tid).user_request == "分析销售额"
    assert r.get_run(rid).session_id == "sess-1"


def test_dataset_dedup_by_path(sqlite_db) -> None:
    """相同文件路径重复登记数据集时应复用已有记录（返回相同主键）。"""
    r = Repository()
    a = r.create_dataset("sales.csv", "datasets/sales.csv", "csv", 2160, 12)
    b = r.create_dataset("sales.csv", "datasets/sales.csv", "csv", 2160, 12)
    assert a == b  # 相同路径复用


def test_task_persistence_status(sqlite_db) -> None:
    """任务更新为 completed 时应同步写入完成时间 completed_at。"""
    r = Repository()
    tid = r.create_task(None, "分析", "pending")
    r.update_task(tid, status="completed")
    task = r.get_task(tid)
    assert task.status == "completed"
    assert task.completed_at is not None


def test_run_persistence(sqlite_db) -> None:
    """运行记录的状态、执行步数、错误信息应能更新并完整回读。"""
    r = Repository()
    rid = r.create_run(None, "sess-2")
    # 模拟一次失败运行：落状态、步数与错误原因
    r.update_run(rid, status="failed", step_count=7, error_message="boom")
    run = r.get_run(rid)
    assert run.status == "failed"
    assert run.step_count == 7
    assert run.error_message == "boom"


def test_tool_call_local_and_mcp(sqlite_db) -> None:
    """同一次运行中的本地工具与 MCP 工具调用都应按顺序持久化，并保留类型区分。"""
    r = Repository()
    rid = r.create_run(None, "sess-3")
    # 分别写入一条本地调用和一条 MCP 调用（含入参与 JSON 结果）
    r.add_tool_call(rid, "read_dataset", "local", {"path": "x"}, '{"num_rows":1}', "success")
    r.add_tool_call(rid, "mcp__detect_outliers", "mcp", {"path": "x"}, '{"outlier_count":47}', "success")
    calls = r.list_tool_calls(rid)
    # 工具名按写入顺序返回
    assert [c.tool_name for c in calls] == ["read_dataset", "mcp__detect_outliers"]
    # 工具类型正确标记为 local / mcp
    assert [c.tool_type for c in calls] == ["local", "mcp"]


def test_analysis_result_persistence(sqlite_db) -> None:
    """同一任务下多条不同类型的分析结果应按写入顺序持久化。"""
    r = Repository()
    tid = r.create_task(None, "分析", "running")
    # 一条异常检测结果 + 一条图表产物结果
    r.add_result(tid, "outlier", {"count": 47})
    r.add_result(tid, "chart", {"path": "x.png"})
    assert [x.result_type for x in r.list_results(tid)] == ["outlier", "chart"]


def test_report_persistence(sqlite_db) -> None:
    """报告路径与 Markdown 正文应能保存并按任务 ID 取回。"""
    r = Repository()
    tid = r.create_task(None, "分析", "running")
    r.add_report(tid, "reports/a.md", "# 报告")
    rep = r.get_report(tid)
    assert rep.report_content == "# 报告"
    assert rep.report_path == "reports/a.md"


def test_get_run_by_session(sqlite_db) -> None:
    """按会话 ID 应能反查到对应的运行记录。"""
    r = Repository()
    r.create_run(None, "sess-x")
    run = r.get_run_by_session("sess-x")
    assert run is not None
    assert run.session_id == "sess-x"
