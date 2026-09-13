"""SQL 只读守卫测试（src.tools.sql_tool.sql_safety_error）。

该函数用于在执行 SQL 前做静态安全把关：
- 只读语句（SELECT / WITH ... SELECT / EXPLAIN）返回 None 表示放行；
- 任何写操作或 DDL（DROP/DELETE/UPDATE/INSERT）、多语句拼接、空 SQL
  都应返回非 None 的错误信息，从而拒绝执行。
"""

from __future__ import annotations

from src.tools.sql_tool import _with_row_guard, sql_safety_error


def test_allows_select() -> None:
    """普通 SELECT 查询应被放行（返回 None）。"""
    assert sql_safety_error("SELECT * FROM sales") is None


def test_allows_with() -> None:
    """WITH 开头的只读 CTE 查询应被放行。"""
    assert sql_safety_error("WITH x AS (SELECT 1) SELECT * FROM x") is None


def test_allows_explain() -> None:
    """EXPLAIN 仅查看执行计划，应被放行。"""
    assert sql_safety_error("EXPLAIN SELECT * FROM sales") is None


def test_rejects_drop() -> None:
    """DROP TABLE 属于破坏性 DDL，必须拒绝。"""
    assert sql_safety_error("DROP TABLE sales") is not None


def test_rejects_delete() -> None:
    """DELETE 删除数据，必须拒绝。"""
    assert sql_safety_error("DELETE FROM sales") is not None


def test_rejects_update() -> None:
    """UPDATE 修改数据，必须拒绝。"""
    assert sql_safety_error("UPDATE sales SET a = 1") is not None


def test_rejects_insert() -> None:
    """INSERT 写入数据，必须拒绝。"""
    assert sql_safety_error("INSERT INTO sales VALUES (1)") is not None


def test_rejects_multi_statement() -> None:
    """用分号拼接第二条危险语句时必须整段拒绝，防止绕过只读限制。"""
    assert sql_safety_error("SELECT 1; DROP TABLE x") is not None


def test_rejects_empty() -> None:
    """空 SQL 没有意义，应返回错误信息而不是放行。"""
    assert sql_safety_error("") is not None


def test_row_guard_wraps_select_without_limit() -> None:
    """没有 LIMIT 的 SELECT 应被外包一层行数上限，防止整表读入内存。"""
    guarded = _with_row_guard("SELECT * FROM sales")
    assert guarded.startswith("SELECT * FROM (SELECT * FROM sales)")
    assert "LIMIT 1000" in guarded


def test_row_guard_keeps_explicit_limit() -> None:
    """已显式书写 LIMIT 的查询应原样保留，不重复加码。"""
    sql = "SELECT * FROM sales LIMIT 5"
    assert _with_row_guard(sql) == sql


def test_row_guard_skips_explain() -> None:
    """EXPLAIN 只返回执行计划文本，不应套用子查询行数包装。"""
    sql = "EXPLAIN SELECT * FROM sales"
    assert _with_row_guard(sql) == sql


def test_row_guard_wraps_cte() -> None:
    """WITH 开头的 CTE 查询同样需要行数保护。"""
    guarded = _with_row_guard("WITH x AS (SELECT 1) SELECT * FROM x")
    assert "LIMIT 1000" in guarded
