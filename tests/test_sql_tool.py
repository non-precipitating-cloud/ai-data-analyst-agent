"""SQL 只读守卫测试。"""

from __future__ import annotations

from src.tools.sql_tool import sql_safety_error


def test_allows_select() -> None:
    assert sql_safety_error("SELECT * FROM sales") is None


def test_allows_with() -> None:
    assert sql_safety_error("WITH x AS (SELECT 1) SELECT * FROM x") is None


def test_allows_explain() -> None:
    assert sql_safety_error("EXPLAIN SELECT * FROM sales") is None


def test_rejects_drop() -> None:
    assert sql_safety_error("DROP TABLE sales") is not None


def test_rejects_delete() -> None:
    assert sql_safety_error("DELETE FROM sales") is not None


def test_rejects_update() -> None:
    assert sql_safety_error("UPDATE sales SET a = 1") is not None


def test_rejects_insert() -> None:
    assert sql_safety_error("INSERT INTO sales VALUES (1)") is not None


def test_rejects_multi_statement() -> None:
    assert sql_safety_error("SELECT 1; DROP TABLE x") is not None


def test_rejects_empty() -> None:
    assert sql_safety_error("") is not None
