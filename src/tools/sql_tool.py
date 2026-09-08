"""SQL 工具：PostgreSQL 只读执行。

安全策略：默认只允许 SELECT / WITH / EXPLAIN，禁止 DROP/DELETE/UPDATE/INSERT/ALTER 等。
在将 SQL 发送到数据库前做静态校验。
"""

from __future__ import annotations

import re

import pandas as pd
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

from src.config.settings import get_settings

# 危险关键字（只要出现即拒绝，宁可误伤也不放行）
DANGEROUS_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "MERGE", "REPLACE", "CALL", "COPY", "VACUUM",
    "ATTACH", "DETACH", "REINDEX", "LOCK", "COMMENT", "RENAME",
}

# 允许的开头关键字
ALLOWED_LEADING = {"SELECT", "WITH", "EXPLAIN"}


def sql_safety_error(sql: str) -> str | None:
    """返回安全违规原因；安全则返回 None。"""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return "SQL 为空"

    # 多语句拒绝（除结尾分号外）
    statements = [s.strip() for s in stripped.split(";") if s.strip()]
    if len(statements) != 1:
        return "禁止一次执行多条语句"

    statement = statements[0]

    # 开头关键字必须是 SELECT / WITH / EXPLAIN
    leading = re.match(r"\s*([A-Za-z]+)", statement)
    if not leading or leading.group(1).upper() not in ALLOWED_LEADING:
        return f"仅允许 SELECT / WITH / EXPLAIN 开头的只读查询，收到: {leading.group(1) if leading else '?'}"

    # 全文检测危险关键字（按词边界匹配）
    for kw in DANGEROUS_KEYWORDS:
        if re.search(rf"\b{kw}\b", statement, flags=re.IGNORECASE):
            return f"检测到危险关键字: {kw}"

    return None


def run_sql(sql: str) -> pd.DataFrame:
    """执行只读 SQL 并返回 DataFrame（数据库不可用时抛异常）。"""
    error = sql_safety_error(sql)
    if error:
        raise ValueError(f"SQL 被安全策略拒绝：{error}")

    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn)
    finally:
        engine.dispose()


@tool
def execute_sql(sql: str) -> str:
    """在 PostgreSQL 上执行只读 SQL 查询（仅允许 SELECT / WITH / EXPLAIN）。

    返回查询结果。若返回行数过多会自动截断。数据库不可用时返回错误提示。
    """
    error = sql_safety_error(sql)
    if error:
        return f"SQL 被安全策略拒绝：{error}"

    try:
        df = run_sql(sql)
    except Exception as e:  # 数据库未启动 / 连接失败等
        return f"SQL 执行失败（数据库可能未启动）：{type(e).__name__}: {e}"

    settings = get_settings()
    result = {
        "num_rows": int(len(df)),
        "columns": list(df.columns),
        "data": df.head(50).to_dict(orient="records"),
    }
    import json

    return json.dumps(result, ensure_ascii=False, default=str)[: settings.output_truncate_chars]
