"""SQL 工具：PostgreSQL 只读查询执行。

输入输出契约：
- 输入：一条 SELECT / WITH / EXPLAIN 只读 SQL 字符串（禁止多语句）。
- 输出（@tool 层）：JSON 字符串 {num_rows, columns, data(前 50 行 records)}，
  并按配置截断总长度；安全违规、数据库不可用均返回中文提示，不抛异常。
- run_sql() 底层返回 pd.DataFrame，供节点/测试直接使用。

安全策略（纵深防御，共四层）：
1. 静态校验：默认只允许 SELECT / WITH / EXPLAIN，禁止 DROP/DELETE/UPDATE/INSERT/ALTER 等；
2. 数据库会话级只读：PostgreSQL 连接强制 default_transaction_read_only=on，
   即使绕过关键字过滤（如调用有写权限的函数），数据库也拒绝写入；
3. 语句超时：statement_timeout=10s，慢查询由数据库主动取消，防拖垮整库；
4. 行数上限：无 LIMIT 的查询自动外包 LIMIT 1000，防止整表读入内存导致 OOM。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config.settings import get_settings

# 危险关键字（只要出现即拒绝，宁可误伤也不放行）
DANGEROUS_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "MERGE", "REPLACE", "CALL", "COPY", "VACUUM",
    "ATTACH", "DETACH", "REINDEX", "LOCK", "COMMENT", "RENAME",
}

# 允许的开头关键字
ALLOWED_LEADING = {"SELECT", "WITH", "EXPLAIN"}

# 单条查询最多拉取的行数（防止 SELECT * 大表把整个结果集读进内存导致 OOM）；
# 工具层本来就只把前 50 行回传 LLM，这里是数据库侧的硬上限
SQL_MAX_ROWS = 1000

# PostgreSQL 语句级超时（毫秒）：超时后数据库主动取消查询，避免慢查询拖垮整库
SQL_STATEMENT_TIMEOUT_MS = 10000


def sql_safety_error(sql: str) -> str | None:
    """对 SQL 做静态安全校验。

    参数：
        sql: 待执行的 SQL 文本。
    返回值：
        str | None: 有风险时返回中文违规原因；通过全部检查时返回 None。
    """
    # 去掉首尾空白与结尾分号，便于后续判断
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return "SQL 为空"

    # 多语句拒绝（除结尾分号外）：防止 "SELECT ...; DROP TABLE ..." 式注入
    statements = [s.strip() for s in stripped.split(";") if s.strip()]
    if len(statements) != 1:
        return "禁止一次执行多条语句"

    statement = statements[0]

    # 开头关键字必须是 SELECT / WITH / EXPLAIN（挡住 SET/PRAGMA/DO 等）
    leading = re.match(r"\s*([A-Za-z]+)", statement)
    if not leading or leading.group(1).upper() not in ALLOWED_LEADING:
        return f"仅允许 SELECT / WITH / EXPLAIN 开头的只读查询，收到: {leading.group(1) if leading else '?'}"

    # 全文检测危险关键字（按词边界匹配，忽略大小写）；
    # 即使在 CTE/子查询/字符串外的任何位置出现都拒绝，采取宁枉毋纵策略
    for kw in DANGEROUS_KEYWORDS:
        if re.search(rf"\b{kw}\b", statement, flags=re.IGNORECASE):
            return f"检测到危险关键字: {kw}"

    return None


def _with_row_guard(sql: str) -> str:
    """给没有 LIMIT 的查询包一层行数上限，防止整表拉取导致内存溢出。

    规则：
    - EXPLAIN 语句不包装（它只返回执行计划文本，且语法上不能套子查询限制行数）；
    - 已显式书写 LIMIT 的查询尊重作者意图，不再加码；
    - 其余 SELECT/WITH 查询包装为 ``SELECT * FROM (<原 SQL>) AS _row_guard LIMIT N``。

    参数：
        sql: 已通过静态校验、去掉结尾分号的单条只读 SQL。
    返回值：
        带行数保护的 SQL 字符串。
    """
    if re.match(r"\s*explain\b", sql, flags=re.IGNORECASE):
        return sql
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        return sql
    return f"SELECT * FROM ({sql}) AS _row_guard LIMIT {SQL_MAX_ROWS}"


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    """返回进程级复用的 SQLAlchemy 引擎（首次调用时创建，之后走连接池）。

    之前每次查询都 create_engine + dispose，建连开销大且无法复用连接池；
    改为单例缓存后，多次工具调用共享同一连接池。

    返回值：
        Engine: 配置好的数据库引擎。PostgreSQL 额外通过 libpq 的 options
        在数据库会话级强制「只读事务 + 语句超时」，这是比关键字过滤更硬的保障：
        即使绕过静态校验（如调用有写权限的函数），数据库也会拒绝写入。
    """
    settings = get_settings()
    url = settings.database_url
    connect_args: dict = {}
    if url.startswith(("postgresql://", "postgresql+")):
        # default_transaction_read_only=on：该连接上的所有事务默认只读；
        # statement_timeout：单条语句执行超过阈值即被数据库取消
        connect_args["options"] = (
            f"-c default_transaction_read_only=on "
            f"-c statement_timeout={SQL_STATEMENT_TIMEOUT_MS}"
        )
    # pool_pre_ping：取连接前先探活，避免拿到数据库重启后的失效连接
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def run_sql(sql: str) -> pd.DataFrame:
    """执行只读 SQL 并返回 DataFrame（数据库不可用时抛异常）。

    参数：
        sql: 待执行的 SQL 文本（函数内会再次执行安全校验）。
    返回值：
        pd.DataFrame: 查询结果集（受 SQL_MAX_ROWS 行数上限保护）。
    异常：
        ValueError: SQL 未通过安全策略。
        SQLAlchemyError / OperationalError: 连接或执行失败时由底层抛出。
    """
    error = sql_safety_error(sql)
    if error:
        raise ValueError(f"SQL 被安全策略拒绝：{error}")

    # 与 sql_safety_error 相同的规范化：去空白、去结尾分号，保证只执行单条语句
    stripped = sql.strip().rstrip(";").strip()

    # 复用全局引擎（连接池），而非每次新建/销毁
    engine = _get_engine()
    with engine.connect() as conn:
        # text() 构造合法 SQL 表达式；外包行数上限后由 read_sql_query 拉成 DataFrame
        return pd.read_sql_query(text(_with_row_guard(stripped)), conn)


@tool
def execute_sql(sql: str) -> str:
    """在 PostgreSQL 上执行只读 SQL 查询（仅允许 SELECT / WITH / EXPLAIN）。

    返回查询结果。若返回行数过多会自动截断。数据库不可用时返回错误提示。

    返回值：
        str: {num_rows, columns, data} 的 JSON 字符串（data 仅含前 50 行，
        整体再按 output_truncate_chars 截断）；安全或连接问题返回提示文本。
    """
    # 工具层先做一次快速拦截，避免连数据库都不创建就直接拒绝
    error = sql_safety_error(sql)
    if error:
        return f"SQL 被安全策略拒绝：{error}"

    try:
        df = run_sql(sql)
    except Exception as e:  # 数据库未启动 / 连接失败等
        return f"SQL 执行失败（数据库可能未启动）：{type(e).__name__}: {e}"

    settings = get_settings()
    # 组装轻量结果：总行数 + 列名 + 前 50 行数据，控制喂给 LLM 的体量
    # 注意 num_rows 受 SQL_MAX_ROWS 上限保护，data 再只取前 50 行
    result = {
        "num_rows": int(len(df)),
        "columns": list(df.columns),
        "data": df.head(50).to_dict(orient="records"),
    }

    # default=str 兜底日期/Decimal 等类型；最终再做一次硬截断
    return json.dumps(result, ensure_ascii=False, default=str)[: settings.output_truncate_chars]
