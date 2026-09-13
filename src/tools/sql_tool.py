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
from src.tools.errors import (
    KIND_DB_UNAVAILABLE,
    KIND_INVALID_ARGUMENT,
    KIND_SECURITY,
    tool_error,
)


# 「SQL 本身写错了」的错误信息特征（跨方言，不做数据库专属假设）：
# PostgreSQL 与 SQLite 对未定义列/表、语法错误的措辞不同，但都落在这几类里。
_SQL_MISTAKE_PATTERNS = (
    "no such column", "no such table",          # SQLite
    "does not exist",                            # PostgreSQL：relation/column does not exist
    "undefined column", "undefined table",
    "syntax error", "syntaxerror",
    "permission denied for",                    # 表存在但无权限：改 SQL 或换表
)


def _classify_sql_failure(exc: BaseException) -> str:
    """把 SQL 执行失败粗分为「SQL 写错了」还是「数据库不可用」。

    为什么不能只看异常类型：SQLite 与 PostgreSQL 都会把「列不存在」这类
    编写错误包装成 OperationalError，而连接失败也是 OperationalError。
    两者对 Agent 的意义完全不同——前者要改 SQL，后者要么等环境恢复、要么
    换用 execute_python。因此这里**先看错误信息特征，再退回异常类型**。

    参数：
        exc: run_sql 抛出的异常。

    返回：
        错误类型常量（KIND_INVALID_ARGUMENT 或 KIND_DB_UNAVAILABLE）。
    """
    # 延迟导入：显式引入让依赖关系更清晰
    from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError

    message = str(exc).lower()
    if any(p in message for p in _SQL_MISTAKE_PATTERNS):
        # 列名/表名/语法问题 → 模型改 SQL 即可修复
        return KIND_INVALID_ARGUMENT
    if isinstance(exc, ProgrammingError):
        return KIND_INVALID_ARGUMENT
    if isinstance(exc, (OperationalError, InterfaceError)):
        return KIND_DB_UNAVAILABLE
    return KIND_DB_UNAVAILABLE


# 危险关键字（出现在 SQL **语法骨架**里即拒绝）。
#
# 说明：扫描前会先剥离字符串字面量与注释（见 _strip_sql_literals），因此
# `WHERE name = 'delete'` 这类把关键字当**数据**用的查询不会被误杀，而
# `DELETE FROM t`、`SELECT 1; DROP TABLE x` 仍会被拦住。
# 这里刻意不收 REPLACE / COMMENT：前者是合法的字符串函数（REPLACE(col,'a','b')），
# 后者是常见的列名，把它们一刀切会误伤正常分析查询；两者都无法作为语句出现在
# 以 SELECT/WITH/EXPLAIN 开头的只读语句里，且数据库侧还有只读事务兜底。
DANGEROUS_KEYWORDS = {
    # DDL / DML
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "MERGE", "CALL", "COPY", "VACUUM",
    "ATTACH", "DETACH", "REINDEX", "LOCK", "RENAME", "DO", "SET",
    # SELECT ... INTO <table> 会在只读语句里偷偷建表，必须拦截
    "INTO",
}

# 危险函数：出现在 SELECT 里就能读写服务器文件或访问外部数据源，
# 单靠关键字拦不住（它们不是语句），因此单独按函数名拒绝。
DANGEROUS_FUNCTIONS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "pg_write_file",
}

# 允许的开头关键字
ALLOWED_LEADING = {"SELECT", "WITH", "EXPLAIN"}

# 单条查询最多拉取的行数（防止 SELECT * 大表把整个结果集读进内存导致 OOM）；
# 工具层本来就只把前 50 行回传 LLM，这里是数据库侧的硬上限
SQL_MAX_ROWS = 1000

# PostgreSQL 语句级超时（毫秒）：超时后数据库主动取消查询，避免慢查询拖垮整库
SQL_STATEMENT_TIMEOUT_MS = 10000


def _strip_sql_literals(sql: str) -> str:
    """把 SQL 中的字符串字面量、引号标识符与注释替换为空格，返回“语法骨架”。

    为什么需要它：直接对原始 SQL 做关键字/分号扫描会同时产生两类错误——
    1. **误杀**：``WHERE name = 'delete'``、``SELECT REPLACE(x,'a','b')``、
       名为 ``"comment"`` 的列都会命中关键字黑名单；
    2. **误判语句数**：``SELECT ';'`` 里的分号会被当成第二条语句的分隔符。

    因此先剥离“不参与语法”的内容，再对骨架做校验。剥离规则覆盖 PostgreSQL 的
    全部字面量形式：行注释 ``--``、可嵌套的块注释 ``/* */``、单引号字符串
    （含 ``''`` 与反斜杠转义）、双引号标识符（含 ``""`` 转义）、
    以及 ``$$ ... $$`` / ``$tag$ ... $tag$`` 美元引用。

    参数：
        sql: 原始 SQL 文本。
    返回值：
        等长的语法骨架字符串（被剥离的字符统一替换为空格，便于按位置对齐）。
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]

        # ---- 行注释：-- 到行尾 ----
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                out.append(" ")
                i += 1
            continue

        # ---- 块注释：/* ... */（PostgreSQL 支持嵌套）----
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            depth = 1
            out.append("  ")
            i += 2
            while i < n and depth > 0:
                if sql.startswith("/*", i):
                    depth += 1
                    out.append("  ")
                    i += 2
                elif sql.startswith("*/", i):
                    depth -= 1
                    out.append("  ")
                    i += 2
                else:
                    out.append(" ")
                    i += 1
            continue

        # ---- 单引号字符串：'...'，'' 表示转义的单引号 ----
        if ch == "'":
            # 反斜杠只在 PostgreSQL 的 E'...' 转义字符串里才是转义符。
            # 普通字符串（standard_conforming_strings=on）中反斜杠是普通字符，
            # 若一律当作转义，`SELECT '\' ; DROP ...` 会把后面的内容全吞进
            # “字符串”，从而让骨架丢失真实语句——所以必须区分对待。
            escape_string = (
                i > 0
                and sql[i - 1] in "Ee"
                and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_"))
            )
            out.append(" ")
            i += 1
            while i < n:
                if escape_string and sql[i] == "\\" and i + 1 < n:
                    # E'...' 形式的反斜杠转义，连同被转义字符一起吞掉
                    out.append("  ")
                    i += 2
                    continue
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        out.append("  ")
                        i += 2
                        continue
                    out.append(" ")
                    i += 1
                    break
                out.append(" ")
                i += 1
            continue

        # ---- 双引号标识符："..."，"" 表示转义 ----
        if ch == '"':
            out.append(" ")
            i += 1
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        out.append("  ")
                        i += 2
                        continue
                    out.append(" ")
                    i += 1
                    break
                out.append(" ")
                i += 1
            continue

        # ---- 美元引用：$$ ... $$ 或 $tag$ ... $tag$ ----
        if ch == "$":
            m = re.match(r"\$[A-Za-z_]\w*\$|\$\$", sql[i:])
            if m:
                tag = m.group(0)
                end = sql.find(tag, i + len(tag))
                stop = n if end == -1 else end + len(tag)
                out.append(" " * (stop - i))
                i = stop
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _top_level_limit_value(skeleton: str) -> int | None:
    """取出语法骨架中**顶层** LIMIT 的数值（括号深度为 0 的那一个）。

    只看顶层是关键：``WITH x AS (SELECT ... LIMIT 5) SELECT * FROM x`` 里的
    LIMIT 作用在子查询上，无法限制外层结果集大小，不能当作已有行数保护。

    参数：
        skeleton: 经 _strip_sql_literals 处理后的 SQL 语法骨架。
    返回值：
        顶层 LIMIT 的整数值；没有顶层 LIMIT 或写法不是纯数字时返回 None。
    """
    depth = 0
    for m in re.finditer(r"[()]|\blimit\b", skeleton, flags=re.IGNORECASE):
        token = m.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            # 命中顶层 LIMIT，尝试读取紧随其后的数字
            rest = skeleton[m.end():].strip()
            num = re.match(r"(\d+)", rest)
            return int(num.group(1)) if num else None
    return None


def sql_safety_error(sql: str) -> str | None:
    """对 SQL 做静态安全校验。

    校验在 _strip_sql_literals 产出的“语法骨架”上进行，因此字符串里的关键字
    不会被误杀，而真正的写操作/多语句拼接依然会被拦住。

    参数：
        sql: 待执行的 SQL 文本。
    返回值：
        str | None: 有风险时返回中文违规原因；通过全部检查时返回 None。
    """
    # 去掉首尾空白与结尾分号，便于后续判断
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return "SQL 为空"

    # 剥离字面量/注释后再做全部判断，避免把数据当成语法
    skeleton = _strip_sql_literals(stripped)

    # 多语句拒绝（除结尾分号外）：防止 "SELECT ...; DROP TABLE ..." 式注入
    statements = [s.strip() for s in skeleton.split(";") if s.strip()]
    if len(statements) != 1:
        return "禁止一次执行多条语句"

    # 开头关键字必须是 SELECT / WITH / EXPLAIN（挡住 SET/PRAGMA/DO 等）
    leading = re.match(r"\s*([A-Za-z]+)", skeleton)
    if not leading or leading.group(1).upper() not in ALLOWED_LEADING:
        return f"仅允许 SELECT / WITH / EXPLAIN 开头的只读查询，收到: {leading.group(1) if leading else '?'}"

    # 骨架中检测危险关键字（按词边界匹配，忽略大小写）
    for kw in DANGEROUS_KEYWORDS:
        if re.search(rf"\b{kw}\b", skeleton, flags=re.IGNORECASE):
            return f"检测到危险关键字: {kw}"

    # 危险函数（如 pg_read_file）可借 SELECT 读写服务器文件，按函数名单独拦截
    for fn in DANGEROUS_FUNCTIONS:
        if re.search(rf"\b{fn}\b", skeleton, flags=re.IGNORECASE):
            return f"检测到危险函数: {fn}"

    return None


def _with_row_guard(sql: str) -> str:
    """给查询包一层行数上限，防止整表拉取导致内存溢出（OOM）。

    规则：
    - EXPLAIN 语句不包装（它只返回执行计划文本，且语法上不能套子查询限制行数）；
    - 有**顶层** LIMIT 且其数值不超过上限时，尊重作者意图不再加码；
    - 顶层 LIMIT 缺失、写法无法解析（如 ``LIMIT ALL``）或数值超过上限时，
      统一包装为 ``SELECT * FROM (<原 SQL>) AS _row_guard LIMIT N``。

    参数：
        sql: 已通过静态校验、去掉结尾分号的单条只读 SQL。
    返回值：
        带行数保护的 SQL 字符串。
    """
    if re.match(r"\s*explain\b", sql, flags=re.IGNORECASE):
        return sql
    # 用语法骨架判断 LIMIT，避免字符串字面量里的 "limit" 误导判断
    limit_value = _top_level_limit_value(_strip_sql_literals(sql))
    if limit_value is not None and limit_value <= SQL_MAX_ROWS:
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
        整体再按 output_truncate_chars 截断）；安全拒绝或连接失败返回
        **结构化错误文本**（首行带错误类型标记）。
    """
    # 工具层先做一次快速拦截，避免连数据库都不创建就直接拒绝
    error = sql_safety_error(sql)
    if error:
        return tool_error(
            KIND_SECURITY,
            f"SQL 被安全策略拒绝：{error}",
            hint="请改写为单条 SELECT / WITH / EXPLAIN 只读查询，"
                 "不要包含写操作、DDL 或多语句。",
        )

    try:
        df = run_sql(sql)
    except Exception as e:  # 数据库未启动 / 连接失败等
        # 区分「SQL 本身写错」与「数据库连不上」：前者要改 SQL，后者要等环境恢复。
        # 通过异常类型粗判，给出不同的修复建议，避免模型对着环境问题反复改 SQL。
        kind = _classify_sql_failure(e)
        return tool_error(
            kind,
            f"SQL 执行失败：{type(e).__name__}: {e}",
            hint=(
                "请检查列名、表名与语法是否正确后重试。"
                if kind == "invalid_argument"
                else "数据库可能未启动或不可用；若无法恢复，请改用 execute_python 分析数据集。"
            ),
        )

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
