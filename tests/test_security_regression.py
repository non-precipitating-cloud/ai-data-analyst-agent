"""安全回归测试：Python 沙箱逃逸与 SQL 只读守卫。

这些用例对应真实修复过的漏洞与误杀问题，属于**必须长期守住**的行为契约：

Python 沙箱
- 旧实现只拦「以名字调用 eval/open」，导致 ``__builtins__.open(...)`` 与
  ``__builtins__["open"](...)`` 可以绕过 AST 校验读取沙箱外任意文件
  （包括含 API Key 的 .env）。现在 AST 层拦截全部双下划线名字与下标间接调用，
  运行时再把用户代码放进「受限 builtins」命名空间执行，形成第二道防线。
- 无输出死循环、无限打印必须被超时/输出上限终止，不能拖垮进程。

SQL 守卫
- 扫描改为在「语法骨架」（剥离字符串与注释）上进行：既能拦住
  ``SELECT 1; DROP TABLE x``，又不再误杀 ``REPLACE(...)``、名为 comment 的列、
  以及字符串里带分号或 "delete" 的合法查询。
- ``SELECT ... INTO`` 与 pg_read_file 一类危险函数必须被拦截。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.tools.python_tool import (
    MAX_OUTPUT_BYTES,
    SandboxSecurityError,
    _build_preamble,
    _run_subprocess,
    validate_code,
)
from src.tools.sql_tool import (
    SQL_MAX_ROWS,
    _strip_sql_literals,
    _top_level_limit_value,
    _with_row_guard,
    sql_safety_error,
)
from tests.conftest import SALES_CSV


# ==========================================================================
# Python 沙箱 —— AST 层
# ==========================================================================
@pytest.mark.parametrize(
    "code",
    [
        # 旧实现可绕过的两种写法：__builtins__ 是 Name，不是被禁的调用目标
        "__builtins__.open('C:/Windows/win.ini')",
        "__builtins__['open']('C:/Windows/win.ini')",
        "__builtins__['eval']('1+1')",
        # 借助内省链逃逸
        "x = ''.__class__",
        "__loader__.load_module('os')",
        # 下标取出函数再调用（间接调用一律拒绝）
        "d = {'f': print}\nd['f']('x')",
        # 目录/属性枚举是逃逸的前置步骤
        "dir(df)",
        "getattr(df, 'to_csv')('x.csv')",
    ],
)
def test_sandbox_ast_blocks_escapes(code: str) -> None:
    """所有沙箱逃逸写法都必须在 AST 校验阶段被拒绝。"""
    with pytest.raises(SandboxSecurityError):
        validate_code(code)


def test_sandbox_still_allows_normal_analysis() -> None:
    """加固不能误伤正常的数据分析代码。"""
    validate_code("import pandas as pd\nprint(df.groupby('region')['sales'].sum())")
    validate_code("result = df[df['sales'] > 1000]['profit'].mean()\nprint(result)")
    validate_code("import numpy as np\nprint(np.corrcoef(df['sales'], df['profit']))")


# ==========================================================================
# Python 沙箱 —— 运行时受限 builtins（第二道防线）
# ==========================================================================
def _run_in_sandbox(code: str, timeout: int = 20):
    """在当前解释器里跑一段用户代码（绕过 AST，用于验证运行时防线）。"""
    script = _build_preamble(Path(SALES_CSV).resolve(), code)
    return _run_subprocess([sys.executable, "-c", script], cwd="workspace", timeout=timeout)


def test_runtime_builtins_deny_open_even_if_ast_bypassed() -> None:
    """即使 AST 校验被绕过，用户命名空间里也不存在 open —— 这是真正的兜底。"""
    out, err, failure = _run_in_sandbox(
        "try:\n"
        "    open('x')\n"
        "except NameError as e:\n"
        "    print('NO_OPEN')\n"
    )
    assert failure is None
    assert "NO_OPEN" in out or "NameError" in (out + err)


def test_runtime_builtins_deny_eval_and_dunder_import() -> None:
    """eval 与 __import__ 在受限命名空间中同样不可用。"""
    out, err, failure = _run_in_sandbox(
        "try:\n"
        "    eval('1+1')\n"
        "except NameError:\n"
        "    print('NO_EVAL')\n"
    )
    assert failure is None
    assert "NO_EVAL" in out


def test_runtime_import_guard_blocks_non_whitelisted_module() -> None:
    """import 语句走运行时的白名单再校验，非白名单模块导入失败。"""
    out, err, failure = _run_in_sandbox(
        "try:\n"
        "    import os\n"
        "except ImportError as e:\n"
        "    print('BLOCKED_IMPORT')\n"
    )
    assert failure is None
    assert "BLOCKED_IMPORT" in out


def test_runtime_still_allows_whitelisted_import_and_data() -> None:
    """白名单库与预加载的 df 仍可正常使用。"""
    out, err, failure = _run_in_sandbox(
        "import statistics\nprint('OK', len(df), round(statistics.mean([1, 2, 3]), 2))\n"
    )
    assert failure is None
    # statistics.mean([1,2,3]) == 2，这里只断言「能导入 + 能拿到 df + 能算」
    assert "OK 2160 2" in out


# ==========================================================================
# Python 沙箱 —— 资源限制
# ==========================================================================
def test_sandbox_output_overflow_is_terminated() -> None:
    """无限打印必须在输出超过上限时被立即终止，而不是把父进程内存撑爆。"""
    out, err, failure = _run_in_sandbox("while True:\n    print('x' * 10000)\n")
    assert failure is not None
    assert "输出超过上限" in failure
    assert f"{MAX_OUTPUT_BYTES // 1024}KB" in failure


def test_sandbox_timeout_kills_cpu_loop() -> None:
    """无输出的纯 CPU 死循环必须被超时终止。"""
    out, err, failure = _run_in_sandbox("x = 0\nwhile True:\n    x += 1\n", timeout=3)
    assert failure is not None
    assert "超时" in failure


# ==========================================================================
# SQL 守卫 —— 不再误杀合法查询
# ==========================================================================
@pytest.mark.parametrize(
    "sql",
    [
        # REPLACE 是合法的字符串函数，旧实现因关键字黑名单直接拒绝
        "SELECT REPLACE(region, '华东', 'East') FROM sales",
        # comment 是常见列名，旧实现同样误杀
        "SELECT comment FROM sales",
        "SELECT * FROM sales WHERE region = 'delete'",
        # 字符串里的分号不应被当成第二条语句
        "SELECT * FROM sales WHERE name = 'a;b'",
        # 行注释里的内容不构成语法
        "SELECT * FROM sales -- DROP TABLE x",
        # 美元引用字符串同样不参与语法
        "SELECT $$ DROP TABLE x $$ AS note FROM sales",
    ],
)
def test_sql_allows_legitimate_queries(sql: str) -> None:
    """含关键字字样但语法合法的只读查询必须放行。"""
    assert sql_safety_error(sql) is None


# ==========================================================================
# SQL 守卫 —— 危险操作仍然拦住
# ==========================================================================
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "UPDATE sales SET a = 1",
        "INSERT INTO sales VALUES (1)",
        "ALTER TABLE sales ADD COLUMN x INT",
        "TRUNCATE sales",
        "SELECT 1; DROP TABLE x",
        # SELECT INTO 会借只读语句建表
        "SELECT * INTO evil FROM sales",
        # 危险函数可读服务器文件
        "SELECT pg_read_file('/etc/passwd') FROM sales",
        "SELECT * FROM dblink('x', 'y') AS t(a int)",
        # 伪造转义字符串试图把后面的语句吞进字面量
        "SELECT '\\' ; DROP TABLE x",
    ],
)
def test_sql_blocks_dangerous_statements(sql: str) -> None:
    """写操作、DDL、危险函数与多语句拼接必须全部拒绝。"""
    assert sql_safety_error(sql) is not None


def test_sql_literal_stripper_is_length_preserving() -> None:
    """语法骨架与原串等长，便于按位置对齐与调试。"""
    sql = "SELECT 'abc', \"col\", 1 -- x"
    assert len(_strip_sql_literals(sql)) == len(sql)


# ==========================================================================
# SQL 守卫 —— 行数上限
# ==========================================================================
def test_row_guard_ignores_limit_inside_string() -> None:
    """字符串里的 "limit" 不应被当成已有行数限制。"""
    guarded = _with_row_guard("SELECT * FROM t WHERE n = 'limit'")
    assert f"LIMIT {SQL_MAX_ROWS}" in guarded


def test_row_guard_wraps_subquery_limit() -> None:
    """子查询里的 LIMIT 限制不了外层结果集，仍应外包行数上限。"""
    guarded = _with_row_guard("WITH x AS (SELECT 1 LIMIT 3) SELECT * FROM x")
    assert f"LIMIT {SQL_MAX_ROWS}" in guarded


def test_row_guard_clamps_oversized_limit() -> None:
    """显式写了超过上限的 LIMIT 时，仍需收敛到安全行数。"""
    guarded = _with_row_guard("SELECT * FROM sales LIMIT 999999")
    assert f"LIMIT {SQL_MAX_ROWS}" in guarded


def test_row_guard_keeps_small_explicit_limit() -> None:
    """小于上限的显式 LIMIT 应被尊重，不重复包装。"""
    sql = "SELECT * FROM sales LIMIT 5"
    assert _with_row_guard(sql) == sql


def test_top_level_limit_value_skips_comments() -> None:
    """注释里的 limit 不应被识别为顶层 LIMIT。"""
    assert _top_level_limit_value(_strip_sql_literals("SELECT 1 -- limit 5")) is None
