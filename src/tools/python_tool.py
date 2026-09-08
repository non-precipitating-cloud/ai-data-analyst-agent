"""Python 代码执行工具（沙箱化）。

安全限制（尽力而为，非严格操作系统级隔离）：
- 子进程运行，工作目录锁定在 workspace
- AST 白名单：只允许导入白名单内的模块
- 禁止危险内置函数（eval/exec/open/...）与危险方法（.system/.popen/...）
- 禁止访问双下划线属性（阻断常见沙箱逃逸）
- 执行超时 + 输出截断

注意：Windows 下无法使用 `resource` 限制内存，故内存限制依赖超时与白名单。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

from src.config.settings import get_settings

# 允许导入的模块白名单
ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "collections", "itertools",
    "json", "datetime", "re", "functools", "random", "typing",
}

# 禁止调用的内置函数 / 危险方法
FORBIDDEN_BUILTINS = {
    "eval", "exec", "compile", "__import__", "input", "open", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "hasattr",
    "exit", "quit", "breakpoint", "memoryview",
}

FORBIDDEN_CALL_ATTRS = {
    "system", "popen", "spawn", "fork", "execv", "execve", "remove", "unlink",
    "rmdir", "kill", "startfile", "read_sql", "read_sql_query", "read_sql_table",
    "to_sql", "to_pickle", "read_pickle", "urlopen",
    # pandas 文件读写：防止访问 workspace 之外的文件（数据已由 preamble 加载为 df）
    "read_csv", "read_excel", "read_json", "read_table", "read_fwf",
    "read_clipboard", "read_parquet", "read_feather", "read_orc", "read_html",
    "to_csv", "to_excel", "to_json", "to_html", "to_markdown", "to_clipboard",
    "to_parquet", "to_feather", "to_orc",
}

MAX_CODE_LENGTH = 8000


class SandboxSecurityError(ValueError):
    """代码未通过安全校验时抛出。"""


class _SecurityVisitor(ast.NodeVisitor):
    """遍历 AST 收集违规项。"""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                self.violations.append(f"禁止导入模块: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        top = (node.module or "").split(".")[0]
        if top not in ALLOWED_IMPORTS:
            self.violations.append(f"禁止导入模块: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
            self.violations.append(f"禁止调用内置函数: {func.id}()")
        elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALL_ATTRS:
            self.violations.append(f"禁止调用方法: .{func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.violations.append(f"禁止访问双下划线属性: .{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in {"eval", "exec", "compile", "__import__", "open", "input"}:
            self.violations.append(f"禁止使用: {node.id}")
        self.generic_visit(node)


def validate_code(code: str) -> None:
    """校验代码安全性，违规则抛 SandboxSecurityError。"""
    if not code or not code.strip():
        raise SandboxSecurityError("代码为空")

    if len(code) > MAX_CODE_LENGTH:
        raise SandboxSecurityError(f"代码过长（> {MAX_CODE_LENGTH} 字符）")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxSecurityError(f"代码语法错误: {e}") from e

    visitor = _SecurityVisitor()
    visitor.visit(tree)
    if visitor.violations:
        raise SandboxSecurityError("；".join(visitor.violations))


def _build_preamble(dataset_path: Path) -> str:
    """构造加载数据的前导代码。"""
    path_literal = repr(str(dataset_path))
    suffix = dataset_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        reader = f"df = pd.read_excel({path_literal})"
    elif suffix == ".json":
        reader = f"df = pd.read_json({path_literal})"
    else:
        reader = f"df = pd.read_csv({path_literal}, encoding='utf-8-sig')"

    return (
        "import sys\n"
        "try:\n"
        "    sys.stdout.reconfigure(encoding='utf-8')\n"
        "    sys.stderr.reconfigure(encoding='utf-8')\n"
        "except Exception:\n"
        "    pass\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import math, statistics, json, re, datetime, collections, itertools, functools\n\n"
        f"{reader}\n\n"
    )


def run_python(code: str, dataset_path: str) -> str:
    """执行代码并返回 stdout/stderr 文本。"""
    settings = get_settings()
    path = settings.resolve_dataset_path(dataset_path)
    validate_code(code)

    full_code = _build_preamble(path) + code

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_code],
            cwd=str(settings.workspace_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.python_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"错误：Python 代码执行超时（>{settings.python_timeout_seconds}s）。"

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append("[stderr]\n" + stderr)
    if not parts:
        return "(无输出——请在代码末尾用 print() 输出结果)"

    return ("\n\n".join(parts))[: settings.output_truncate_chars]


@tool
def execute_python(code: str, dataset_path: str) -> str:
    """在沙箱中执行 Pandas/NumPy 数据分析代码。

    数据已加载为变量 `df`，直接对 df 做分析即可。
    必须用 print() 输出结果。只允许导入白名单模块，禁止访问系统与网络。
    """
    try:
        return run_python(code, dataset_path)
    except SandboxSecurityError as e:
        return f"代码被安全策略拒绝：{e}"
    except FileNotFoundError as e:
        return f"错误：{e}"
    except Exception as e:  # 兜底，避免工具抛异常中断 Agent 链路
        return f"执行出错：{type(e).__name__}: {e}"
