"""Python 代码执行工具（沙箱化）。

输入输出契约：
- 输入：LLM 生成的 Python 代码字符串 code + 数据文件路径 dataset_path。
- 子进程执行前会自动拼接 preamble：预导入 pandas/numpy/math 等白名单库，
  并把目标数据集加载为变量 `df`；用户代码无需也无法自行读文件。
- 输出：stdout/stderr 合并后的纯文本（要求代码用 print() 产出结果），
  超时、安全违规、运行异常均返回中文提示字符串，不向 Agent 抛异常。

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

# 允许导入的模块白名单（只取顶层包名判断，如 matplotlib.pyplot 只看 matplotlib）
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

# 单次提交代码的最大字符数，防止超长代码拖垮执行与上下文
MAX_CODE_LENGTH = 8000


class SandboxSecurityError(ValueError):
    """代码未通过安全校验时抛出。"""


class _SecurityVisitor(ast.NodeVisitor):
    """遍历代码 AST，收集全部违规调用/导入/属性访问。

    采用“先收集后统一报错”的方式，一次性把所有违规点反馈给 LLM，便于其修正。
    """

    def __init__(self) -> None:
        # 遍历过程中发现的违规描述列表
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """处理 `import x.y` 语句：顶层包名必须在白名单内。"""
        for alias in node.names:
            # 只取第一段包名：numpy.random → numpy，避免子模块绕过白名单
            top = alias.name.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                self.violations.append(f"禁止导入模块: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """处理 `from x import y` 语句：来源模块的顶层包名必须在白名单内。"""
        top = (node.module or "").split(".")[0]
        if top not in ALLOWED_IMPORTS:
            self.violations.append(f"禁止导入模块: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """处理所有函数/方法调用：拦截危险内置函数与危险方法名。"""
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
            # 直接以名字调用，如 open("x")、eval(...)
            self.violations.append(f"禁止调用内置函数: {func.id}()")
        elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALL_ATTRS:
            # 以属性形式调用，如 os.system(...)、pd.read_csv(...)
            self.violations.append(f"禁止调用方法: .{func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """拦截双下划线属性访问（如 __globals__/__subclasses__），阻断常见逃逸手法。"""
        if node.attr.startswith("__"):
            self.violations.append(f"禁止访问双下划线属性: .{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """兜底：即使危险名字未被调用（如赋值/传参引用），也禁止出现。"""
        if node.id in {"eval", "exec", "compile", "__import__", "open", "input"}:
            self.violations.append(f"禁止使用: {node.id}")
        self.generic_visit(node)


def validate_code(code: str) -> None:
    """校验代码安全性，违规则抛 SandboxSecurityError。

    参数：
        code: LLM 提交的 Python 源码。
    异常：
        SandboxSecurityError: 代码为空、超长、语法错误或命中任一安全规则。
    """
    if not code or not code.strip():
        raise SandboxSecurityError("代码为空")

    if len(code) > MAX_CODE_LENGTH:
        raise SandboxSecurityError(f"代码过长（> {MAX_CODE_LENGTH} 字符）")

    # 先做语法解析：语法不合法的代码既无法遍历也无法执行
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxSecurityError(f"代码语法错误: {e}") from e

    # AST 静态检查：白名单导入 + 危险调用/属性扫描
    visitor = _SecurityVisitor()
    visitor.visit(tree)
    if visitor.violations:
        raise SandboxSecurityError("；".join(visitor.violations))


def _build_preamble(dataset_path: Path) -> str:
    """构造加载数据的前导代码（拼在用户代码之前）。

    参数：
        dataset_path: 已通过路径安全校验的数据集绝对路径。
    返回值：
        str: 一段 Python 源码：设置 UTF-8 输出、预导入常用库、把数据读入 df。
    """
    # repr 会给字符串加引号并转义，安全地把路径嵌成 Python 字面量
    path_literal = repr(str(dataset_path))
    suffix = dataset_path.suffix.lower()
    # 按扩展名选择读取方式（与 file_tools.load_dataframe 保持一致）
    if suffix in (".xlsx", ".xls"):
        reader = f"df = pd.read_excel({path_literal})"
    elif suffix == ".json":
        reader = f"df = pd.read_json({path_literal})"
    else:
        reader = f"df = pd.read_csv({path_literal}, encoding='utf-8-sig')"

    return (
        # 强制子进程标准流按 UTF-8 编解码，保证 Windows 下中文 print 不乱码
        "import sys\n"
        "try:\n"
        "    sys.stdout.reconfigure(encoding='utf-8')\n"
        "    sys.stderr.reconfigure(encoding='utf-8')\n"
        "except Exception:\n"
        "    pass\n"
        # 预导入白名单库，用户代码可直接使用，免去自行 import
        "import pandas as pd\n"
        "import numpy as np\n"
        "import math, statistics, json, re, datetime, collections, itertools, functools\n\n"
        # 最后把数据集加载为 df，用户代码从这里开始执行
        f"{reader}\n\n"
    )


def run_python(code: str, dataset_path: str) -> str:
    """在隔离子进程中执行代码并返回 stdout/stderr 文本。

    参数：
        code: 通过安全校验的用户 Python 代码。
        dataset_path: 数据集路径（会被解析校验并预加载为 df）。
    返回值：
        str: 标准输出文本（有 stderr 时附 "[stderr]" 段）；
        超时时返回超时提示；无输出时返回引导 print 的提示。
    异常：
        SandboxSecurityError / FileNotFoundError: 由上层 @tool 捕获转提示。
    """
    settings = get_settings()
    # 路径安全校验（与文件工具同源），防止借沙箱读取 workspace 之外的文件
    path = settings.resolve_dataset_path(dataset_path)
    validate_code(code)

    # 前导代码 + 用户代码组成完整脚本
    full_code = _build_preamble(path) + code

    try:
        # 用当前解释器以 -c 方式启动隔离子进程：
        # cwd 锁定 workspace；capture_output 捕获输出；timeout 到时父进程会 kill 子进程
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
        # subprocess 超时后会自动终止子进程，这里仅把事件转成可读提示
        return f"错误：Python 代码执行超时（>{settings.python_timeout_seconds}s）。"

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        # 保留 stderr（通常是异常栈），帮助 LLM 据报错修正代码
        parts.append("[stderr]\n" + stderr)
    if not parts:
        return "(无输出——请在代码末尾用 print() 输出结果)"

    # 输出截断，防止巨型结果撑爆 LLM 上下文
    return ("\n\n".join(parts))[: settings.output_truncate_chars]


@tool
def execute_python(code: str, dataset_path: str) -> str:
    """在沙箱中执行 Pandas/NumPy 数据分析代码（LangChain 工具入口）。

    数据已加载为变量 `df`，直接对 df 做分析即可。
    必须用 print() 输出结果。只允许导入白名单模块，禁止访问系统与网络。

    返回值：
        str: 执行输出文本；安全拒绝/文件缺失/运行异常均返回中文提示字符串，
        不向外抛异常，保证 Agent 链路不中断。
    """
    try:
        return run_python(code, dataset_path)
    except SandboxSecurityError as e:
        return f"代码被安全策略拒绝：{e}"
    except PermissionError as e:
        # 数据文件路径不在允许的数据目录内（路径穿越/任意文件读取拦截）
        return f"错误：{e}"
    except FileNotFoundError as e:
        return f"错误：{e}"
    except Exception as e:  # 兜底，避免工具抛异常中断 Agent 链路
        return f"执行出错：{type(e).__name__}: {e}"
