"""Python 代码执行工具（沙箱子进程）。

输入输出契约：
- 输入：LLM 生成的 Python 代码字符串 code + 数据文件路径 dataset_path。
- 子进程执行前会自动拼接 preamble：预导入 pandas/numpy/math 等白名单库，
  把目标数据集加载为变量 `df`；用户代码在「受限 builtins 命名空间」中 exec。
- 输出：stdout/stderr 合并后的纯文本（要求代码用 print() 产出结果），
  超时、安全违规、运行异常均返回中文提示字符串，不向 Agent 抛异常。

安全模型（纵深防御，四层，尽力而为而非操作系统级隔离）：
1. AST 静态校验：只允许白名单模块导入；拦截危险内置函数与危险方法调用；
   拦截**全部**双下划线名字与属性（阻断 ``__builtins__``/``__class__`` 一类逃逸）。
2. 运行时受限 builtins：用户代码并非在完整的 Python 内置命名空间中执行，
   而是在一个只含白名单内置函数的字典里 exec。即使 AST 校验被绕过，
   ``open`` / ``eval`` / ``exec`` / ``__import__`` / ``getattr`` 等能力也不存在。
   ``import`` 语句走运行时的白名单再校验，形成第二道闸门。
3. 资源限制：执行超时 + 输出体积上限（超限立即终止子进程）+ 进程树终止，
   避免无限循环、内存膨胀与孤儿进程。
4. 路径边界：数据路径经 settings.resolve_dataset_path 校验，只能读取允许的数据目录。

**能力边界（不做过度承诺）**：本沙箱不是容器/虚拟机级别的隔离——
没有 seccomp、没有独立的文件系统命名空间，Windows 下也无法用 ``resource``
限制内存。它与「只读数据目录 + 无网络客户端库 + 白名单导入 + 超时」共同降低风险，
但不应被当作可以安全执行**任意不可信代码**的环境。需要更强隔离请启用 Docker
（见 README「Python 沙箱的安全边界」一节）。
"""

from __future__ import annotations

import ast
import base64
import contextlib
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from langchain_core.tools import tool

from src.config.settings import get_settings
from src.tools.errors import (
    KIND_NOT_FOUND,
    KIND_SECURITY,
    describe_exception,
    tool_error,
)

# 允许导入的模块白名单（只取顶层包名判断，如 matplotlib.pyplot 只看 matplotlib）
ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "collections", "itertools",
    "json", "datetime", "re", "functools", "random", "typing",
}

# 禁止调用的内置函数 / 危险方法。
# 这是 AST 层的「快速失败」校验；真正的兜底是运行时的受限 builtins
# （见 SAFE_BUILTIN_NAMES）——即使这里漏掉某个名字，用户代码也拿不到它。
FORBIDDEN_BUILTINS = {
    "eval", "exec", "compile", "__import__", "input", "open", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "hasattr",
    "exit", "quit", "breakpoint", "memoryview", "help", "dir", "object",
}

FORBIDDEN_CALL_ATTRS = {
    "system", "popen", "spawn", "fork", "execv", "execve", "remove", "unlink",
    "rmdir", "kill", "startfile", "read_sql", "read_sql_query", "read_sql_table",
    "to_sql", "to_pickle", "read_pickle", "urlopen", "urlretrieve",
    # pandas 文件读写：防止访问 workspace 之外的文件（数据已由 preamble 加载为 df）
    "read_csv", "read_excel", "read_json", "read_table", "read_fwf",
    "read_clipboard", "read_parquet", "read_feather", "read_orc", "read_html",
    "read_pickle", "to_csv", "to_excel", "to_json", "to_html", "to_markdown",
    "to_clipboard", "to_parquet", "to_feather", "to_orc",
}

# 单次提交代码的最大字符数，防止超长代码拖垮执行与上下文
MAX_CODE_LENGTH = 8000

# 子进程输出（stdout+stderr 合计）的最大字节数；超过即终止子进程。
# 目的：防止 `while True: print("x" * 10000)` 这类代码在超时前把父进程内存撑爆
MAX_OUTPUT_BYTES = 256 * 1024

# 运行时允许用户代码使用的内置函数白名单（第二层防御的核心）。
# 刻意不含 open/eval/exec/compile/__import__/getattr/setattr/delattr/dir/vars/locals/globals。
# 覆盖数据分析常用能力 + 常用异常类型，保证正常分析代码不受影响。
SAFE_BUILTIN_NAMES = (
    # 输出与基础转换
    "print", "repr", "format", "str", "int", "float", "bool", "complex",
    "bytes", "bytearray", "list", "dict", "set", "frozenset", "tuple",
    # 常用序列/迭代/数学
    "len", "range", "sum", "min", "max", "abs", "round", "sorted", "reversed",
    "enumerate", "zip", "map", "filter", "any", "all", "divmod", "pow",
    "iter", "next", "callable", "slice", "hash", "ord", "chr", "bin", "hex", "oct",
    # 类型判断
    "isinstance", "issubclass", "type",
    # 异常类型（try/except 需要）
    "BaseException", "Exception", "ArithmeticError", "AssertionError",
    "AttributeError", "FloatingPointError", "ImportError", "IndexError",
    "KeyError", "LookupError", "MemoryError", "ModuleNotFoundError",
    "NameError", "NotImplementedError", "OSError", "OverflowError",
    "RuntimeError", "StopIteration", "TypeError", "UnicodeDecodeError",
    "ValueError", "ZeroDivisionError", "Warning",
    # 常量对象
    "NotImplemented", "Ellipsis",
)


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
        """处理所有函数/方法调用：拦截危险内置函数与危险方法名。

        额外拦截「下标取函数再调用」的间接调用（如 ``__builtins__["open"](...)``、
        ``m["f"](...)``）：这类写法没有合法的数据分析用途，却是绕过
        Name/Attribute 检查的常见手法。
        """
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
            # 直接以名字调用，如 open("x")、eval(...)
            self.violations.append(f"禁止调用内置函数: {func.id}()")
        elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALL_ATTRS:
            # 以属性形式调用，如 os.system(...)、pd.read_csv(...)
            self.violations.append(f"禁止调用方法: .{func.attr}()")
        elif isinstance(func, ast.Subscript):
            # 例如 __builtins__["open"](...) / d["func"](...)：间接调用一律拒绝
            self.violations.append("禁止通过下标取出函数再调用")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """拦截双下划线属性访问（如 __globals__/__subclasses__），阻断常见逃逸手法。"""
        if node.attr.startswith("__"):
            self.violations.append(f"禁止访问双下划线属性: .{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """拦截全部双下划线名字与已知危险名字。

        覆盖 ``__builtins__`` / ``__loader__`` / ``__spec__`` 等：这些名字能让
        用户代码重新拿到完整的内置能力，是沙箱逃逸最常见的入口。
        """
        if node.id.startswith("__"):
            # 语言级构造（如 __name__）在用户代码里没有分析价值，一律拒绝更安全
            self.violations.append(f"禁止使用双下划线名字: {node.id}")
        elif node.id in FORBIDDEN_BUILTINS:
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


def _build_preamble(dataset_path: Path, code: str) -> str:
    """构造子进程要执行的完整脚本：受限运行环境 + 预加载数据 + 用户代码。

    用户代码**不直接拼进脚本正文**，而是以 base64 字面量嵌入，再由 preamble
    解码后用 ``exec`` 在受限命名空间中执行。这样做有三点好处：
    1. 彻底避免用户代码里的引号/换行/三引号破坏生成的脚本；
    2. 句法错误（SyntaxError）发生在 exec 时，能被用户代码的异常处理友好捕获；
    3. 受限 builtins 只在用户代码的命名空间生效，preamble 自身与 pandas
       仍使用完整内置能力，因此不会误伤数据读取。

    参数：
        dataset_path: 已通过路径安全校验的数据集绝对路径。
        code: 已通过 AST 校验的用户代码。
    返回：
        str: 可直接交给 ``python -c`` 执行的完整脚本源码。
    """
    # base64 编码：内容只含 ASCII 字母数字，绝不可能破坏外层脚本结构
    code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
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

    allowed_names = repr(list(SAFE_BUILTIN_NAMES))
    allowed_imports = repr(sorted(ALLOWED_IMPORTS))

    return (
        # 强制子进程标准流按 UTF-8 编解码，保证 Windows 下中文 print 不乱码
        "import sys\n"
        "try:\n"
        "    sys.stdout.reconfigure(encoding='utf-8')\n"
        "    sys.stderr.reconfigure(encoding='utf-8')\n"
        "except Exception:\n"
        "    pass\n"
        # 预导入白名单库（在完整内置命名空间下完成，pandas 内部实现不受限制影响）
        "import pandas as pd\n"
        "import numpy as np\n"
        "import math, statistics, json, re, datetime, collections, itertools, functools\n"
        "import builtins as _builtins\n"
        "import base64 as _base64\n\n"
        # 组装受限 builtins：只暴露白名单内的名字（第二层防御）
        f"_ALLOWED_NAMES = {allowed_names}\n"
        f"_ALLOWED_IMPORTS = set({allowed_imports})\n"
        "_real_import = _builtins.__import__\n"
        "def _guarded_import(name, *a, **k):\n"
        "    if name.split('.')[0] not in _ALLOWED_IMPORTS:\n"
        "        raise ImportError('沙箱禁止导入模块: ' + name)\n"
        "    return _real_import(name, *a, **k)\n"
        "_safe_builtins = {n: getattr(_builtins, n) for n in _ALLOWED_NAMES if hasattr(_builtins, n)}\n"
        # import 语句通过 __import__ 生效，这里换成带白名单再校验的版本
        "_safe_builtins['__import__'] = _guarded_import\n"
        # 允许用户代码定义 class（class 语句需要 __build_class__）
        "_safe_builtins['__build_class__'] = _builtins.__build_class__\n\n"
        # 数据加载（使用完整内置能力，保证 pandas 正常读取）
        f"{reader}\n\n"
        # 用户命名空间：内置函数受限，数据与常用库预置
        "_user_globals = {\n"
        "    '__builtins__': _safe_builtins,\n"
        "    '__name__': '__main__',\n"
        "    'df': df, 'pd': pd, 'np': np,\n"
        "    'math': math, 'statistics': statistics, 'json': json, 're': re,\n"
        "    'datetime': datetime, 'collections': collections,\n"
        "    'itertools': itertools, 'functools': functools,\n"
        "}\n"
        # 用户代码以 base64 嵌入，解码后在受限命名空间中执行
        f"_USER_CODE = _base64.b64decode('{code_b64}').decode('utf-8')\n"
        "exec(compile(_USER_CODE, '<agent_code>', 'exec'), _user_globals)\n"
    )


def _terminate_tree(proc: subprocess.Popen) -> None:
    """终止子进程**及其派生的全部子进程**，避免留下孤儿进程。

    Windows 下 ``Popen.kill`` 只结束直接子进程，因此用 ``taskkill /T`` 杀进程树；
    POSIX 下子进程以独立会话启动，直接对进程组发送 SIGKILL。

    参数：
        proc: 已启动的子进程句柄。
    """
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            # /T 连同子进程树一起终止；/F 强制结束
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 —— 进程可能已退出，退化为单进程终止
        with contextlib.suppress(Exception):
            proc.kill()
    finally:
        # 兜底：无论上面成败，都确保直接子进程被回收
        with contextlib.suppress(Exception):
            proc.kill()


def _run_subprocess(cmd: list[str], cwd: str, timeout: int) -> tuple[str, str, str | None]:
    """在子进程中执行命令，带输出体积上限与超时保护。

    单独使用 stdout/stderr 两条管道，各自起一个后台线程读取：
    读取线程在累计字节数超过 MAX_OUTPUT_BYTES 时立即终止整个进程树并停止读取，
    从而避免「无限打印」把父进程内存吃光（超时机制对此反应太慢）。

    参数：
        cmd: 命令行参数列表。
        cwd: 子进程工作目录。
        timeout: 超时秒数。

    返回：
        三元组 (stdout 文本, stderr 文本, 错误提示)；
        正常结束时错误提示为 None，超时/输出超限时为中文说明。
    """
    # 让子进程自成一个进程组/会话，便于超时后连同其子进程一起终止
    popen_kwargs: dict = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )

    # 每条管道一个读取线程，共享同一个「超限即终止」策略
    state = {"total": 0, "overflow": False}
    lock = threading.Lock()
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}

    def _drain(stream_name: str, stream) -> None:
        """持续读取一条管道；累计超限时终止进程树（运行在后台线程）。"""
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                with lock:
                    state["total"] += len(chunk)
                    buffers[stream_name].extend(chunk)
                    over = state["total"] > MAX_OUTPUT_BYTES
                    if over:
                        state["overflow"] = True
                if over:
                    # 输出已超限：立即杀掉进程树，不再继续读取
                    _terminate_tree(proc)
                    break
        except (ValueError, OSError):
            # 进程被终止后管道关闭，读取抛错属于预期情况
            pass

    readers = [
        threading.Thread(target=_drain, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=_drain, args=("stderr", proc.stderr), daemon=True),
    ]
    for t in readers:
        t.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_tree(proc)
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)

    # 等待读取线程收尾（进程已终止，管道会很快 EOF）
    for t in readers:
        t.join(timeout=5)
    for stream in (proc.stdout, proc.stderr):
        with contextlib.suppress(Exception):
            stream.close()

    def _decode(buf: bytearray) -> str:
        """按 UTF-8 解码（无法解码的字节替换掉），并裁掉被截断的尾部。"""
        return bytes(buf[:MAX_OUTPUT_BYTES]).decode("utf-8", errors="replace").strip()

    stdout, stderr = _decode(buffers["stdout"]), _decode(buffers["stderr"])

    if timed_out:
        return stdout, stderr, f"错误：Python 代码执行超时（>{timeout}s）。"
    if state["overflow"]:
        return stdout, stderr, (
            f"错误：Python 代码输出超过上限（>{MAX_OUTPUT_BYTES // 1024}KB），已终止执行。"
            "请只 print() 关键结论，不要打印整表或逐行输出。"
        )
    return stdout, stderr, None


def run_python(code: str, dataset_path: str) -> str:
    """在隔离子进程中执行代码并返回 stdout/stderr 文本。

    参数：
        code: 通过安全校验的用户 Python 代码。
        dataset_path: 数据集路径（会被解析校验并预加载为 df）。
    返回：
        str: 标准输出文本（有 stderr 时附 "[stderr]" 段）；
        超时/输出超限时返回对应提示；无输出时返回引导 print 的提示。
    异常：
        SandboxSecurityError / PermissionError / FileNotFoundError: 由上层 @tool 捕获转提示。
    """
    settings = get_settings()
    # 路径安全校验（与文件工具同源），防止借沙箱读取 workspace 之外的文件
    path = settings.resolve_dataset_path(dataset_path)
    validate_code(code)

    # 受限运行环境 + 预加载数据 + 用户代码组成完整脚本
    full_code = _build_preamble(path, code)

    stdout, stderr, failure = _run_subprocess(
        [sys.executable, "-c", full_code],
        cwd=str(settings.workspace_dir),
        timeout=settings.python_timeout_seconds,
    )
    if failure:
        # 超时 / 输出超限：把已产生的部分输出附上，便于 LLM 判断进度
        if stdout:
            return f"{failure}\n\n[已产生的部分输出]\n{stdout[:1000]}"
        return failure

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
        str: 执行输出文本；安全拒绝/文件缺失/运行异常均返回**结构化错误文本**
        （首行带错误类型标记，便于 Agent 与工具执行节点识别失败），
        不向外抛异常，保证 Agent 链路不中断。
    """
    try:
        return run_python(code, dataset_path)
    except SandboxSecurityError as e:
        # 安全拒绝视为「可改写的参数问题」：模型完全可以换一种合规写法继续分析，
        # 因此标记为可重试，并在建议里明确指出该怎么做
        return tool_error(
            KIND_SECURITY,
            f"代码被安全策略拒绝：{e}",
            hint="请改用白名单内的库（pandas/numpy 等）与合规写法完成同一分析，"
                 "不要访问文件系统、网络或双下划线属性。",
        )
    except PermissionError as e:
        # 数据文件路径不在允许的数据目录内（路径穿越/任意文件读取拦截）
        return tool_error(
            KIND_SECURITY,
            f"数据路径越界：{e}",
            hint=f"请改用允许目录内的数据集路径（当前为 {dataset_path!r}）。",
            retryable=False,
        )
    except FileNotFoundError as e:
        return tool_error(
            KIND_NOT_FOUND,
            f"数据集不存在：{e}",
            hint="请确认 dataset_path 指向 datasets/ 目录下真实存在的文件。",
            retryable=False,
        )
    except Exception as e:  # 兜底，避免工具抛异常中断 Agent 链路
        kind, retryable = describe_exception(e)
        return tool_error(
            kind,
            f"执行出错：{type(e).__name__}: {e}",
            hint="请检查代码逻辑后重试。",
            retryable=retryable,
        )
