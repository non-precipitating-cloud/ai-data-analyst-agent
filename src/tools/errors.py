"""工具错误的结构化表达（供 Tool Calling 自我纠错使用）。

背景：工具失败时如果只回一句自由文本，LLM 很难判断「这是参数写错了、还是
数据本身为空、还是环境没起来」，于是容易反复重试同一个错误调用。本模块把
工具错误统一成一种**既给人看、也给模型看**的格式：首行带机器可读的错误类型
与可重试标记，后续行给出可选值与修复建议。

统一格式（实际输出为中文，此处示意结构）::

    错误[invalid_argument|retryable]：字段 'xyz' 不存在
    可选：['sales', 'profit', 'region']
    建议：请从上列字段中选择一个已存在的列名后重试

设计约束：
- 仍然是**字符串**，不改变「工具不抛异常、只返回文本」的既有契约，
  Agent 的工具循环与 MCP 适配层无需改动；
- 保留「错误」前缀与原有中文措辞，不破坏既有测试与人类可读性；
- 类型与可重试标记用固定词表，便于日志聚合与回归测试断言。
"""

from __future__ import annotations

# 错误类型词表（固定集合，避免各处随手写导致无法聚合统计）
KIND_INVALID_ARGUMENT = "invalid_argument"   # 参数缺失/类型不对/取值非法
KIND_NOT_FOUND = "not_found"                 # 文件、字段、图表等目标不存在
KIND_UNSUPPORTED = "unsupported"             # 不支持的类型或方法
KIND_PERMISSION = "permission"               # 路径越界等权限问题
KIND_DATA_EMPTY = "data_empty"               # 数据为空或筛选后无结果
KIND_DB_UNAVAILABLE = "db_unavailable"       # 数据库不可用
KIND_TIMEOUT = "timeout"                     # 执行超时
KIND_EXECUTION_ERROR = "execution_error"     # 执行期异常（代码报错等）
KIND_UNKNOWN_TOOL = "unknown_tool"           # 模型幻觉出不存在的工具名
KIND_SECURITY = "security"                   # 代码/SQL 被安全策略拒绝

# 结构化错误的首行标记：形如 "错误[not_found|non-retryable]：..."
ERROR_MARKER = "错误["


def is_tool_error(text: str) -> bool:
    """判断一段工具返回文本是否为结构化错误。

    Agent 的工具执行节点靠它区分「工具成功返回」与「工具失败返回」——
    工具按契约不抛异常，失败也只是一段文本，没有这个判断就无法统计失败、
    也就无法实现「连续失败达上限就停止重试」。

    参数：
        text: 工具返回的文本。

    返回：
        True 表示这是结构化错误文本。
    """
    return bool(text) and text.lstrip().startswith(ERROR_MARKER)


def tool_error(
    kind: str,
    message: str,
    *,
    options: list | tuple | None = None,
    hint: str = "",
    retryable: bool = True,
) -> str:
    """构造结构化的工具错误文本。

    参数：
        kind: 错误类型，取自本模块的 KIND_* 常量（也允许自定义短标识）。
        message: 面向人的中文错误描述（应包含具体值，便于定位）。
        options: 可选的有效取值列表（如可用字段名），帮助 LLM 直接改正参数。
        hint: 修复建议（一句话，说明下一步该怎么做）。
        retryable: 该错误是否值得重试。环境类问题（数据库不可用）通常
            可重试，而「文件不存在」这类重试也不会变好，标为不可重试可
            抑制 Agent 反复重试。

    返回：
        str: 首行形如 ``错误[<kind>|retryable|non-retryable]：<message>``
        的多行文本。
    """
    marker = "retryable" if retryable else "non-retryable"
    lines = [f"错误[{kind}|{marker}]：{message}"]
    if options:
        # 取值列表可能较长，截断避免错误信息本身撑爆上下文
        rendered = str(list(options))
        if len(rendered) > 500:
            rendered = rendered[:500] + " ...(更多略)"
        lines.append(f"可选：{rendered}")
    if hint:
        lines.append(f"建议：{hint}")
    return "\n".join(lines)


def describe_exception(exc: BaseException) -> tuple[str, bool]:
    """把异常归类为 (错误类型, 是否可重试)。

    用于工具层兜底捕获异常时保持一致的结构化语义，避免每个工具各写一套判断。

    参数：
        exc: 被捕获的异常对象。

    返回：
        二元组 (KIND_* 常量, 是否可重试)。
    """
    # 路径越界：换参数也不会变好，重试无意义
    if isinstance(exc, PermissionError):
        return KIND_PERMISSION, False
    if isinstance(exc, FileNotFoundError):
        return KIND_NOT_FOUND, False
    if isinstance(exc, TimeoutError):
        return KIND_TIMEOUT, True
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        # 参数/数据形状问题：属于「改参数后可重试」的典型场景
        return KIND_INVALID_ARGUMENT, True
    # 其余（连接失败、驱动异常等）多为环境问题，允许重试
    return KIND_EXECUTION_ERROR, True
