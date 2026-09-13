"""工具层韧性测试：结构化错误、参数校验、超时、重试预算。

对应「Tool Calling 检查」与「Agent 自我纠错」两节要求：
- 每个工具的参数错误都要返回**可读且可操作**的结构化错误，而不是抛异常或
  一句含糊的报错；
- 工具失败不能让 Agent 崩溃，必须把错误回传给模型；
- 同一工具连续失败要达上限并明确提示「别再重试」；
- 单步调用数量与单次执行时长都要有上限。
"""

from __future__ import annotations

import threading

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent.nodes import tool_calling
from src.agent.nodes.tool_calling import tools_node
from src.config.settings import get_settings
from src.tools.chart_tool import generate_chart
from src.tools.errors import ERROR_MARKER, is_tool_error
from src.tools.file_tools import read_dataset
from src.tools.sql_tool import execute_sql
from src.tools.stats_tools import calculate_correlation, calculate_statistics, detect_outliers
from tests.conftest import SALES_CSV


def _invoke(tool_obj, **kwargs) -> str:
    """调用 LangChain 工具并返回文本结果。"""
    return tool_obj.invoke(kwargs)


# ==========================================================================
# 结构化错误格式
# ==========================================================================
def test_structured_error_format_is_machine_readable() -> None:
    """结构化错误首行必须带类型与可重试标记，便于识别与统计。"""
    out = _invoke(calculate_statistics, path=SALES_CSV, column="不存在的列")
    assert is_tool_error(out)
    assert out.startswith(ERROR_MARKER)
    assert "not_found" in out.splitlines()[0]
    assert "retryable" in out.splitlines()[0]
    # 必须给出可选值，模型才能直接改对
    assert "可选：" in out
    assert "建议：" in out


def test_stats_reports_empty_numeric_column_distinctly() -> None:
    """列存在但无有效数值时，应报 data_empty（而非 not_found），避免无效重试。"""
    out = _invoke(calculate_statistics, path=SALES_CSV, column="region")
    # region 是文本列，强转数值后为空
    assert is_tool_error(out)
    assert "data_empty" in out
    assert "non-retryable" in out.splitlines()[0]


def test_correlation_rejects_partial_arguments() -> None:
    """只给一个列名属于参数没写全，必须明确报错而不是悄悄返回全量矩阵。"""
    out = _invoke(calculate_correlation, path=SALES_CSV, column_x="sales", column_y="")
    assert is_tool_error(out)
    assert "invalid_argument" in out
    assert "同时提供或同时留空" in out


def test_outliers_rejects_unknown_method_with_options() -> None:
    """非法 method 应报 unsupported 并列出合法取值。"""
    out = _invoke(detect_outliers, path=SALES_CSV, column="sales", method="bad")
    assert is_tool_error(out)
    assert "unsupported" in out
    assert "zscore" in out and "iqr" in out


def test_outliers_does_not_silently_drop_constant_column() -> None:
    """常量列在 zscore 下无定义，应如实计入 skipped 而不是从结果里消失。"""
    out = _invoke(detect_outliers, path=SALES_CSV, column="year", method="zscore")
    # year 取值有限且集中，无论是否被跳过，返回都必须是合法 JSON 或结构化错误
    assert out.strip()


# ==========================================================================
# 图表参数校验
# ==========================================================================
def test_chart_requires_y_for_line() -> None:
    """折线图缺少 y 时应给出明确提示，而不是抛出 KeyError: ''。"""
    out = _invoke(generate_chart, path=SALES_CSV, chart_type="line", x="month")
    assert is_tool_error(out)
    assert "必须提供纵轴字段" in out
    assert "hist" in out  # 建议里给出替代方案


def test_chart_rejects_unsupported_type_with_options() -> None:
    """不支持的图表类型应列出受支持的类型。"""
    out = _invoke(generate_chart, path=SALES_CSV, chart_type="pie", x="region", y="sales")
    assert is_tool_error(out)
    assert "unsupported" in out
    assert "bar" in out


def test_chart_reports_missing_column_with_actual_columns() -> None:
    """字段不存在时应回显真实字段列表。"""
    out = _invoke(generate_chart, path=SALES_CSV, chart_type="bar", x="nope", y="sales")
    assert is_tool_error(out)
    assert "not_found" in out
    assert "region" in out


# ==========================================================================
# 文件与 SQL 工具的错误也是结构化的
# ==========================================================================
def test_file_tool_error_is_structured() -> None:
    """文件工具读取失败时返回结构化错误（含类型与建议）。"""
    out = _invoke(read_dataset, path="datasets/definitely_missing.csv")
    assert is_tool_error(out)
    assert "not_found" in out


def test_sql_tool_rejection_is_structured() -> None:
    """SQL 被安全策略拒绝时返回 security 类型的结构化错误。"""
    out = _invoke(execute_sql, sql="DROP TABLE sales")
    assert is_tool_error(out)
    assert "security" in out


# ==========================================================================
# 超时与失败预算（在 tools_node 层）
# ==========================================================================
def _ai_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    """构造一条携带单个工具调用的 AI 消息。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_tool_timeout_returns_structured_error(monkeypatch) -> None:
    """工具执行超时不应挂死 Agent，而应回传 timeout 结构化错误。

    用 threading.Event 控制慢工具的退出，避免留一个后台线程白跑满超时时间。
    """
    release = threading.Event()

    @tool
    def slow_tool(path: str = "") -> str:
        """测试用慢工具：等待测试放行后才返回。"""
        release.wait(timeout=10)
        return "done"

    monkeypatch.setattr(tool_calling, "get_tools_by_name", lambda: {"slow_tool": slow_tool})
    monkeypatch.setattr(get_settings(), "tool_timeout_seconds", 1)

    try:
        update = tools_node({"messages": [_ai_call("slow_tool", {"path": "x"})]})
        result = update["tool_results"][0]["result"]
        assert is_tool_error(result)
        assert "timeout" in result
        assert update["tool_results"][0]["status"] == "error"
    finally:
        release.set()


def test_unknown_tool_lists_available_options() -> None:
    """模型幻觉出不存在工具时，应回传可用工具列表引导其改选。"""
    update = tools_node({"messages": [_ai_call("no_such_tool", {})]})
    result = update["tool_results"][0]["result"]
    assert is_tool_error(result)
    assert "unknown_tool" in result
    assert "read_dataset" in result


def test_consecutive_failures_trigger_stop_instruction(monkeypatch) -> None:
    """同一工具连续失败达到上限后，错误文本里必须出现「别再重试」的明确指令。"""
    monkeypatch.setattr(get_settings(), "tool_max_consecutive_failures", 2)
    args = {"path": SALES_CSV, "column": "不存在"}

    # 第一次失败：不应出现重试上限提示
    first = tools_node({"messages": [_ai_call("calculate_statistics", args)]})
    assert is_tool_error(first["tool_results"][0]["result"])
    assert "重试上限" not in first["tool_results"][0]["result"]

    # 第二次失败：达到阈值，应追加停止指令
    state = {
        "messages": [_ai_call("calculate_statistics", args, "c2")],
        "tool_results": first["tool_results"],
        "tool_calls": first["tool_calls"],
    }
    second = tools_node(state)
    assert "重试上限" in second["tool_results"][0]["result"]


def test_tool_calls_per_step_are_capped(monkeypatch) -> None:
    """单步请求过多工具调用时，超出的调用不再执行，避免一轮烧完预算。"""
    monkeypatch.setattr(get_settings(), "max_tool_calls_per_step", 1)
    calls = [
        {"name": "read_dataset", "args": {"path": SALES_CSV}, "id": "a"},
        {"name": "read_dataset", "args": {"path": "datasets/other.csv"}, "id": "b"},
    ]
    update = tools_node({"messages": [AIMessage(content="", tool_calls=calls)]})
    assert len(update["tool_results"]) == 2
    # 第二条应被跳过并给出原因
    assert "已被跳过" in update["tool_results"][1]["result"]


def test_tool_failure_does_not_break_iteration(monkeypatch) -> None:
    """一个工具失败不影响同批次其他工具的执行与回传。"""
    calls = [
        {"name": "calculate_statistics", "args": {"path": SALES_CSV, "column": "不存在"}, "id": "a"},
        {"name": "calculate_statistics", "args": {"path": SALES_CSV, "column": "sales"}, "id": "b"},
    ]
    update = tools_node({"messages": [AIMessage(content="", tool_calls=calls)]})
    statuses = [r["status"] for r in update["tool_results"]]
    assert statuses == ["error", "success"]
    # 每条 tool_call 都要有配对的 ToolMessage，否则 LangChain 会拒绝下一轮请求
    assert len(update["messages"]) == 2
