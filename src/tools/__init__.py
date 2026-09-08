"""分析工具集合。

每个工具都以 LangChain `@tool` 封装，同时底层函数（以 `_` 前缀）返回 dict，
供 Agent 节点直接调用并写入 State，也便于单元测试。
"""

from src.tools.chart_tool import generate_chart
from src.tools.file_tools import (
    inspect_schema,
    load_dataframe,
    profile_dataset,
    read_dataset,
)
from src.tools.python_tool import execute_python
from src.tools.rag_tool import retrieve_knowledge
from src.tools.report_tool import save_report
from src.tools.sql_tool import execute_sql
from src.tools.stats_tools import (
    calculate_correlation,
    calculate_statistics,
    detect_outliers,
)

# 所有可供 Agent 调用的工具（Tool Calling 用）
ALL_TOOLS = [
    read_dataset,
    inspect_schema,
    profile_dataset,
    execute_python,
    execute_sql,
    calculate_statistics,
    calculate_correlation,
    detect_outliers,
    generate_chart,
    retrieve_knowledge,
    save_report,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def get_all_tools() -> list:
    """普通 LangChain 工具 + MCP 工具（动态发现，MCP 未启用则为空）。"""
    from src.mcp import get_mcp_tools

    return list(ALL_TOOLS) + list(get_mcp_tools())


def get_tools_by_name() -> dict:
    """返回所有可用工具（含 MCP）按名称索引的字典。"""
    return {t.name: t for t in get_all_tools()}


__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "get_all_tools",
    "get_tools_by_name",
    "read_dataset",
    "inspect_schema",
    "profile_dataset",
    "load_dataframe",
    "execute_python",
    "execute_sql",
    "calculate_statistics",
    "calculate_correlation",
    "detect_outliers",
    "generate_chart",
    "retrieve_knowledge",
    "save_report",
]
