"""分析工具集合（tools 包入口）。

模块职责：
- 汇总 Agent 可调用的全部分析工具（数据读取、画像、Python/SQL 执行、统计、
  图表、RAG 知识检索、报告保存），对外暴露统一的工具清单与按名索引。
- 每个工具都以 LangChain `@tool` 封装：对外（Tool Calling）输入输出均为字符串
  （多为 JSON 字符串或人类可读提示）；同时底层函数（以 `_` 前缀）返回 dict，
  供 Agent 节点直接调用并写入 State，也便于单元测试。
- get_all_tools() 会额外拼接动态发现的 MCP 工具（MCP 未启用时为空列表）。
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

# 所有可供 Agent 调用的本地工具（Tool Calling 用）；顺序即默认注册顺序
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

# 本地工具按工具名索引，便于 Agent 节点直接按名取用
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def get_all_tools() -> list:
    """返回全部可用工具：本地 LangChain 工具 + 动态发现的 MCP 工具。

    返回值：
        list: 工具对象列表；MCP 未配置/未启用时 get_mcp_tools() 返回空列表。
    """
    # 延迟导入，避免 tools 包在 MCP 依赖缺失时无法加载
    from src.mcp import get_mcp_tools

    return list(ALL_TOOLS) + list(get_mcp_tools())


def get_tools_by_name() -> dict:
    """返回所有可用工具（含 MCP）按名称索引的字典。

    返回值：
        dict: {工具名: 工具对象}，重名时后注册者覆盖前者。
    """
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
