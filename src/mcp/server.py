"""MCP Server：通过 Model Context Protocol（stdio）暴露数据分析能力。

真实 MCP Server：复用 src/tools 与 src/rag 的底层实现（安全沙箱/只读 SQL 等），
通过 stdio 传输协议对外提供工具。

启动方式：
    python -m src.mcp.server
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from src.rag import get_retriever
from src.tools.chart_tool import generate_chart_file
from src.tools.file_tools import _inspect_schema, _read_dataset
from src.tools.python_tool import run_python
from src.tools.sql_tool import run_sql
from src.tools.stats_tools import (
    calculate_correlation,
    detect_outliers as detect_outliers_tool,
)

server = MCPServer(
    name="data-analysis",
    title="Data Analysis MCP Server",
    instructions="数据分析能力：读取数据、schema、只读 SQL、Python 沙箱分析、异常检测、图表、知识库检索。",
)


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


@server.tool(name="read_dataset", description="读取数据文件，返回行数、列数、字段与前几行示例。")
def read_dataset(path: str) -> str:
    """读取数据文件。

    Args:
        path: 数据文件路径（支持 .csv / .xlsx / .json）。
    """
    try:
        return _json(_read_dataset(path))
    except Exception as e:  # noqa: BLE001
        return f"读取失败：{type(e).__name__}: {e}"


@server.tool(name="get_schema", description="查看数据字段名称与数据类型。")
def get_schema(path: str) -> str:
    """查看数据字段与类型。

    Args:
        path: 数据文件路径。
    """
    try:
        return _json(_inspect_schema(path))
    except Exception as e:  # noqa: BLE001
        return f"获取 schema 失败：{type(e).__name__}: {e}"


@server.tool(name="execute_sql", description="在 PostgreSQL 上执行只读 SQL（仅 SELECT/WITH/EXPLAIN）。")
def execute_sql(sql: str) -> str:
    """执行只读 SQL。

    Args:
        sql: 只读 SQL 语句（SELECT/WITH/EXPLAIN）。
    """
    try:
        df = run_sql(sql)
        return _json(
            {
                "num_rows": int(len(df)),
                "columns": list(df.columns),
                "data": df.head(50).to_dict(orient="records"),
            }
        )
    except Exception as e:  # noqa: BLE001
        return f"SQL 执行失败：{type(e).__name__}: {e}"


@server.tool(name="run_analysis", description="在沙箱中执行 Pandas/NumPy 分析代码。")
def run_analysis(code: str, dataset_path: str) -> str:
    """在沙箱中执行 Python 分析代码。

    Args:
        code: 分析代码，数据已加载为变量 df，需用 print() 输出结果。
        dataset_path: 数据文件路径。
    """
    try:
        return run_python(code, dataset_path)
    except Exception as e:  # noqa: BLE001
        return f"执行出错：{type(e).__name__}: {e}"


@server.tool(name="detect_outliers", description="检测数值列中的异常值（zscore / iqr）。")
def detect_outliers(path: str, column: str = "", method: str = "zscore") -> str:
    """检测异常值。

    Args:
        path: 数据文件路径。
        column: 数值列名，留空检测所有数值列。
        method: zscore 或 iqr。
    """
    return detect_outliers_tool.invoke(
        {"path": path, "column": column, "method": method}
    )


@server.tool(name="generate_chart", description="生成图表并保存为 PNG（bar/line/scatter/hist/box）。")
def generate_chart(path: str, chart_type: str, x: str, y: str = "", title: str = "") -> str:
    """生成图表。

    Args:
        path: 数据文件路径。
        chart_type: 图表类型（bar/line/scatter/hist/box）。
        x: 横轴字段。
        y: 数值字段（hist 可省略）。
        title: 标题（可选）。
    """
    try:
        out = generate_chart_file(path, chart_type, x, y, title)
        return f"图表已生成: {out}"
    except Exception as e:  # noqa: BLE001
        return f"图表生成失败：{type(e).__name__}: {e}"


@server.tool(name="retrieve_knowledge", description="从数据分析知识库检索相关方法论知识。")
def retrieve_knowledge(query: str, top_k: int = 5) -> str:
    """检索知识库。

    Args:
        query: 检索问题。
        top_k: 返回条数。
    """
    try:
        results = get_retriever().retrieve(query, top_k)
    except Exception as e:  # noqa: BLE001
        return f"知识库检索失败：{type(e).__name__}: {e}"
    if not results:
        return "知识库为空或未检索到相关内容。"
    return "\n\n".join(
        f"[{r['metadata'].get('category', '?')}] (相似度 {r['score']}) {r['text'][:240]}"
        for r in results
    )


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
