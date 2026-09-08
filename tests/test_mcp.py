"""MCP 测试：Server 注册、Client 发现、真实调用、错误处理、工具适配。

使用真实 MCP stdio 链路（子进程运行 src.mcp.server），非 mock。
"""

from __future__ import annotations

import asyncio

import pytest

from src.mcp.client import McpClient, build_mcp_tools, get_server_params

EXPECTED_TOOLS = {
    "read_dataset",
    "get_schema",
    "execute_sql",
    "run_analysis",
    "detect_outliers",
    "generate_chart",
    "retrieve_knowledge",
}


@pytest.fixture(scope="module")
def mcp_client():
    """模块级持久化 MCP 客户端（复用同一个 server 子进程）。"""
    client = McpClient(get_server_params())
    yield client
    client.close()


def test_server_registers_7_tools() -> None:
    from src.mcp.server import server

    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_mcp_client_discovery(mcp_client) -> None:
    tools = mcp_client.list_tools()
    assert {t["name"] for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t["description"]
        assert isinstance(t["input_schema"], dict)
        assert t["input_schema"].get("properties")


def test_mcp_client_read_dataset(mcp_client) -> None:
    r = mcp_client.call_tool("read_dataset", {"path": "datasets/sales.csv"})
    assert "num_rows" in r
    assert "2160" in r


def test_mcp_client_detect_outliers(mcp_client) -> None:
    r = mcp_client.call_tool(
        "detect_outliers",
        {"path": "datasets/sales.csv", "column": "sales", "method": "zscore"},
    )
    assert "outlier_count" in r


def test_mcp_client_error_handling(mcp_client) -> None:
    # 不存在的文件 → 返回结构化错误信息，不崩溃
    r = mcp_client.call_tool("read_dataset", {"path": "datasets/nonexistent.csv"})
    assert "读取失败" in r


def test_build_mcp_tools(mcp_client) -> None:
    tools = build_mcp_tools(mcp_client)
    # MCP 工具名带 mcp__ 前缀，与普通工具区分
    assert {t.name for t in tools} == {f"mcp__{n}" for n in EXPECTED_TOOLS}
    assert all("[MCP]" in t.description for t in tools)

    # 通过 LangChain 工具调用真实 MCP 链路
    t = {x.name: x for x in tools}["mcp__read_dataset"]
    out = t.invoke({"arguments": {"path": "datasets/sales.csv"}})
    assert "num_rows" in out
