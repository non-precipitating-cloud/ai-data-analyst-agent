"""MCP（Model Context Protocol）链路测试（src.mcp）。

覆盖：
- Server 端注册的 7 个工具清单；
- Client 通过 stdio 子进程发现工具及其描述 / 入参 schema；
- 真实调用 read_dataset、detect_outliers；
- 错误输入返回结构化错误信息而不崩溃；
- build_mcp_tools 把 MCP 工具适配为带 mcp__ 前缀、[MCP] 描述的 LangChain 工具。

使用真实 MCP stdio 链路（子进程运行 src.mcp.server），非 mock。
"""

from __future__ import annotations

import asyncio

import pytest

from src.mcp.client import (
    McpClient,
    build_mcp_tools,
    get_server_params,
    mcp_stdio_runtime_check,
)

# MCP Server 应暴露的 7 个工具名
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
    # 环境不满足 stdio 运行条件时（如 Windows 缺 pywin32）明确 skip，
    # 而不是让每个用例都因含义不明的 "Connection closed" 失败
    reason = mcp_stdio_runtime_check()
    if reason:
        pytest.skip(reason)
    client = McpClient(get_server_params())
    yield client
    # 模块结束后关闭子进程连接
    client.close()


def test_server_registers_7_tools() -> None:
    """Server 端 list_tools 返回的工具名集合应恰好为约定的 7 个。"""
    from src.mcp.server import server

    # Server API 是协程，用 asyncio.run 在同步测试中驱动
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_start_failure_surfaces_server_stderr() -> None:
    """server 子进程启动即崩溃时，错误信息必须带出其 stderr 内容（而非只报 Connection closed）。"""
    import sys

    from mcp.client.stdio import StdioServerParameters

    # 假 server：启动后立刻往 stderr 写入唯一标记并退出，模拟缺依赖/导入失败
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "import sys; sys.stderr.write('MCP_BOOM_MARKER'); sys.exit(1)"],
        cwd=".",
        encoding="utf-8",
    )
    client = McpClient(params)
    _reason = mcp_stdio_runtime_check()
    if _reason:
        pytest.skip(_reason)
    with pytest.raises(RuntimeError) as exc_info:
        client.start()
    # 异常信息应同时包含失败描述与子进程 stderr 里的标记
    message = str(exc_info.value)
    assert "MCP server 启动或握手失败" in message
    assert "MCP_BOOM_MARKER" in message
    client.close()


def _running_mcp_server_pids() -> set[int]:
    """返回系统中正在运行的 src.mcp.server 子进程 PID 集合（跨平台）。

    Windows 用 CIM 查询命令行；Linux/macOS 用 pgrep -f。查询工具不可用时
    返回空集合（不因环境缺工具而阻断测试）。
    """
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            ps_cmd = (
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -like '*src.mcp.server*' } | "
                "Select-Object -ExpandProperty ProcessId"
            )
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=15,
            )
            return {int(x) for x in out.stdout.split() if x.isdigit()}
        out = subprocess.run(
            ["pgrep", "-f", "src.mcp.server"],
            capture_output=True, text=True, timeout=10,
        )
        # pgrep 无匹配时退出码为 1，stdout 为空，不应当作错误
        return {int(x) for x in out.stdout.split() if x.isdigit()}
    except (OSError, subprocess.SubprocessError):
        return set()


def test_server_process_terminated_on_close() -> None:
    """close() 必须真正终止 server 子进程（回归：跨任务退出 cancel scope 失败导致的进程泄漏）。"""
    import time

    _reason = mcp_stdio_runtime_check()
    if _reason:
        pytest.skip(_reason)
    # 以「启动前已存在的 PID 集合」为基线，避免与其他并行/模块级 client 相互干扰
    before = _running_mcp_server_pids()
    client = McpClient(get_server_params())
    client.start()
    client.list_tools()
    client.close()

    # 轮询最多约 2 秒等待作业对象/信号收尾，断言本次创建的子进程没有残留
    leaked: set[int] = set()
    for _ in range(20):
        leaked = _running_mcp_server_pids() - before
        if not leaked:
            break
        time.sleep(0.1)
    assert not leaked, f"MCP server 子进程在 close() 后仍存活：{leaked}"


def test_mcp_client_discovery(mcp_client) -> None:
    """Client 经 stdio 发现的工具名、描述与 input_schema 都应完整可用。"""
    tools = mcp_client.list_tools()
    assert {t["name"] for t in tools} == EXPECTED_TOOLS
    for t in tools:
        # 每个工具必须带描述
        assert t["description"]
        # 入参 schema 必须是 dict 且声明了 properties
        assert isinstance(t["input_schema"], dict)
        assert t["input_schema"].get("properties")


def test_mcp_client_read_dataset(mcp_client) -> None:
    """通过真实 MCP 链路调用 read_dataset，应返回行数信息（含 2160）。"""
    r = mcp_client.call_tool("read_dataset", {"path": "datasets/sales.csv"})
    assert "num_rows" in r
    assert "2160" in r


def test_mcp_client_detect_outliers(mcp_client) -> None:
    """通过真实 MCP 链路调用 detect_outliers（zscore），应返回异常数量字段。"""
    r = mcp_client.call_tool(
        "detect_outliers",
        {"path": "datasets/sales.csv", "column": "sales", "method": "zscore"},
    )
    assert "outlier_count" in r


def test_mcp_client_error_handling(mcp_client) -> None:
    """读取不存在的文件时应返回结构化中文错误信息，而不是让进程崩溃。"""
    # 不存在的文件 → 返回结构化错误信息，不崩溃
    r = mcp_client.call_tool("read_dataset", {"path": "datasets/nonexistent.csv"})
    assert "读取失败" in r


def test_build_mcp_tools(mcp_client) -> None:
    """适配出的 LangChain 工具应带 mcp__ 前缀与 [MCP] 标记，并能被真实调用。"""
    tools = build_mcp_tools(mcp_client)
    # MCP 工具名带 mcp__ 前缀，与普通工具区分
    assert {t.name for t in tools} == {f"mcp__{n}" for n in EXPECTED_TOOLS}
    assert all("[MCP]" in t.description for t in tools)

    # 通过 LangChain 工具调用真实 MCP 链路
    t = {x.name: x for x in tools}["mcp__read_dataset"]
    # 适配层把 MCP 参数包在 arguments 字段里
    out = t.invoke({"arguments": {"path": "datasets/sales.csv"}})
    assert "num_rows" in out
