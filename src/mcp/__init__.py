"""MCP 模块：Client/Adapter（服务端由 src.mcp.server 作为独立进程运行，不在此导入）。"""

from src.mcp.client import (
    McpClient,
    build_mcp_tools,
    get_mcp_client,
    get_mcp_tools,
    get_server_params,
)

__all__ = [
    "McpClient",
    "build_mcp_tools",
    "get_mcp_client",
    "get_mcp_tools",
    "get_server_params",
]
