"""MCP（Model Context Protocol）模块：Client/Adapter 侧。

- 本包只导出客户端能力（McpClient 及 LangChain 工具适配函数）。
- 服务端 src.mcp.server 由客户端以子进程方式（``python -m src.mcp.server``）
  独立启动，双方通过子进程的 stdin/stdout 交换 JSON-RPC 消息（stdio 传输）。
  因此这里刻意不导入 server，避免在主进程内拉起服务端、造成循环依赖。
"""

# 客户端：启动/持有 stdio 子进程、发现工具、调用工具，并适配为 LangChain Tool
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
