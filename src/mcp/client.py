"""MCP Client/Adapter：连接 MCP Server、发现工具、调用工具，并适配为 LangChain Tool。

真实调用链：
    Agent → MCP Client (stdio) → MCP Server (子进程) → Tool → Result → Agent

Client 用后台 asyncio 线程维持与 MCP Server 的持久化 stdio 连接，避免每次调用都重新启动
（MCP Server 需导入 pandas/numpy 等重依赖，启动成本较高）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.config.settings import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)


def _result_to_text(result) -> str:
    """把 MCP CallToolResult 转成文本。"""
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    structured = getattr(result, "structured_content", None)
    return json.dumps(structured, ensure_ascii=False, default=str) if structured else ""


class McpClient:
    """持久化 MCP stdio 客户端。"""

    def __init__(self, params: StdioServerParameters, timeout: float = 120.0) -> None:
        self._params = params
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._ctx = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """启动后台线程并建立连接（幂等）。"""
        with self._lock:
            if self._loop is not None:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
            self._submit(self._connect())

    async def _connect(self) -> None:
        self._ctx = stdio_client(self._params)
        read, write = await self._ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

    async def _disconnect(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None

    def _submit(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=self._timeout)

    def list_tools(self) -> list[dict]:
        """发现 MCP Server 提供的工具及其 schema。"""
        self.start()
        result = self._submit(self._session.list_tools())
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.input_schema or {},
            }
            for t in result.tools
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        """调用 MCP 工具，返回结果文本；服务端报错时抛异常。"""
        self.start()
        result = self._submit(self._session.call_tool(name, arguments))
        text = _result_to_text(result)
        if getattr(result, "is_error", False):
            raise RuntimeError(text)
        return text

    def close(self) -> None:
        """关闭连接并停止后台线程。"""
        with self._lock:
            loop = self._loop
            if loop is None:
                return
            self._loop = None
            self._thread = None
            try:
                fut = asyncio.run_coroutine_threadsafe(self._disconnect(), loop)
                fut.result(timeout=10)
            except Exception:  # noqa: BLE001
                logger.debug("MCP 连接关闭异常", exc_info=True)
            finally:
                loop.call_soon_threadsafe(loop.stop)


def _format_input_schema(schema: dict) -> str:
    """把 input_schema 转成 LLM 易读的参数说明。"""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return "（无参数）"
    lines = []
    for name, spec in props.items():
        ptype = spec.get("type", "string")
        desc = spec.get("description", "")
        req = "必填" if name in required else "可选"
        lines.append(f"  - {name} ({ptype}, {req}): {desc}")
    return "\n".join(lines)


def build_mcp_tools(client: McpClient):
    """发现 MCP 工具并适配为 LangChain Tool（动态，不硬编码名称）。

    LangChain 工具名加 ``mcp__`` 前缀，与普通工具明确区分：
    普通工具 ``read_dataset``（本地函数）vs MCP 工具 ``mcp__read_dataset``（走 MCP 协议）。
    """
    tools = []
    for meta in client.list_tools():
        name = meta["name"]
        schema_desc = _format_input_schema(meta["input_schema"])
        description = (
            f"[MCP] {meta['description']}\n\n"
            f"参数（以 JSON 对象传入 arguments）：\n{schema_desc}"
        )

        def _call(arguments: dict, _name: str = name, _client: McpClient = client) -> str:
            return _client.call_tool(_name, arguments or {})

        t = tool(f"mcp__{name}", description=description)(_call)
        tools.append(t)
    return tools


def get_server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp.server"],
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
    )


@lru_cache
def get_mcp_client() -> McpClient:
    return McpClient(get_server_params())


@lru_cache
def get_mcp_tools() -> tuple:
    """返回 MCP LangChain 工具；未启用或不可用时返回空元组。"""
    if not get_settings().mcp_enabled:
        return ()
    try:
        client = get_mcp_client()
        return tuple(build_mcp_tools(client))
    except Exception as e:  # noqa: BLE001
        logger.warning("MCP 不可用，跳过 MCP 工具：%s", e)
        return ()
