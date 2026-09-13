"""MCP Client/Adapter：连接 MCP Server、发现工具、调用工具，并适配为 LangChain Tool。

真实调用链：
    Agent → MCP Client (stdio) → MCP Server (子进程) → Tool → Result → Agent

Client 用后台 asyncio 线程维持与 MCP Server 的持久化 stdio 连接，避免每次调用都重新启动
（MCP Server 需导入 pandas/numpy 等重依赖，启动成本较高）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.config.settings import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)


def mcp_stdio_runtime_check() -> str | None:
    """检查当前环境是否具备运行 MCP stdio 链路的条件。

    Windows 上 mcp 的 stdio 传输会无条件导入 pywintypes（来自 pywin32）；
    缺失时子进程一启动就崩溃，客户端只能收到含义不明的 "Connection closed"。
    本函数把该问题前置为一条可读原因，供测试 skip 与启动诊断使用。

    Returns:
        不可用原因字符串；环境就绪时返回 None。
    """
    if sys.platform == "win32":
        try:
            import pywintypes  # noqa: F401  (仅做可用性探测，不需要使用该模块)
        except ImportError as e:
            return (
                "Windows 上 MCP stdio 传输依赖 pywin32（import pywintypes 失败："
                f"{e}），请执行 `pip install pywin32` 后重试。"
            )
    return None


def _result_to_text(result) -> str:
    """把 MCP CallToolResult 转成文本。

    MCP 工具返回的 content 是内容块列表（文本块/结构化块等）：
    优先拼接其中的 text 块；没有文本块时退化为把 structured_content
    序列化成 JSON 字符串，保证 Agent 总能拿到一段可读文本。

    Args:
        result: MCP SDK 返回的 CallToolResult 对象。

    Returns:
        供 LLM 阅读的工具结果纯文本。
    """
    parts = []
    # 优先提取 content 内容块里的 text 字段（stdio JSON-RPC 响应解析后的对象）
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    # 无文本块时用结构化内容兜底（default=str 兼容日期等不可直接 JSON 化的对象）
    structured = getattr(result, "structured_content", None)
    return json.dumps(structured, ensure_ascii=False, default=str) if structured else ""


class McpClient:
    """持久化 MCP stdio 客户端。

    MCP 的异步 SDK 跑在 asyncio 事件循环上，而 Agent 主体是同步代码，
    因此用「一个后台守护线程 + 一条常驻事件循环」承接所有协程；
    同步方法通过 run_coroutine_threadsafe 把调用投递到该循环并阻塞等待结果。
    与 Server 子进程的 stdio 连接只建立一次并长期复用。
    """

    def __init__(self, params: StdioServerParameters, timeout: float = 120.0) -> None:
        """初始化客户端（此时尚未启动子进程，首次调用时才懒连接）。

        Args:
            params: stdio 子进程启动参数（命令、参数、工作目录、编码）。
            timeout: 每次 RPC 调用的等待超时秒数（分析类工具可能较慢）。
        """
        self._params = params
        self._timeout = timeout
        # 后台事件循环与承载它的线程（start 后才有值）
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # MCP 会话与 stdio_client 异步上下文（持有子进程及其 stdin/stdout 管道）
        self._session: ClientSession | None = None
        self._ctx = None
        # server 子进程 stderr 的落地日志（握手失败时据此给出真实报错，而非“Connection closed”）
        self._errlog = None
        # 常驻「连接任务」及其关闭信号：anyio 的 cancel scope 要求 __aenter__ 与
        # __aexit__ 必须在同一个 Task 内执行，因此连接、等待关闭、断连必须放在
        # 同一个长任务 _connection_runner 里，不能拆成两个被分别提交的协程
        self._runner_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None
        # start() 与连接任务之间的握手同步：ready 表示连接成功或启动失败已落定
        self._ready = threading.Event()
        # runner_done 表示连接任务的 finally（含断连/子进程终止/日志清理）已全部完成
        self._runner_done = threading.Event()
        self._start_error: BaseException | None = None
        # 保护 start/close 并发调用的互斥锁
        self._lock = threading.Lock()

    def start(self) -> None:
        """启动后台线程并建立 stdio 连接（幂等，可重复调用）。

        连接在后台循环的「同一个任务」内完成建立；启动失败（如缺 pywin32、
        server 导入错误）会把带 server stderr 的异常同步抛给调用方。
        """
        with self._lock:
            # 已有事件循环说明连接已建立，直接返回
            if self._loop is not None:
                return
            # 新建独立事件循环并让守护线程永久运行，供协程投递使用
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._ready.clear()
            self._runner_done.clear()
            self._start_error = None
            self._thread.start()
            # 连接任务承载完整生命周期：建立连接 → 等关闭信号 → 断连（同一 Task）
            asyncio.run_coroutine_threadsafe(self._connection_runner(), self._loop)
            # 阻塞等待握手结果；超时则停掉循环并向上报错
            if not self._ready.wait(timeout=self._timeout):
                self._stop_loop()
                raise TimeoutError("MCP server 连接超时")
            if self._start_error is not None:
                # 启动失败也要等连接任务的清理（终止子进程、删日志）结束再停循环
                self._runner_done.wait(timeout=10)
                self._stop_loop()
                raise self._start_error

    async def _connection_runner(self) -> None:
        """后台常驻任务：建立连接 → 等待关闭信号 → 在同一任务内断连。

        anyio（stdio_client / ClientSession 内部的任务组与 cancel scope）要求
        异步上下文的进入与退出发生在同一 Task，因此本函数是唯一允许调用
        ``_connect`` / ``_disconnect`` 的地方；RPC 调用则可由其他任务经
        ClientSession 发起（协议层本身是跨任务安全的）。
        """
        # Event 在循环线程内创建，确保绑定到本后台事件循环
        self._shutdown_event = asyncio.Event()
        self._runner_task = asyncio.current_task()
        try:
            # 拉起子进程并完成 MCP 握手（失败会抛带 stderr 的 RuntimeError）
            await self._connect()
            # 通知 start()：连接已就绪
            self._ready.set()
            # 挂起直到 close() 设置关闭信号
            await self._shutdown_event.wait()
        except BaseException as e:  # noqa: BLE001
            # 记录启动阶段的异常交给 start() 重抛；就绪后被取消等情况仅记录
            self._start_error = e
            self._ready.set()
        finally:
            # 关键：与 _connect 在同一个 Task 内退出，cancel scope 不会跨任务
            with contextlib.suppress(Exception):
                await self._disconnect()
            self._runner_task = None
            self._shutdown_event = None
            # 通知等待方（start 失败路径）：清理已全部完成
            self._runner_done.set()

    def _stop_loop(self) -> None:
        """停止后台事件循环与线程并清空引用（调用方须持有 _lock）。"""
        loop, self._loop = self._loop, None
        self._thread = None
        if loop is not None and loop.is_running():
            # 线程安全地请求循环停止，后台守护线程随之退出
            loop.call_soon_threadsafe(loop.stop)

    async def _request_shutdown(self) -> None:
        """（在后台循环上执行）设置关闭信号并等待连接任务完成断连。"""
        event = self._shutdown_event
        if event is not None:
            event.set()
        # 必须 await 连接任务结束，确保子进程已终止、stderr 日志已清理，
        # 然后才能停止事件循环
        if self._runner_task is not None:
            await self._runner_task

    async def _connect(self) -> None:
        """（协程）启动 stdio 子进程、建立读写管道并完成 MCP 会话初始化。

        子进程的 stderr 重定向到临时日志文件：一旦启动/握手失败，立即读出
        其中的真实报错（如 ImportError、缺 pywin32、端口占用等）并随异常抛出，
        避免调用方只收到无法定位原因的 "Connection closed"。
        断连清理由调用方 _connection_runner 的 finally 统一在同一任务内完成。
        """
        # 文本模式临时文件，承载 server 端全部 stderr 输出（协议数据走 stdout 不受影响）
        self._errlog = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="replace",
            suffix=".log",
            prefix="mcp-server-stderr-",
            delete=False,
        )
        try:
            # stdio_client 按 params 拉起子进程，read/write 即其 stdout/stdin 消息流；
            # errlog 承接子进程 stderr（mcp 2.x 通过该参数传入）
            self._ctx = stdio_client(self._params, errlog=self._errlog)
            read, write = await self._ctx.__aenter__()
            # ClientSession 在管道之上封装 JSON-RPC（initialize / tools/list / tools/call）
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            # 发送 MCP initialize 请求，协商协议版本与能力
            await self._session.initialize()
        except Exception as e:
            # 握手失败：取出 server stderr 尾部，组装可直接定位的错误信息后重抛
            detail = self._read_server_stderr()
            raise RuntimeError(
                "MCP server 启动或握手失败"
                f"（command={self._params.command} args={self._params.args}）：{e}\n"
                f"---- server stderr ----\n{detail or '(无输出)'}"
            ) from e

    def _read_server_stderr(self) -> str:
        """读取 server 子进程 stderr 日志的尾部内容（最多约 3000 字符）。

        Returns:
            stderr 文本尾部；日志不可读时返回占位说明。
        """
        if self._errlog is None:
            return ""
        try:
            # flush 子进程/缓冲区已写入内容，再按文件路径重新读取，
            # 避免与子进程共享同一文件句柄时的读写位置问题
            self._errlog.flush()
            return Path(self._errlog.name).read_text(
                encoding="utf-8", errors="replace"
            )[-3000:]
        except OSError:
            return "(无法读取 server stderr 日志文件)"

    async def _disconnect(self) -> None:
        """（协程）按与会话、stdio 上下文相反的顺序关闭，释放子进程管道。"""
        try:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
                self._session = None
            if self._ctx is not None:
                await self._ctx.__aexit__(None, None, None)
                self._ctx = None
        finally:
            # stderr 日志清理必须放在 finally：server 崩溃时 ctx 退出可能抛异常，
            # 若清理写在后面会被跳过，导致临时文件与文件句柄泄漏
            if self._errlog is not None:
                log_path = self._errlog.name
                try:
                    self._errlog.close()
                except OSError:
                    logger.debug("关闭 MCP server stderr 日志失败", exc_info=True)
                finally:
                    self._errlog = None
                    # Windows 上子进程传输层关闭句柄依赖事件循环回调，且作业对象
                    # 终止存在短暂延迟，因此用「异步」退避重试删除（不能用
                    # time.sleep——它会冻结事件循环，句柄释放回调永远没机会执行）；
                    # 最终仍失败则放弃，交由系统临时目录清理
                    for attempt in range(20):
                        try:
                            os.unlink(log_path)
                            break
                        except OSError:
                            if attempt == 19:
                                logger.debug(
                                    "删除 MCP server stderr 临时日志失败：%s", log_path
                                )
                            else:
                                # 让出事件循环，使底层传输的句柄关闭回调得以执行
                                await asyncio.sleep(0.05)

    def _submit(self, coro):
        """把协程投递到后台事件循环执行，并同步等待其结果。

        Args:
            coro: 需要在后台循环上运行的协程对象。

        Returns:
            协程的返回值。

        Raises:
            协程内异常或等待超时会向上抛出（由调用方处理）。
        """
        # 线程安全地把协程提交给另一个线程里的事件循环
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # 阻塞同步代码直到 RPC 返回，超时由 self._timeout 控制
        return fut.result(timeout=self._timeout)

    def list_tools(self) -> list[dict]:
        """发现 MCP Server 提供的工具及其 schema。

        通过 JSON-RPC 的 tools/list 拿到服务端注册的全部工具元数据。

        Returns:
            字典列表，每项含 name（工具名）、description（描述）、
            input_schema（JSON Schema 形式的参数定义）。
        """
        self.start()  # 懒连接：首次发现工具时才真正拉起子进程
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
        """调用 MCP 工具，返回结果文本；服务端报错时抛异常。

        对应 JSON-RPC 的 tools/call：参数以 JSON 对象发送给 Server 子进程。

        Args:
            name: MCP Server 注册的工具名。
            arguments: 传给工具的参数字典。

        Returns:
            工具执行后的结果文本。

        Raises:
            RuntimeError: 当返回结果 is_error=True（工具执行失败）时抛出。
        """
        self.start()
        result = self._submit(self._session.call_tool(name, arguments))
        text = _result_to_text(result)
        # MCP 协议用 is_error 标记工具级错误（与传输层异常区分）
        if getattr(result, "is_error", False):
            raise RuntimeError(text)
        return text

    def close(self) -> None:
        """关闭连接、终止 Server 子进程并停止后台线程（幂等）。"""
        with self._lock:
            loop = self._loop
            # 尚未启动（或启动失败已清理）则无事可做
            if loop is None:
                return
            try:
                # 在连接任务「同一个 Task」内发出关闭信号并等待其完成断连，
                # 不能直接提交 _disconnect()（跨任务退出 cancel scope 会报错，
                # 旧实现因此静默泄漏 Server 子进程）
                fut = asyncio.run_coroutine_threadsafe(self._request_shutdown(), loop)
                fut.result(timeout=10)
            except Exception:  # noqa: BLE001
                logger.debug("MCP 连接关闭异常", exc_info=True)
            finally:
                # 断连完成后再停止事件循环，后台守护线程随之退出
                self._stop_loop()


def _format_input_schema(schema: dict) -> str:
    """把 input_schema 转成 LLM 易读的参数说明。

    Args:
        schema: MCP 工具声明的 JSON Schema（properties/required）。

    Returns:
        多行参数说明文本；无参数时返回“（无参数）”。
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return "（无参数）"
    lines = []
    # 逐参数渲染：名称、类型、必填/可选、自然语言描述
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

    Args:
        client: 已可用的 McpClient。

    Returns:
        LangChain StructuredTool 列表，每个工具内部都经 stdio JSON-RPC 调用远端 MCP 工具。
    """
    tools = []
    # 工具集在运行时从 Server 动态发现，新增服务端工具无需改这里
    for meta in client.list_tools():
        name = meta["name"]
        schema_desc = _format_input_schema(meta["input_schema"])
        # 描述中显式标注 [MCP]，并附上参数说明，帮助 LLM 正确生成 arguments JSON
        description = (
            f"[MCP] {meta['description']}\n\n"
            f"参数（以 JSON 对象传入 arguments）：\n{schema_desc}"
        )

        def _call(arguments: dict, _name: str = name, _client: McpClient = client) -> str:
            # 默认参数在定义时绑定当前工具名/客户端，避免闭包晚绑定导致全部指向最后一个工具
            return _client.call_tool(_name, arguments or {})

        # 用 LangChain @tool 装饰器把同步闭包包装成带名字和描述的工具
        t = tool(f"mcp__{name}", description=description)(_call)
        tools.append(t)
    return tools


def get_server_params() -> StdioServerParameters:
    """构造 MCP Server 子进程的 stdio 启动参数。

    Returns:
        StdioServerParameters: 用当前 Python 解释器以模块方式运行
        ``src.mcp.server``，工作目录为项目根目录，消息编码 UTF-8。
    """
    return StdioServerParameters(
        command=sys.executable,  # 用与 Agent 相同的 Python 解释器，保证环境一致
        args=["-m", "src.mcp.server"],  # 以模块方式启动服务端
        cwd=str(PROJECT_ROOT),  # 固定在项目根目录运行，确保包导入路径正确
        encoding="utf-8",  # stdio JSON-RPC 消息统一按 UTF-8 编解码
    )


@lru_cache
def get_mcp_client() -> McpClient:
    """返回进程级单例 McpClient（lru_cache 保证只构造一次）。"""
    return McpClient(get_server_params())


@lru_cache
def get_mcp_tools() -> tuple:
    """返回 MCP LangChain 工具；未启用或不可用时返回空元组。

    工具发现失败（如子进程启动失败）被视为可恢复问题：记录告警后返回
    空元组，Agent 仍可只使用本地工具继续运行。
    """
    # 配置开关：未启用 MCP 时完全不拉子进程
    if not get_settings().mcp_enabled:
        return ()
    try:
        client = get_mcp_client()
        return tuple(build_mcp_tools(client))
    except Exception as e:  # noqa: BLE001
        # 连接/发现失败时优雅降级为“无 MCP 工具”
        logger.warning("MCP 不可用，跳过 MCP 工具：%s", e)
        return ()
