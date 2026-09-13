"""Tool Calling 节点模块：实现 Agent 的「推理-行动」核心循环。

包含成对的两个节点：
- agent_node：把对话历史交给绑定了工具的 LLM，由模型决定下一步是调用工具
  还是输出最终结论（LangChain 的 tool_calls 机制）；
- tools_node：取出 LLM 最新消息中的工具调用请求，逐个真正执行，并把结果以
  ToolMessage 形式回灌，同时记录调用台账、错误、耗时与图表路径。

二者在 graph.py 中构成 agent ⇄ tools 循环，直到满足以下任一终止条件：
1. LLM 不再请求工具（正常收敛）；
2. 达到 max_steps 步数上限（防死循环）；
3. 本轮请求的工具调用**全部是已成功执行过的重复调用**（无进展，见 graph.route_after_agent）。

本模块对「失败」的处理原则：工具失败不让 Agent 崩溃，而是把**结构化的错误**
（错误类型 + 可选值 + 修复建议 + 是否可重试）回传给 LLM，让它有机会改参数重试；
同时对「同一工具连续失败」「同一步发起过多调用」设上限，防止把步数预算耗在
无效重试上。
"""

from __future__ import annotations

# 并发执行工具调用（每个工具调用带独立超时）
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
import logging
import time

# 工具执行结果专用的消息类型，会带上 tool_call_id 与 LLM 的调用请求配对
from langchain_core.messages import AIMessage, ToolMessage

# LLM 调用容错与可观测封装
from src.agent.observability import AgentLLMError, invoke_llm
# 图共享状态类型
from src.agent.state import AgentState
# 历史裁剪：控制每轮请求携带的上下文规模
from src.agent.utils import trim_messages_for_llm
# 全局配置
from src.config.settings import get_settings
# LLM 工厂
from src.llm import get_llm
# 工具注册表：get_all_tools 用于绑定到 LLM，get_tools_by_name 用于按名执行
from src.tools import get_all_tools, get_tools_by_name
# 结构化工具错误
from src.tools.errors import (
    KIND_TIMEOUT,
    KIND_UNKNOWN_TOOL,
    is_tool_error,
    tool_error,
)

logger = logging.getLogger(__name__)

# 工具输出中代表「成功生成了图表」的标记（chart_tool 成功时输出该前缀）
CHART_MARKER = "图表已生成:"

# 单次工具调用的超时控制线程池。工具大多是 IO/CPU 混合的短任务，
# 线程数取 4 足以覆盖单步并行调用，又不会无限膨胀。
_TOOL_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-exec")


def _extract_chart_path(result: str) -> str | None:
    """从图表工具的输出文本中提取图表文件路径。

    图表工具成功时会输出形如「图表已生成: /path/to/xxx.png」的文本，
    本函数按固定标记切出路径，用于登记到 generated_charts 白名单。

    :param result: 工具返回给 LLM 的字符串结果
    :return: 提取到的路径字符串；输出中不含标记时返回 None
    """
    marker = CHART_MARKER
    if marker in result:
        # 以标记为界取后半段并去除首尾空白
        return result.split(marker, 1)[1].strip()
    return None


def is_chart_tool(name: str) -> bool:
    """判断工具名是否为「生成图表」（本地或 MCP 版本）。

    同时匹配 ``generate_chart`` 与 ``mcp__generate_chart``：旧实现只认本地名，
    导致通过 MCP 链路生成的图表不会进入报告白名单，报告里明明有图却说「未生成图表」。

    :param name: LLM 请求调用的工具名
    :return: 是图表工具返回 True
    """
    return name.removeprefix("mcp__") == "generate_chart"


def call_signature(name: str, args: dict) -> str:
    """生成一次工具调用的规范化签名，用于识别重复调用。

    参数字典的键顺序不影响语义，因此先排序再序列化；
    无法序列化的值退化为字符串表示，保证签名总能生成。

    :param name: 工具名
    :param args: 调用参数字典
    :return: 可用于相等比较的签名字符串
    """
    try:
        # sort_keys 保证 {"a":1,"b":2} 与 {"b":2,"a":1} 得到同一签名
        payload = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(args)
    return f"{name}#{payload}"


def executed_signatures(state: AgentState) -> set[str]:
    """收集此前**成功执行过**的工具调用签名。

    只统计成功的调用：失败调用应当允许 LLM 改参数重试，
    若把失败也计入「已执行」，模型就再也没有纠正的机会。

    :param state: 图当前共享状态
    :return: 成功调用过的签名集合
    """
    signatures: set[str] = set()
    for tc in state.get("tool_calls") or []:
        if tc.get("status") == "success":
            signatures.add(call_signature(tc.get("name", ""), tc.get("args") or {}))
    return signatures


def _consecutive_failures(state: AgentState) -> dict[str, int]:
    """统计每个工具当前的「连续失败次数」（按工具结果时间顺序倒推）。

    用于在错误信息里给出「你已经在同一个工具上失败 N 次」的提示，
    引导模型换工具或换思路，而不是继续微调参数空转。

    :param state: 图当前共享状态
    :return: {工具名: 连续失败次数}，连续成功过的工具不会出现在结果中
    """
    counter: dict[str, int] = {}
    # 已经确定「最近一次是成功」的工具，不再往更早的历史回溯
    stopped: set[str] = set()
    for tr in reversed(state.get("tool_results") or []):
        name = tr.get("name", "")
        if name in stopped:
            continue
        if tr.get("status") == "success":
            stopped.add(name)
            continue
        counter[name] = counter.get(name, 0) + 1
    return counter


def _execute_one(tool, args: dict) -> tuple[str, str | None, int]:
    """执行单个工具调用，带超时保护与异常收敛。

    :param tool: LangChain 工具对象
    :param args: 调用参数字典
    :return: 三元组 (回传给 LLM 的文本, 错误信息或 None, 耗时毫秒)
    """
    settings = get_settings()
    started = time.monotonic()
    try:
        # 用线程池包一层：工具本身是同步函数，超时后主流程不再等待。
        # 注意：超时只是「不再等它」，后台线程可能仍在运行（Python 无法强杀线程），
        # 因此这里依赖各工具自身也有更细粒度的超时（如 SQL statement_timeout、
        # Python 沙箱进程超时），本层是兜底而非唯一防线。
        future = _TOOL_POOL.submit(tool.invoke, args)
        result = future.result(timeout=settings.tool_timeout_seconds)
        content = result if isinstance(result, str) else str(result)
        elapsed = int((time.monotonic() - started) * 1000)
        return content, None, elapsed
    except FutureTimeout:
        elapsed = int((time.monotonic() - started) * 1000)
        message = tool_error(
            KIND_TIMEOUT,
            f"工具执行超过 {settings.tool_timeout_seconds}s 未返回，已放弃本次调用",
            hint="请缩小数据范围（加筛选条件 / 减少列数）后重试。",
        )
        return message, message, elapsed
    except Exception as e:  # noqa: BLE001 —— 工具异常转为可读文本，保证图不断
        elapsed = int((time.monotonic() - started) * 1000)
        content = f"工具执行异常：{type(e).__name__}: {e}"
        return content, content, elapsed


def agent_node(state: AgentState) -> dict:
    """Agent 决策节点：LLM 观察当前上下文，决定调用工具或输出结论。

    本节点不直接执行任何分析动作，只负责「思考与决策」。两点关键处理：

    1. **上下文裁剪**：每轮都重新发送全部历史会让请求体随步数线性膨胀
       （工具结果可达数千字符），既费 token 也容易触发上下文超限。
       因此对较早的工具结果做占位替换，只保留最近若干条原文。
    2. **LLM 故障降级**：调用失败（重试后仍失败）时不抛出中断整张图，
       而是返回一条不带 tool_calls 的消息，让流程转入 insight 节点，
       由下游产出「分析不完整」的明确结论，而不是让用户拿到一个崩溃栈。

    :param state: 图当前共享状态，必须包含 messages
    :return: 状态增量：messages 累加 LLM 回复，step_count 加 1，
             llm_calls 累加本次调用记录
    """
    llm = get_llm()
    # bind_tools 把工具的名称/参数 schema 告知 LLM，使其能产出结构化 tool_calls
    llm_with_tools = llm.bind_tools(get_all_tools())

    # 裁剪历史：完整保留系统提示与最近的工具结果，更早的结果替换为占位摘要
    messages = trim_messages_for_llm(
        state["messages"], keep_recent_tool_results=get_settings().keep_recent_tool_results
    )

    try:
        response, record = invoke_llm(messages, llm=llm_with_tools, node="agent")
    except AgentLLMError as e:
        # 模型不可用：不再请求工具，直接进入收尾流程并留下降级标记。
        # 同时记一条失败的调用记录——否则「调用了几次、失败几次」的运行级
        # 统计会漏掉最严重的那类失败，可观测性就失去意义了。
        logger.error("Agent 决策阶段 LLM 不可用，转入降级收尾：%s", e)
        return {
            "messages": [AIMessage(content=f"[LLM 不可用] {e}")],
            "step_count": state.get("step_count", 0) + 1,
            "errors": [str(e)],
            "llm_calls": [{"node": "agent", "success": False, "error": str(e)}],
            "degraded": ["LLM 调用失败，分析在未完成全部步骤的情况下提前收尾。"],
        }

    return {
        "messages": [response],
        # 每完成一次 LLM 决策即计一步，与 max_steps 配合防止无限循环
        "step_count": state.get("step_count", 0) + 1,
        "llm_calls": [record],
    }


def tools_node(state: AgentState) -> dict:
    """工具执行节点：执行上一条 AIMessage 中的全部工具调用并回传结果。

    处理要点：
    - **数量上限**：单步请求超过 max_tool_calls_per_step 时，超出的调用直接
      回一条「未执行」说明，防止一次决策就把整轮预算烧掉；
    - **重复调用**：与已成功执行过的调用签名相同时不再真正执行，直接复用
      首次结果并提示 LLM 换方法（既省算力，也让循环尽快收敛）；
    - **连续失败提示**：同一工具连续失败达阈值时，在错误文本后追加显式指令，
      告诉模型停止在该工具上继续尝试；
    - **超时保护**：每个工具调用独立超时（tool_timeout_seconds）；
    - **图表登记**：本地与 MCP 两个图表工具成功产出的文件都记入白名单。

    :param state: 图当前共享状态，messages 末尾应为带 tool_calls 的 AIMessage
    :return: 状态增量：messages 累加 ToolMessage 列表，tool_calls/
             tool_results/errors/generated_charts 分别累加本批次台账
    """
    settings = get_settings()
    # 最后一条消息即 agent_node 产出的工具调用请求
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    # 预分配本批次要累加的各类容器
    messages: list[ToolMessage] = []
    new_tool_calls: list[dict] = []
    new_tool_results: list[dict] = []
    new_errors: list[str] = []
    new_generated_charts: list[str] = []

    # 工具名 → 工具实例的映射，循环中按 LLM 给的名称查找
    tools_by_name = get_tools_by_name()
    # 已成功执行过的调用签名：用于识别重复调用
    already_done = executed_signatures(state)
    # 本轮内部也要去重：同一步里重复请求同一个调用同样只执行一次
    seen_this_step: dict[str, str] = {}
    # 各工具的历史连续失败次数：用于给出「别再试了」的提示
    failures = _consecutive_failures(state)

    for index, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        # args 可能为 None，统一兜底成空字典，避免下方解包/传参报错
        args = tc.get("args", {}) or {}
        tool = tools_by_name.get(name)
        signature = call_signature(name, args)
        status = "error"
        error: str | None = None
        duration_ms = 0

        # ---- 分支 1：单步调用数量超限，不再执行 ----
        if index >= settings.max_tool_calls_per_step:
            content = tool_error(
                KIND_UNKNOWN_TOOL,
                f"单步最多执行 {settings.max_tool_calls_per_step} 个工具调用，"
                f"本次的第 {index + 1} 个（{name}）已被跳过",
                hint="请把分析拆成更小的步骤，分批调用工具。",
                retryable=False,
            )
            error = content

        # ---- 分支 2：重复调用，复用首次成功的结果 ----
        elif signature in already_done or signature in seen_this_step:
            previous = seen_this_step.get(signature) or _find_previous_result(state, signature)
            content = (
                "该工具调用与之前成功执行过的调用完全相同，已直接返回首次结果"
                "（未重复执行）。请基于这些数据继续下一步，或换一个不同的查询/分析角度。\n\n"
                f"首次结果：\n{previous}"
            )
            # 重复调用按成功记录：数据本身是有效的，只是不该重跑
            status = "success"

        # ---- 分支 3：工具名不存在（模型幻觉） ----
        elif tool is None:
            content = tool_error(
                KIND_UNKNOWN_TOOL,
                f"未知工具: {name}",
                options=sorted(tools_by_name.keys()),
                hint="请从上列工具名中选择一个重新调用。",
            )
            error = content

        # ---- 分支 4：正常执行 ----
        else:
            content, error, duration_ms = _execute_one(tool, args)
            status = "error" if error else "success"
            # 工具按契约「失败也不抛异常」，而是返回结构化错误文本。
            # 因此必须识别这种文本并计入错误台账，否则「连续失败」统计会漏掉
            # 最常见的一类失败（参数写错、字段不存在、被安全策略拒绝等），
            # 中止重试机制也就形同虚设。
            if error is None and is_tool_error(content):
                status = "error"
                error = content

        # 连续失败达阈值的工具，追加显式停止指令（只提示一次，避免信息重复）
        if status == "error" and failures.get(name, 0) + 1 >= settings.tool_max_consecutive_failures:
            content += (
                f"\n\n【重试上限】工具 {name} 已连续失败 "
                f"{failures.get(name, 0) + 1} 次。请不要再重试该工具，"
                "改用其他工具或换一种分析思路；若确实无法获取该数据，"
                "请在结论中说明该项未能完成。"
            )
        # 本轮内同一签名的结果缓存，供后续重复请求复用
        if status == "success":
            seen_this_step.setdefault(signature, content)

        # ToolMessage 必须带 tool_call_id，LangChain 据此与请求一一配对
        messages.append(
            ToolMessage(content=content, tool_call_id=tc.get("id", ""), name=name)
        )
        # 台账：调用记录（含入参/耗时）与结果记录（含返回内容/状态）分开存放
        new_tool_calls.append(
            {"name": name, "args": args, "status": status, "duration_ms": duration_ms}
        )
        new_tool_results.append({"name": name, "result": content, "status": status})
        if error:
            # 只取错误首行（含错误类型标记）入库，完整文本仍随 tool_results 回传 LLM
            new_errors.append(f"{name}: {error.splitlines()[0]}")

        # 记录实际成功生成的图表路径。
        # 覆盖 mcp__generate_chart：否则走 MCP 链路出图时报告会误报「未生成图表」
        if is_chart_tool(name) and status == "success":
            chart_path = _extract_chart_path(content)
            if chart_path:
                new_generated_charts.append(chart_path)

    return {
        "messages": messages,
        "tool_calls": new_tool_calls,
        "tool_results": new_tool_results,
        "errors": new_errors,
        "generated_charts": new_generated_charts,
    }


def _find_previous_result(state: AgentState, signature: str) -> str:
    """按签名在历史结果中回查首次成功执行的返回文本。

    :param state: 图当前共享状态
    :param signature: 目标调用的签名
    :return: 首次成功的工具结果文本；找不到时返回占位说明
    """
    calls = state.get("tool_calls") or []
    results = state.get("tool_results") or []
    for tc, tr in zip(calls, results):
        if call_signature(tc.get("name", ""), tc.get("args") or {}) == signature:
            return str(tr.get("result", ""))
    return "(未找到首次结果)"
