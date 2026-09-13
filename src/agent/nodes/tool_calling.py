"""Tool Calling 节点模块：实现 Agent 的「推理-行动」核心循环。

包含成对的两个节点：
- agent_node：把完整对话历史交给绑定了工具的 LLM，由模型决定下一步
  是调用工具还是输出最终结论（LangChain 的 tool_calls 机制）；
- tools_node：取出 LLM 最新消息中的工具调用请求，逐个真正执行，
  并把结果以 ToolMessage 形式回灌，同时记录调用台账、错误与图表路径。

二者在 graph.py 中构成 agent ⇄ tools 循环，直到无工具调用或达到步数上限。
"""

from __future__ import annotations

# 工具执行结果专用的消息类型，会带上 tool_call_id 与 LLM 的调用请求配对
from langchain_core.messages import ToolMessage

# 图共享状态类型
from src.agent.state import AgentState
# LLM 工厂
from src.llm import get_llm
# 工具注册表：get_all_tools 用于绑定到 LLM，get_tools_by_name 用于按名执行
from src.tools import get_all_tools, get_tools_by_name


def _extract_chart_path(result: str) -> str | None:
    """从 generate_chart 工具的输出文本中提取图表文件路径。

    图表工具成功时会输出形如「图表已生成: /path/to/xxx.png」的文本，
    本函数按固定标记切出路径，用于登记到 generated_charts 白名单。

    :param result: 工具返回给 LLM 的字符串结果
    :return: 提取到的路径字符串；输出中不含标记时返回 None
    """
    marker = "图表已生成:"
    if marker in result:
        # 以标记为界取后半段并去除首尾空白
        return result.split(marker, 1)[1].strip()
    return None


def agent_node(state: AgentState) -> dict:
    """Agent 决策节点：LLM 观察当前上下文，决定调用工具或输出结论。

    本节点不直接执行任何分析动作，只负责「思考与决策」：把全部消息历史
    交给绑定工具后的 LLM，返回的 AIMessage 若带 tool_calls，条件边会把
    流程导向 tools_node；否则路由到 insight 节点收尾。

    :param state: 图当前共享状态，必须包含 messages
    :return: 状态增量：messages 累加 LLM 的回复消息，
             step_count 在原值基础上加 1（用于循环步数控制）
    """
    llm = get_llm()
    # bind_tools 把工具的名称/参数 schema 告知 LLM，使其能产出结构化 tool_calls
    llm_with_tools = llm.bind_tools(get_all_tools())
    # 把完整对话历史（含此前的工具回执）一次性传入，支持多轮推理
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        # 每完成一次 LLM 决策即计一步，与 max_steps 配合防止无限循环
        "step_count": state.get("step_count", 0) + 1,
    }


def tools_node(state: AgentState) -> dict:
    """工具执行节点：执行上一条 AIMessage 中的全部工具调用并回传结果。

    节点对每个 tool_call 按名称找到工具实例并调用；任何单个工具的异常
    都被捕获并转成文本错误回传给 LLM（而不是中断整条图），使模型有机会
    根据错误自我纠正。generate_chart 成功时还会额外登记图表路径。

    :param state: 图当前共享状态，messages 末尾应为带 tool_calls 的 AIMessage
    :return: 状态增量：messages 累加 ToolMessage 列表，tool_calls/
             tool_results/errors/generated_charts 分别累加本批次台账
    """
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
    for tc in tool_calls:
        name = tc.get("name", "")
        # args 可能为 None，统一兜底成空字典，避免下方解包/传参报错
        args = tc.get("args", {}) or {}
        tool = tools_by_name.get(name)

        if tool is None:
            # 模型幻觉出不存在的工具名：不抛异常，回文本让其改选其他工具
            content = f"未知工具: {name}"
            status = "error"
        else:
            try:
                # 真正执行工具；返回值统一转成字符串供 LLM 阅读
                result = tool.invoke(args)
                content = result if isinstance(result, str) else str(result)
                status = "success"
            except Exception as e:  # 工具异常 → 记录并回传给 LLM，让其调整
                # 把异常类型与信息拼成可读文本，作为该工具的返回内容
                content = f"工具执行异常：{type(e).__name__}: {e}"
                status = "error"
                new_errors.append(f"{name}: {e}")

        # ToolMessage 必须带 tool_call_id，LangChain 据此与请求一一配对
        messages.append(
            ToolMessage(content=content, tool_call_id=tc.get("id", ""), name=name)
        )
        # 台账：调用记录（含入参）与结果记录（含返回内容/状态）分开存放
        new_tool_calls.append({"name": name, "args": args, "status": status})
        new_tool_results.append({"name": name, "result": content, "status": status})

        # 记录实际成功生成的图表路径
        # 仅当图表工具成功且输出中能解析出路径时，才加入图表白名单
        if name == "generate_chart" and status == "success":
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
