"""Tool Calling 节点：LLM 决策（agent_node）+ 工具执行（tools_node）。"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from src.agent.state import AgentState
from src.llm import get_llm
from src.tools import get_all_tools, get_tools_by_name


def _extract_chart_path(result: str) -> str | None:
    """从 generate_chart 工具的输出中提取图表文件路径。"""
    marker = "图表已生成:"
    if marker in result:
        return result.split(marker, 1)[1].strip()
    return None


def agent_node(state: AgentState) -> dict:
    """LLM 观察当前上下文，决定调用工具或输出结论。"""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(get_all_tools())
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }


def tools_node(state: AgentState) -> dict:
    """执行上一条 AIMessage 中的工具调用，记录结果与错误（不中断链路）。"""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    messages: list[ToolMessage] = []
    new_tool_calls: list[dict] = []
    new_tool_results: list[dict] = []
    new_errors: list[str] = []
    new_generated_charts: list[str] = []

    tools_by_name = get_tools_by_name()
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        tool = tools_by_name.get(name)

        if tool is None:
            content = f"未知工具: {name}"
            status = "error"
        else:
            try:
                result = tool.invoke(args)
                content = result if isinstance(result, str) else str(result)
                status = "success"
            except Exception as e:  # 工具异常 → 记录并回传给 LLM，让其调整
                content = f"工具执行异常：{type(e).__name__}: {e}"
                status = "error"
                new_errors.append(f"{name}: {e}")

        messages.append(
            ToolMessage(content=content, tool_call_id=tc.get("id", ""), name=name)
        )
        new_tool_calls.append({"name": name, "args": args, "status": status})
        new_tool_results.append({"name": name, "result": content, "status": status})

        # 记录实际成功生成的图表路径
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
