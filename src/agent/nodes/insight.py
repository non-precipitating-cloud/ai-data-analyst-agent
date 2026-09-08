"""洞察节点：基于工具结果提炼关键洞察。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.prompts import INSIGHT_SYSTEM
from src.agent.state import AgentState
from src.agent.utils import truncate
from src.llm import get_llm


def insight_node(state: AgentState) -> dict:
    """汇总工具执行结果，提炼洞察。"""
    llm = get_llm()

    observations = "\n".join(state.get("observations", []))
    tool_results = state.get("tool_results", [])
    results_text = "\n\n".join(
        f"[{tr.get('name')}]\n{truncate(str(tr.get('result', '')), 2000)}"
        for tr in tool_results
    )

    prompt = HumanMessage(
        content=(
            f"分析需求：{state['user_request']}\n\n"
            f"观察记录：\n{truncate(observations)}\n\n"
            f"工具结果：\n{results_text}"
        )
    )
    resp = llm.invoke([SystemMessage(content=INSIGHT_SYSTEM), prompt])
    insights = resp.content

    return {
        "insights": [insights],
        "messages": [AIMessage(content=f"[洞察]\n{insights}")],
    }
