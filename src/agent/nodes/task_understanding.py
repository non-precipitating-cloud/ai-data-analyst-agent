"""任务理解节点：解析用户需求。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.prompts import TASK_UNDERSTANDING_SYSTEM
from src.agent.state import AgentState
from src.llm import get_llm


def task_understanding_node(state: AgentState) -> dict:
    """用 LLM 解析用户需求，输出任务理解。"""
    llm = get_llm()
    request = state["user_request"]
    columns = state.get("dataset_metadata", {}).get("columns", [])

    prompt = HumanMessage(content=f"用户分析需求：{request}\n\n数据集字段：{columns}")
    resp = llm.invoke([SystemMessage(content=TASK_UNDERSTANDING_SYSTEM), prompt])
    understanding = resp.content

    return {
        "understanding": understanding,
        "observations": [f"[任务理解]\n{understanding}"],
        "messages": [AIMessage(content=f"[任务理解]\n{understanding}")],
    }
