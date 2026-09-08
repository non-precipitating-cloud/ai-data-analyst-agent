"""规划节点：基于任务理解与数据画像生成分析计划。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.prompts import PLANNER_SYSTEM
from src.agent.state import AgentState
from src.agent.utils import parse_json, truncate
from src.llm import get_llm


def planner_node(state: AgentState) -> dict:
    """生成分析计划（JSON 数组）。"""
    llm = get_llm()
    meta = state.get("dataset_metadata", {})
    columns = meta.get("columns", [])
    numeric = meta.get("numeric_columns", [])

    prompt = HumanMessage(
        content=(
            f"用户需求：{state['user_request']}\n"
            f"任务理解：{state.get('understanding', '')}\n"
            f"已选择的 Skills 方法论：\n{truncate(state.get('skills_context', ''), 2500)}\n\n"
            f"数据集字段：{columns}\n"
            f"数值字段：{numeric}"
        )
    )
    resp = llm.invoke([SystemMessage(content=PLANNER_SYSTEM), prompt])
    plan = parse_json(resp.content)
    if not isinstance(plan, list):
        plan = []

    plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
    return {
        "analysis_plan": plan,
        "observations": [f"[分析计划]\n{plan_text}"],
        "messages": [AIMessage(content=f"[分析计划]\n{plan_text}")],
    }
