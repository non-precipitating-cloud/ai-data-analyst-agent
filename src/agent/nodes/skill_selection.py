"""Skill 选择节点：根据任务选择相关 Skill 并注入上下文。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.skills import format_skills, get_all_skills, select_skills


def skill_selection_node(state: AgentState) -> dict:
    """选择相关 Skills，并把正文注入 messages 与 skills_context。"""
    skills = get_all_skills()
    slugs = select_skills(
        state["user_request"], state.get("understanding", ""), skills
    )
    selected = [s for s in skills if s.slug in slugs]
    context = format_skills(selected)
    label = ", ".join(slugs) if slugs else "（无）"

    return {
        "selected_skills": slugs,
        "skills_context": context,
        "messages": [AIMessage(content=f"[已选择 Skills]\n{label}\n\n{context}")],
        "observations": [f"[Skill 选择]\n已选择: {label}"],
    }
