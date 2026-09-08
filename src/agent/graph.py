"""LangGraph 组装：节点 + 条件边 + 循环。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agent.prompts import AGENT_SYSTEM
from src.agent.nodes import (
    agent_node,
    insight_node,
    planner_node,
    profiler_node,
    report_node,
    skill_selection_node,
    task_understanding_node,
    tools_node,
)
from src.agent.state import AgentState
from src.config.settings import get_settings


def make_initial_state(dataset_path: str, user_request: str) -> AgentState:
    """构造初始状态。"""
    settings = get_settings()
    return {
        "messages": [
            SystemMessage(content=AGENT_SYSTEM),
            HumanMessage(content=f"数据文件：{dataset_path}\n分析需求：{user_request}"),
        ],
        "user_request": user_request,
        "dataset_path": dataset_path,
        "dataset_metadata": {},
        "analysis_plan": [],
        "selected_skills": [],
        "skills_context": "",
        "step_count": 0,
        "max_steps": settings.max_steps,
        "status": "running",
        "observations": [],
        "insights": [],
        "errors": [],
        "tool_calls": [],
        "tool_results": [],
        "generated_charts": [],
    }


def route_after_agent(state: AgentState) -> str:
    """决定 tool_calling 之后的走向：继续调工具 / 进入洞察。"""
    last = state.get("messages", [None])[-1] if state.get("messages") else None
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)

    if has_tool_calls and step_count < max_steps:
        return "tools"
    return "insight"


def build_graph():
    """构建并编译 Agent 图。"""
    g = StateGraph(AgentState)

    g.add_node("task_understanding", task_understanding_node)
    g.add_node("skill_selection", skill_selection_node)
    g.add_node("profiler", profiler_node)
    g.add_node("planner", planner_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("insight", insight_node)
    g.add_node("report", report_node)

    g.add_edge(START, "task_understanding")
    g.add_edge("task_understanding", "skill_selection")
    g.add_edge("skill_selection", "profiler")
    g.add_edge("profiler", "planner")
    g.add_edge("planner", "agent")

    # 循环：agent ⇄ tools，直到足够 / 达到最大步数
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "insight": "insight"},
    )
    g.add_edge("tools", "agent")

    g.add_edge("insight", "report")
    g.add_edge("report", END)

    return g.compile()


# 模块级单例
graph = build_graph()
