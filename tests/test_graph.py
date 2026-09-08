"""LangGraph 图与循环路由测试。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state, route_after_agent
from tests.conftest import SALES_CSV


def test_build_graph() -> None:
    g = build_graph()
    assert g is not None
    node_names = set(g.get_graph().nodes.keys())
    assert {"task_understanding", "profiler", "planner", "agent", "tools", "insight", "report"} <= node_names


def test_make_initial_state() -> None:
    s = make_initial_state(SALES_CSV, "分析销售额下降原因")
    assert s["user_request"] == "分析销售额下降原因"
    assert s["dataset_path"] == SALES_CSV
    assert s["max_steps"] > 0
    assert s["step_count"] == 0
    assert len(s["messages"]) == 2  # system + human


def _msg(tool_calls=None):
    if tool_calls is not None:
        return AIMessage(content="", tool_calls=tool_calls)
    return AIMessage(content="分析完成")


def test_route_with_tool_calls() -> None:
    tc = [{"name": "calculate_statistics", "args": {}, "id": "1"}]
    s = {"messages": [_msg(tc)], "step_count": 1, "max_steps": 15}
    assert route_after_agent(s) == "tools"


def test_route_without_tool_calls() -> None:
    s = {"messages": [_msg()], "step_count": 1, "max_steps": 15}
    assert route_after_agent(s) == "insight"


def test_route_hits_max_steps() -> None:
    tc = [{"name": "calculate_statistics", "args": {}, "id": "1"}]
    s = {"messages": [_msg(tc)], "step_count": 15, "max_steps": 15}
    assert route_after_agent(s) == "insight"
