"""LangGraph 工作流构建与条件路由测试（src.agent.graph）。

覆盖：
- build_graph：图可成功构建且包含全部关键节点；
- make_initial_state：初始状态字段完整，消息列表恰为 system + human 两条；
- route_after_agent：依据 Agent 是否请求工具调用、以及是否达到最大步数来决定下一跳。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state, route_after_agent
from tests.conftest import SALES_CSV


def test_build_graph() -> None:
    """构建出的图应非空，并包含理解/画像/规划/Agent/工具/洞察/报告等关键节点。"""
    g = build_graph()
    assert g is not None
    node_names = set(g.get_graph().nodes.keys())
    assert {"task_understanding", "profiler", "planner", "agent", "tools", "insight", "report"} <= node_names


def test_make_initial_state() -> None:
    """初始状态应携带用户需求、数据集路径、正数步数上限，且初始步数为 0、消息为两条。"""
    s = make_initial_state(SALES_CSV, "分析销售额下降原因")
    assert s["user_request"] == "分析销售额下降原因"
    assert s["dataset_path"] == SALES_CSV
    assert s["max_steps"] > 0
    assert s["step_count"] == 0
    assert len(s["messages"]) == 2  # system + human


def _msg(tool_calls=None):
    """测试辅助：构造一条 AI 消息，可按需携带工具调用请求。"""
    if tool_calls is not None:
        return AIMessage(content="", tool_calls=tool_calls)
    return AIMessage(content="分析完成")


def test_route_with_tool_calls() -> None:
    """Agent 请求调用工具且未达步数上限时，应路由到 tools 节点继续执行。"""
    tc = [{"name": "calculate_statistics", "args": {}, "id": "1"}]
    s = {"messages": [_msg(tc)], "step_count": 1, "max_steps": 15}
    assert route_after_agent(s) == "tools"


def test_route_without_tool_calls() -> None:
    """Agent 不再请求工具时，应直接路由到 insight 节点产出洞察。"""
    s = {"messages": [_msg()], "step_count": 1, "max_steps": 15}
    assert route_after_agent(s) == "insight"


def test_route_hits_max_steps() -> None:
    """即使仍有工具调用，达到最大步数也强制进入 insight，防止工具循环无限执行。"""
    tc = [{"name": "calculate_statistics", "args": {}, "id": "1"}]
    s = {"messages": [_msg(tc)], "step_count": 15, "max_steps": 15}
    assert route_after_agent(s) == "insight"
