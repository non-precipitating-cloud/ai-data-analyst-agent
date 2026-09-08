"""MCP 集成测试：普通工具与 MCP 工具共存 + 完整 Agent 调用 MCP 工具。"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state
from src.mcp.client import McpClient, build_mcp_tools, get_server_params
from tests.conftest import SALES_CSV


@pytest.fixture(scope="module")
def mcp_tools():
    client = McpClient(get_server_params())
    try:
        yield build_mcp_tools(client)
    finally:
        client.close()


def test_normal_and_mcp_tools_coexist(monkeypatch, mcp_tools) -> None:
    monkeypatch.setattr("src.mcp.get_mcp_tools", lambda: tuple(mcp_tools))
    from src.tools import ALL_TOOLS, get_all_tools

    tools = get_all_tools()
    names = {t.name for t in tools}

    # 普通工具仍在
    assert "execute_python" in names
    assert "read_dataset" in names
    # MCP 工具以 mcp__ 前缀共存
    assert "mcp__read_dataset" in names
    assert "mcp__detect_outliers" in names
    assert len(tools) == len(ALL_TOOLS) + len(mcp_tools)


class FakeChatModel:
    """脚本化 LLM：Agent 自主调用一个 MCP 工具（mcp__read_dataset）。"""

    def __init__(self) -> None:
        self._tools = []
        self.agent_calls = 0

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        first = messages[0].content if messages else ""
        if "技能选择器" in first:
            return AIMessage(content="[]")
        if "自主的数据分析智能体" in first:
            return self._agent_response()
        if "规划者" in first:
            return AIMessage(content='[{"step":1,"goal":"读取数据","tool":"mcp__read_dataset","note":"通过 MCP 读取"}]')
        if "洞察专家" in first:
            return AIMessage(content="洞察1：通过 MCP 读取到 2160 行数据。")
        if "报告撰写" in first:
            return AIMessage(content="# 报告\n\n## 图表说明\n本次分析未生成图表。\n\n## 核心发现\n已通过 MCP 读取数据。")
        if "理解" in first:
            return AIMessage(content="目标：读取销售数据。")
        return AIMessage(content="ok")

    def _agent_response(self) -> AIMessage:
        self.agent_calls += 1
        if self.agent_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "mcp__read_dataset",
                        "args": {"arguments": {"path": SALES_CSV}},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="结论：已通过 MCP 读取数据。")


def _patch_all_llm(monkeypatch, fake: FakeChatModel) -> None:
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_full_agent_with_mcp_tool(monkeypatch, mcp_tools) -> None:
    monkeypatch.setattr("src.mcp.get_mcp_tools", lambda: tuple(mcp_tools))
    fake = FakeChatModel()
    _patch_all_llm(monkeypatch, fake)

    graph = build_graph()
    initial = make_initial_state(SALES_CSV, "读取销售数据")

    result = graph.invoke(initial)

    # MCP 工具被 Agent 调用并真实执行
    names = [tc["name"] for tc in result["tool_calls"]]
    assert "mcp__read_dataset" in names

    mcp_results = [tr for tr in result["tool_results"] if tr["name"] == "mcp__read_dataset"]
    assert mcp_results
    assert mcp_results[0]["status"] == "success"
    assert "num_rows" in mcp_results[0]["result"]

    assert result["status"] == "done"
    Path(result["report_path"]).unlink(missing_ok=True)
