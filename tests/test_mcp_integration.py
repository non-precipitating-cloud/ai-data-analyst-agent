"""MCP 集成测试（src.mcp × src.agent）。

验证两件事：
- 普通本地工具与适配后的 MCP 工具能在同一个工具清单中共存；
- 用 FakeChatModel 驱动完整 LangGraph，Agent 能自主选择并真实执行 mcp__read_dataset。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state
from src.mcp.client import (
    McpClient,
    build_mcp_tools,
    get_server_params,
    mcp_stdio_runtime_check,
)
from tests.conftest import SALES_CSV


@pytest.fixture(scope="module")
def mcp_tools():
    """模块级夹具：建立 MCP 连接并一次性适配出全部 MCP 工具，结束后关闭连接。"""
    # 环境不满足 stdio 运行条件时（如 Windows 缺 pywin32）明确 skip，
    # 而不是让集成用例因含义不明的 "Connection closed" 失败
    reason = mcp_stdio_runtime_check()
    if reason:
        pytest.skip(reason)
    client = McpClient(get_server_params())
    try:
        yield build_mcp_tools(client)
    finally:
        client.close()


def test_normal_and_mcp_tools_coexist(monkeypatch, mcp_tools) -> None:
    """本地工具与 mcp__ 前缀工具应同时存在，且总数为两者之和、互不覆盖。"""
    # 让 get_mcp_tools 返回本测试建立的 MCP 工具元组
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
    # 总数严格等于本地工具数 + MCP 工具数
    assert len(tools) == len(ALL_TOOLS) + len(mcp_tools)


class FakeChatModel:
    """脚本化 LLM：Agent 自主调用一个 MCP 工具（mcp__read_dataset）。"""

    def __init__(self) -> None:
        self._tools = []
        # 记录 Agent 节点调用次数，用于只发起一次 MCP 工具调用
        self.agent_calls = 0

    def bind_tools(self, tools):
        """模拟 bind_tools 并返回自身。"""
        self._tools = tools
        return self

    def invoke(self, messages):
        """按 system 消息关键词为各节点返回脚本响应。"""
        first = messages[0].content if messages else ""
        if "技能选择器" in first:
            # 不选择任何技能
            return AIMessage(content="[]")
        if "自主的数据分析智能体" in first:
            return self._agent_response()
        if "规划者" in first:
            # 规划里直接指定 MCP 工具
            return AIMessage(content='[{"step":1,"goal":"读取数据","tool":"mcp__read_dataset","note":"通过 MCP 读取"}]')
        if "洞察专家" in first:
            return AIMessage(content="洞察1：通过 MCP 读取到 2160 行数据。")
        if "报告撰写" in first:
            return AIMessage(content="# 报告\n\n## 图表说明\n本次分析未生成图表。\n\n## 核心发现\n已通过 MCP 读取数据。")
        if "理解" in first:
            return AIMessage(content="目标：读取销售数据。")
        return AIMessage(content="ok")

    def _agent_response(self) -> AIMessage:
        """第一次调用请求 mcp__read_dataset，之后返回纯文本结论结束循环。"""
        self.agent_calls += 1
        if self.agent_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "mcp__read_dataset",
                        # MCP 适配层要求把真实参数包在 arguments 内
                        "args": {"arguments": {"path": SALES_CSV}},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="结论：已通过 MCP 读取数据。")


def _patch_all_llm(monkeypatch, fake: FakeChatModel) -> None:
    """把五个节点与技能选择器的 get_llm 全部替换为假模型。"""
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_full_agent_with_mcp_tool(monkeypatch, mcp_tools) -> None:
    """完整 Agent 链路应真实调用 mcp__read_dataset 并成功拿到数据，最终生成报告。"""
    # 注入本测试的 MCP 工具集
    monkeypatch.setattr("src.mcp.get_mcp_tools", lambda: tuple(mcp_tools))
    fake = FakeChatModel()
    _patch_all_llm(monkeypatch, fake)

    graph = build_graph()
    initial = make_initial_state(SALES_CSV, "读取销售数据")

    result = graph.invoke(initial)

    # MCP 工具被 Agent 调用并真实执行
    names = [tc["name"] for tc in result["tool_calls"]]
    assert "mcp__read_dataset" in names

    # 对应工具结果必须标记成功且包含真实行数
    mcp_results = [tr for tr in result["tool_results"] if tr["name"] == "mcp__read_dataset"]
    assert mcp_results
    assert mcp_results[0]["status"] == "success"
    assert "num_rows" in mcp_results[0]["result"]

    assert result["status"] == "done"
    # 清理落盘报告
    Path(result["report_path"]).unlink(missing_ok=True)
