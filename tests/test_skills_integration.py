"""Skills 集成测试（src.skills × src.agent）。

用 FakeChatModel 驱动完整 LangGraph，验证：
技能选择器选中 sales-analysis → 技能上下文注入状态 → Agent 工具循环照常执行
→ 最终报告正常生成，整条链路串通。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state
from tests.conftest import SALES_CSV


class FakeChatModel:
    """脚本化 LLM：先选 Skill，再做一次工具调用，最后生成报告。"""

    def __init__(self) -> None:
        self._tools = []
        # 控制 Agent 节点只发起一次工具调用
        self.agent_calls = 0

    def bind_tools(self, tools):
        """模拟 bind_tools 并返回自身。"""
        self._tools = tools
        return self

    def invoke(self, messages):
        """按 system 消息关键词返回各节点的脚本响应。"""
        first = messages[0].content if messages else ""
        if "技能选择器" in first:
            # 选择器节点：只选销售分析技能
            return AIMessage(content='["sales-analysis"]')
        if "自主的数据分析智能体" in first:
            return self._agent_response()
        if "规划者" in first:
            return AIMessage(content='[{"step":1,"goal":"地区销售对比","tool":"execute_python","note":"按地区聚合销售"}]')
        if "洞察专家" in first:
            return AIMessage(content="洞察1：华东地区销售下滑最严重。")
        if "报告撰写" in first:
            return AIMessage(content="# 报告\n\n## 图表说明\n本次分析未生成图表。\n\n## 核心发现\n遵循销售分析 Skill 方法论完成地区归因。")
        if "理解" in first:
            return AIMessage(content="目标：分析销售额下降原因。")
        return AIMessage(content="ok")

    def _agent_response(self) -> AIMessage:
        """第一次调用请求统计 sales，之后返回纯文本结论结束循环。"""
        self.agent_calls += 1
        if self.agent_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_statistics",
                        "args": {"path": SALES_CSV, "column": "sales"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="结论：销售额下降，需按地区归因。")


def _patch_all_llm(monkeypatch, fake: FakeChatModel) -> None:
    """把五个节点与技能选择器的 get_llm 全部替换为假模型。"""
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_skill_selection_flows_through_graph(monkeypatch) -> None:
    """技能选择结果应贯穿全图：被选中、上下文注入、工具执行、报告生成。"""
    fake = FakeChatModel()
    _patch_all_llm(monkeypatch, fake)

    graph = build_graph()
    initial = make_initial_state(SALES_CSV, "分析销售额下降原因")

    result = graph.invoke(initial)

    # Skill 被选择并注入
    assert result["selected_skills"] == ["sales-analysis"]
    assert "销售分析" in result["skills_context"]

    # 工具循环仍正常执行
    names = [tc["name"] for tc in result["tool_calls"]]
    assert "calculate_statistics" in names

    # 报告成功生成
    assert result["status"] == "done"
    assert result["final_report"].startswith("#")

    # 清理落盘报告
    Path(result["report_path"]).unlink(missing_ok=True)
