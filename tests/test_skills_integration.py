"""Skills 集成测试：FakeChatModel 驱动全图，Skill 选择 → 注入上下文 → 影响计划与报告。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state
from tests.conftest import SALES_CSV


class FakeChatModel:
    """脚本化 LLM：先选 Skill，再做一次工具调用，最后生成报告。"""

    def __init__(self) -> None:
        self._tools = []
        self.agent_calls = 0

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        first = messages[0].content if messages else ""
        if "技能选择器" in first:
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
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_skill_selection_flows_through_graph(monkeypatch) -> None:
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

    Path(result["report_path"]).unlink(missing_ok=True)
