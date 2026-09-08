"""端到端集成测试：用 FakeChatModel 驱动整条 LangGraph 链路。

说明：这里用 FakeChatModel 替代真实 LLM，目的是验证图结构、工具循环、
条件边、报告落盘等“机械流程”是否正确串起来，不依赖外部 API Key。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state
from tests.conftest import SALES_CSV


class FakeChatModel:
    """按 system prompt 内容返回脚本化响应，模拟真实 LLM 的行为。"""

    def __init__(self) -> None:
        self._tools = []
        self.agent_calls = 0

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        first = messages[0].content if messages else ""
        if "自主的数据分析智能体" in first:
            return self._agent_response()
        if "规划者" in first:
            return AIMessage(content='[{"step":1,"goal":"统计销售额","tool":"calculate_statistics","note":"计算 sales 列均值"}]')
        if "洞察专家" in first:
            return AIMessage(content="洞察1：华东地区销售额下降明显。\n洞察2：sales 与 profit 高度相关。")
        if "报告撰写" in first:
            return AIMessage(content="# 分析报告\n\n## 数据集概览\n- 2160 行\n\n## 核心发现\n销售额下降。")
        if "理解" in first:
            return AIMessage(content="目标：分析销售额下降原因。指标：sales；维度：region/product/time。")
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
        if self.agent_calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "detect_outliers",
                        "args": {"path": SALES_CSV, "column": "sales", "method": "zscore"},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="核心发现：2025 年销售额下降，华东跌幅最大，sales 存在异常值。")


def _patch_all_llm(monkeypatch, fake: FakeChatModel) -> None:
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_full_graph_end_to_end(monkeypatch) -> None:
    fake = FakeChatModel()
    _patch_all_llm(monkeypatch, fake)

    graph = build_graph()
    initial = make_initial_state(SALES_CSV, "分析销售额下降原因")

    result = graph.invoke(initial)

    # 报告成功生成
    assert result["status"] == "done"
    assert result["final_report"].startswith("#")
    assert result["report_path"].endswith(".md")

    # 工具确实被调用并记录了结果
    assert len(result["tool_calls"]) == 2
    assert len(result["tool_results"]) == 2
    names = [tc["name"] for tc in result["tool_calls"]]
    assert "calculate_statistics" in names
    assert "detect_outliers" in names

    # 报告文件落盘
    assert Path(result["report_path"]).exists()

    # 清理测试产物
    Path(result["report_path"]).unlink(missing_ok=True)
