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
    """按 system prompt 内容返回脚本化响应，模拟真实 LLM 在各节点上的行为。"""

    def __init__(self) -> None:
        self._tools = []
        # 统计 Agent 节点被调用的次数，用于按次序返回不同的脚本响应
        self.agent_calls = 0

    def bind_tools(self, tools):
        """模拟 LangChain 的 bind_tools：记录绑定工具并返回自身。"""
        self._tools = tools
        return self

    def invoke(self, messages):
        """根据首条 system 消息中包含的节点关键词，分发到对应脚本响应。"""
        first = messages[0].content if messages else ""
        if "自主的数据分析智能体" in first:
            # Agent 节点：走工具调用编排
            return self._agent_response()
        if "规划者" in first:
            # 规划节点：返回只含一个统计步骤的 JSON 计划
            return AIMessage(content='[{"step":1,"goal":"统计销售额","tool":"calculate_statistics","note":"计算 sales 列均值"}]')
        if "洞察专家" in first:
            # 洞察节点：返回两条固定洞察
            return AIMessage(content="洞察1：华东地区销售额下降明显。\n洞察2：sales 与 profit 高度相关。")
        if "报告撰写" in first:
            # 报告节点：返回固定 Markdown 报告
            return AIMessage(content="# 分析报告\n\n## 数据集概览\n- 2160 行\n\n## 核心发现\n销售额下降。")
        if "理解" in first:
            # 任务理解节点：返回目标、指标与维度
            return AIMessage(content="目标：分析销售额下降原因。指标：sales；维度：region/product/time。")
        return AIMessage(content="ok")

    def _agent_response(self) -> AIMessage:
        """脚本化 Agent：前两次各请求一个工具，第三次给出纯文本结论结束循环。"""
        self.agent_calls += 1
        if self.agent_calls == 1:
            # 第一次：请求统计 sales 列
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
            # 第二次：请求 zscore 异常检测
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
        # 第三次：不再调用工具，循环结束进入洞察
        return AIMessage(content="核心发现：2025 年销售额下降，华东跌幅最大，sales 存在异常值。")


def _patch_all_llm(monkeypatch, fake: FakeChatModel) -> None:
    """把五个节点与技能选择器的 get_llm 全部替换为同一个 FakeChatModel。"""
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_full_graph_end_to_end(monkeypatch) -> None:
    """全链路应跑通：两次真实工具调用 → 洞察 → 报告生成并落盘，最终状态为 done。"""
    fake = FakeChatModel()
    # 全节点替换为假 LLM，保证测试离线、确定性
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
