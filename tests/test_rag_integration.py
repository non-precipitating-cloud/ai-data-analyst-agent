"""RAG 集成测试：FakeChatModel 驱动全图，Agent 自主调用 retrieve_knowledge，结果影响报告。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, make_initial_state
from tests.conftest import SALES_CSV


class FakeChatModel:
    """脚本化 LLM：让 Agent 先检索知识库，再做异常检测。"""

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
            return AIMessage(content='[{"step":1,"goal":"检测异常","tool":"retrieve_knowledge","note":"检索异常检测方法"}]')
        if "洞察专家" in first:
            return AIMessage(content="洞察1：结合知识库，采用 z-score（阈值3）检测到异常值。")
        if "报告撰写" in first:
            return AIMessage(content="# 报告\n\n## 图表说明\n本次分析未生成图表。\n\n## 核心发现\n基于知识库方法论完成异常检测。")
        if "理解" in first:
            return AIMessage(content="目标：检测数据中的异常值。")
        return AIMessage(content="ok")

    def _agent_response(self) -> AIMessage:
        self.agent_calls += 1
        if self.agent_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_knowledge",
                        "args": {"query": "异常检测方法 阈值", "top_k": 3},
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
        return AIMessage(content="结论：检索到异常检测方法论，采用 z-score 检测出异常值。")


def _patch_all_llm(monkeypatch, fake: FakeChatModel) -> None:
    monkeypatch.setattr("src.agent.nodes.task_understanding.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.planner.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: fake)
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    monkeypatch.setattr("src.skills.selector.get_llm", lambda: fake)


def test_rag_tool_used_in_agent_loop(monkeypatch) -> None:
    # 用内存检索器替代 get_retriever，避免依赖外部 PG（快速、确定性）
    from src.rag.embeddings import HashingEmbedder
    from src.rag.retriever import KnowledgeRetriever
    from src.rag.vector_store import InMemoryVectorStore

    embedder = HashingEmbedder(dim=128)
    retriever = KnowledgeRetriever(embedder, InMemoryVectorStore(embedder))
    monkeypatch.setattr("src.tools.rag_tool.get_retriever", lambda: retriever)

    fake = FakeChatModel()
    _patch_all_llm(monkeypatch, fake)

    graph = build_graph()
    initial = make_initial_state(SALES_CSV, "检测数据中的异常值")

    result = graph.invoke(initial)

    # retrieve_knowledge 被 Agent 自主调用并记录
    names = [tc["name"] for tc in result["tool_calls"]]
    assert "retrieve_knowledge" in names

    # RAG 检索结果真实返回，且包含知识库内容（真正参与了分析）
    rag_results = [tr for tr in result["tool_results"] if tr["name"] == "retrieve_knowledge"]
    assert rag_results
    assert rag_results[0]["status"] == "success"
    assert "异常检测" in rag_results[0]["result"]

    # 报告成功生成
    assert result["status"] == "done"
    assert result["final_report"].startswith("#")

    # 清理
    Path(result["report_path"]).unlink(missing_ok=True)
