"""基础设施降级与 Embedding Provider 测试。

对应「RAG」「PostgreSQL」「Redis」三节要求：
- 数据库 / Redis 不可用时只告警不中断，Agent 核心分析照常完成；
- Embedding 实现要**明确区分**真实语义向量与离线词法回退，
  不能把「离线能跑」当成「检索可用」；
- 测试不允许因为缺 API Key 被迫联网。
"""

from __future__ import annotations

import pytest

from src.agent.memory import ConversationMemory
from src.cache import Cache
from src.config.settings import get_settings
from src.db.service import PersistenceService
from src.rag.embeddings import (
    DeterministicTestEmbedder,
    LexicalHashEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from src.tools.rag_tool import retrieve_knowledge
from tests.conftest import SALES_CSV

_DATASET_INFO = {"row_count": 10, "column_count": 3, "columns": ["a"]}


# ==========================================================================
# PostgreSQL 不可用
# ==========================================================================
def test_persistence_survives_database_outage(monkeypatch) -> None:
    """数据库不可用时，所有持久化方法都应静默跳过，Agent 流程不受影响。"""
    from src.db import database

    # 让可用性探测直接判定不可用（等价于 PG 没启动）
    monkeypatch.setattr(database, "_db_available", False)
    svc = PersistenceService()

    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)
    # 会话 ID 仍要生成（Redis 或本地都可承载会话）
    assert ids["session_id"]
    assert ids["task_id"] is None
    assert ids["run_id"] is None

    # 这些调用都不得抛异常
    svc.update_status("agent")
    svc.record_plan([{"step": 1}])
    svc.record_tool_call("read_dataset", "local", {}, "{}", "success")
    svc.record_llm_calls([{"node": "agent", "success": True, "total_tokens": 5}])
    svc.record_result("insight", {"insights": []})
    svc.record_report("p.md", "content")
    svc.complete("completed")
    database.reset_engine()


def test_db_unavailable_does_not_block_agent_run(monkeypatch, tmp_path) -> None:
    """端到端：数据库不可用 + LLM 不可用也不能让 run_agent 崩溃。"""
    from src.agent.graph import build_graph, make_initial_state

    class _FailingLLM:
        model_name = "down"

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
            return self

        def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN201
            raise RuntimeError("model down")

    import src.agent.nodes.insight as insight_mod
    import src.agent.nodes.planner as planner_mod
    import src.agent.nodes.report as report_mod
    import src.agent.nodes.task_understanding as tu_mod
    import src.agent.nodes.tool_calling as tc_mod
    import src.skills.selector as selector_mod

    mods = [tu_mod, selector_mod, planner_mod, tc_mod, insight_mod, report_mod]
    originals = [m.get_llm for m in mods]
    for m in mods:
        m.get_llm = lambda: _FailingLLM()  # type: ignore[assignment]
    monkeypatch.setattr(get_settings(), "agent_llm_retries", 0)
    try:
        from pathlib import Path

        out = build_graph().invoke(make_initial_state(SALES_CSV, "分析"))
        # 模型全线不可用：仍应产出报告（降级模板），status 为 done
        assert out["status"] == "done"
        assert out["degraded"]
        Path(out["report_path"]).unlink(missing_ok=True)
    finally:
        for m, o in zip(mods, originals):
            m.get_llm = o  # type: ignore[assignment]


# ==========================================================================
# Redis 不可用
# ==========================================================================
def test_cache_degrades_when_redis_unavailable(monkeypatch) -> None:
    """Redis 不可用时缓存读写返回 False/None，不抛异常。"""
    from src.cache import redis_client

    redis_client.reset_redis()
    monkeypatch.setattr(redis_client, "_redis_available", False)

    cache = Cache()
    assert cache.set_cache("k", {"a": 1}) is False
    assert cache.get_cache("k") is None
    assert cache.delete_cache("k") is False
    redis_client.reset_redis()


def test_memory_and_session_degrade_without_redis(monkeypatch) -> None:
    """Redis 不可用时会话与记忆仍可用（退化为进程内），不丢功能。"""
    from src.cache import redis_client

    redis_client.reset_redis()
    monkeypatch.setattr(redis_client, "_redis_available", False)

    memory = ConversationMemory(session_id="s")
    memory.remember("问题", SALES_CSV, ["结论"], "r.md")
    assert "结论" in memory.build_context("为什么？", SALES_CSV)
    redis_client.reset_redis()


# ==========================================================================
# Embedding Provider 分层
# ==========================================================================
def test_hash_provider_is_explicitly_non_semantic() -> None:
    """离线词法向量必须自报「非语义」，避免被误当成语义检索。"""
    embedder = LexicalHashEmbedder(dim=64)
    assert embedder.is_semantic is False
    assert embedder.name == "lexical-hash"
    # 确定性：同文本同向量
    assert embedder.embed_query("销售额下降") == embedder.embed_query("销售额下降")
    assert len(embedder.embed_query("x")) == 64


def test_test_embedder_is_deterministic_and_distinct() -> None:
    """测试替身只保证确定性与可区分性，且明确标注非语义。"""
    embedder = DeterministicTestEmbedder(dim=8)
    assert embedder.is_semantic is False
    assert embedder.dimension == 8
    a1, a2 = embedder.embed_query("a"), embedder.embed_query("a")
    b = embedder.embed_query("b")
    assert a1 == a2
    assert a1 != b


def test_openai_embedder_declares_semantic() -> None:
    """真实 Embedding 实现必须自报语义（供上层如实展示检索质量）。"""
    assert OpenAICompatibleEmbedder.is_semantic is True


def test_build_embedder_hash_mode_never_requires_network(monkeypatch) -> None:
    """显式选择 hash 模式时完全离线，测试不会因为缺 Key 被迫联网。"""
    monkeypatch.setattr(get_settings(), "embedding_provider", "hash")
    monkeypatch.setattr(get_settings(), "embedding_api_key", "")
    assert isinstance(build_embedder(), LexicalHashEmbedder)


def test_build_embedder_test_mode(monkeypatch) -> None:
    """显式选择 test 模式时使用测试替身。"""
    monkeypatch.setattr(get_settings(), "embedding_provider", "test")
    assert isinstance(build_embedder(), DeterministicTestEmbedder)


def test_build_embedder_openai_mode_requires_key(monkeypatch) -> None:
    """强制语义模式但缺 Key 时应快速失败，而不是悄悄退回词法匹配。"""
    monkeypatch.setattr(get_settings(), "embedding_provider", "openai")
    monkeypatch.setattr(get_settings(), "embedding_api_key", "")
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        build_embedder()


def test_build_embedder_auto_falls_back_without_key(monkeypatch) -> None:
    """auto 模式在无 Key 时回退离线词法，保证离线可跑。"""
    monkeypatch.setattr(get_settings(), "embedding_provider", "auto")
    monkeypatch.setattr(get_settings(), "embedding_api_key", "")
    embedder = build_embedder()
    assert isinstance(embedder, LexicalHashEmbedder)
    assert embedder.is_semantic is False


def test_rag_tool_discloses_non_semantic_retrieval(monkeypatch) -> None:
    """非语义检索时，工具返回里必须如实说明，避免模型把字面重合当语义相关。"""
    monkeypatch.setattr(get_settings(), "embedding_provider", "hash")
    monkeypatch.setattr(get_settings(), "embedding_api_key", "")

    # 重置检索器单例，确保用新的 embedder 配置
    from src.rag import get_retriever
    from src.rag.embeddings import get_embedder
    from src.rag.vector_store import get_vector_store

    get_embedder.cache_clear()
    get_retriever.cache_clear()
    get_vector_store.cache_clear()

    # 用与知识库高度字面重合的查询，保证能命中内容
    out = retrieve_knowledge.invoke({"query": "异常检测 阈值 方法", "top_k": 3})
    assert "检索方式" in out
    assert "非语义" in out or "词法" in out

    get_embedder.cache_clear()
    get_retriever.cache_clear()
    get_vector_store.cache_clear()
