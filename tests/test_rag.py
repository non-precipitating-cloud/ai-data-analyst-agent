"""RAG 单元测试（离线：本地哈希向量 + 内存向量存储，不依赖 PG / 外部 API）。"""

from __future__ import annotations

from src.config.settings import get_settings
from src.rag.embeddings import HashingEmbedder
from src.rag.loader import chunk_documents, load_knowledge_documents
from src.rag.retriever import KnowledgeRetriever
from src.rag.vector_store import InMemoryVectorStore, _cosine, _vec_to_str


def test_hashing_embedder_deterministic() -> None:
    e = HashingEmbedder(dim=128)
    a = e.embed_query("销售额下降")
    b = e.embed_query("销售额下降")
    assert a == b
    assert len(a) == 128


def test_hashing_embedder_similar_texts_rank_higher() -> None:
    e = HashingEmbedder(dim=256)
    q = e.embed_query("异常检测 阈值 z-score")
    related = e.embed_query("异常检测方法 z-score IQR 阈值")
    unrelated = e.embed_query("财务分析 利润率 成本")
    assert _cosine(q, related) > _cosine(q, unrelated)


def test_load_knowledge_documents() -> None:
    docs = load_knowledge_documents(get_settings().knowledge_dir)
    assert len(docs) >= 7
    cats = {d.metadata["category"] for d in docs}
    for expected in ("数据分析基础", "统计分析", "销售分析", "财务分析", "异常检测", "数据清洗", "相关性分析"):
        assert expected in cats


def test_chunk_documents() -> None:
    docs = load_knowledge_documents(get_settings().knowledge_dir)
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= len(docs)
    for c in chunks:
        assert c["text"]
        assert "category" in c["metadata"]
        assert "source" in c["metadata"]


def test_inmemory_store_add_and_search() -> None:
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    store.add_texts(["苹果手机销量", "香蕉水果销量"], [{"cat": "a"}, {"cat": "b"}])
    assert store.count() == 2

    results = store.similarity_search("手机", k=1)
    assert results[0]["text"] == "苹果手机销量"


def test_inmemory_store_clear() -> None:
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    store.add_texts(["x"], [{}])
    store.clear()
    assert store.count() == 0


def test_retriever_ingest_and_retrieve() -> None:
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    retriever = KnowledgeRetriever(e, store)

    n = retriever.ingest()
    assert n > 0

    results = retriever.retrieve("销售下降 地区 产品 归因", k=3)
    assert results
    assert results[0]["metadata"]["category"]
    assert "score" in results[0]


def test_retriever_auto_ingest_when_empty() -> None:
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    retriever = KnowledgeRetriever(e, store)

    assert store.count() == 0
    results = retriever.retrieve("异常检测", k=2)
    assert results
    assert store.count() > 0  # 空库时自动导入


def test_vec_to_str_pgvector_format() -> None:
    assert _vec_to_str([1.0, 0.5, -0.25]) == "[1.000000,0.500000,-0.250000]"
