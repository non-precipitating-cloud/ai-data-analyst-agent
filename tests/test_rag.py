"""RAG 单元测试（离线：本地哈希向量 + 内存向量存储，不依赖 PG / 外部 API）。

覆盖 src.rag 各组件：
- HashingEmbedder：同文本向量确定性一致、相关文本相似度更高；
- loader：知识文档加载与分块（chunk）；
- InMemoryVectorStore：内存向量库的写入、相似检索、清空；
- KnowledgeRetriever：批量导入与检索、空库自动导入；
- pgvector 适配：向量序列化格式与真实 PgVectorStore 回归（PG 不可用则 skip）。
"""

from __future__ import annotations

from src.config.settings import get_settings
from src.rag.embeddings import HashingEmbedder
from src.rag.loader import chunk_documents, load_knowledge_documents
from src.rag.retriever import KnowledgeRetriever
from src.rag.vector_store import InMemoryVectorStore, _cosine, _vec_to_str


def test_hashing_embedder_deterministic() -> None:
    """同一文本两次嵌入结果应完全一致，且向量维度等于设定的 128。"""
    e = HashingEmbedder(dim=128)
    a = e.embed_query("销售额下降")
    b = e.embed_query("销售额下降")
    assert a == b
    assert len(a) == 128


def test_hashing_embedder_similar_texts_rank_higher() -> None:
    """语义相关文本（异常检测）的余弦相似度应高于无关文本（财务分析）。"""
    e = HashingEmbedder(dim=256)
    q = e.embed_query("异常检测 阈值 z-score")
    related = e.embed_query("异常检测方法 z-score IQR 阈值")
    unrelated = e.embed_query("财务分析 利润率 成本")
    assert _cosine(q, related) > _cosine(q, unrelated)


def test_load_knowledge_documents() -> None:
    """知识库目录应加载出至少 7 篇文档，且覆盖 7 个预设分类。"""
    docs = load_knowledge_documents(get_settings().knowledge_dir)
    assert len(docs) >= 7
    cats = {d.metadata["category"] for d in docs}
    # 七个业务分类都应存在
    for expected in ("数据分析基础", "统计分析", "销售分析", "财务分析", "异常检测", "数据清洗", "相关性分析"):
        assert expected in cats


def test_chunk_documents() -> None:
    """分块后块数不少于文档数，且每块带正文、category、source 元数据。"""
    docs = load_knowledge_documents(get_settings().knowledge_dir)
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= len(docs)
    for c in chunks:
        # 每块必须有非空正文
        assert c["text"]
        assert "category" in c["metadata"]
        assert "source" in c["metadata"]


def test_inmemory_store_add_and_search() -> None:
    """写入两条文本后，按“手机”检索的 Top1 应命中“苹果手机销量”。"""
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    store.add_texts(["苹果手机销量", "香蕉水果销量"], [{"cat": "a"}, {"cat": "b"}])
    assert store.count() == 2

    results = store.similarity_search("手机", k=1)
    assert results[0]["text"] == "苹果手机销量"


def test_inmemory_store_clear() -> None:
    """clear 后向量库条目数应归零。"""
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    store.add_texts(["x"], [{}])
    store.clear()
    assert store.count() == 0


def test_retriever_ingest_and_retrieve() -> None:
    """导入真实知识库后检索销售归因问题，应返回带分类与相似度分数的结果。"""
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    retriever = KnowledgeRetriever(e, store)

    # 全量导入知识库
    n = retriever.ingest()
    assert n > 0

    results = retriever.retrieve("销售下降 地区 产品 归因", k=3)
    assert results
    assert results[0]["metadata"]["category"]
    assert "score" in results[0]


def test_retriever_auto_ingest_when_empty() -> None:
    """空库首次检索时应自动触发导入，检索后库内条目数大于 0。"""
    e = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(e)
    retriever = KnowledgeRetriever(e, store)

    assert store.count() == 0
    results = retriever.retrieve("异常检测", k=2)
    assert results
    assert store.count() > 0  # 空库时自动导入


def test_vec_to_str_pgvector_format() -> None:
    """向量转字符串应符合 pgvector 的 "[1.000000,0.500000,-0.250000]" 字面量格式。"""
    assert _vec_to_str([1.0, 0.5, -0.25]) == "[1.000000,0.500000,-0.250000]"


def test_pgvector_add_texts_and_search() -> None:
    """回归测试：PgVectorStore.add_texts 不应因变量遮蔽报错（需真实 PG，不可用则跳过）。"""
    import pytest

    from src.config.settings import get_settings
    from src.rag.embeddings import get_embedder
    from src.rag.vector_store import PgVectorStore

    try:
        # 尝试连接真实 PostgreSQL + pgvector
        store = PgVectorStore(get_embedder(), get_settings().database_url)
    except Exception:  # noqa: BLE001
        # 本地没有 PG 环境时跳过，不算失败
        pytest.skip("PostgreSQL + pgvector 不可用")

    # 先清空历史数据，保证断言不受污染
    store.clear()
    try:
        n = store.add_texts(
            ["苹果手机销量", "香蕉水果销量"],
            [{"category": "a"}, {"category": "b"}],
        )
        assert n == 2
        results = store.similarity_search("手机", k=1)
        assert results[0]["text"] == "苹果手机销量"
    finally:
        # 无论成功失败都清空测试写入的数据
        store.clear()
