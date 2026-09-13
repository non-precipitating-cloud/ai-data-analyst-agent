"""Retriever：知识库检索器，封装向量存储并串起整个 RAG 链路。

入库（ingest）：加载 knowledge/*.md → 切分文本块 → 交由向量存储批量 embedding 写入。
检索（retrieve）：query 经同一 embedder 向量化 → 向量存储做相似度 Top-K 查询。
存储为空时检索会自动触发一次导入（离线自愈），调用方无需关心初始化顺序。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config.settings import get_settings
from src.rag.embeddings import get_embedder
from src.rag.loader import chunk_documents, load_knowledge_documents
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """知识库检索器。存储为空时自动从 knowledge/ 目录导入（离线自愈）。"""

    def __init__(self, embedder, store) -> None:
        """
        Args:
            embedder: 向量化实现（真实 embedding 或本地哈希），文档与 query 必须同源。
            store: 向量存储（pgvector 或内存实现），需与 embedder 维度匹配。
        """
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """检索与 query 最相关的 k 个知识块。

        Args:
            query: 用户问题或检索文本。
            k: 返回条数；为 None 时使用配置 rag_top_k。

        Returns:
            命中列表，每项含 text、metadata、score（相似度分数），按相关度降序。
        """
        k = k or get_settings().rag_top_k
        # 自愈：向量库为空（首次使用或被清空）时先导入再检索
        if self.store.count() == 0:
            self.ingest()
        # 内部完成 query embedding 与相似度 Top-K 查询
        return self.store.similarity_search(query, k)

    def ingest(self) -> int:
        """从 knowledge/ 导入知识库，返回文本块数量。

        Returns:
            成功写入向量存储的文本块数量；目录为空时返回 0。
        """
        settings = get_settings()
        # 第 1 步：加载 Markdown 文档（含分类/标题元数据）
        docs = load_knowledge_documents(settings.knowledge_dir)
        # 第 2 步：切分为带重叠的文本块
        chunks = chunk_documents(docs)
        if not chunks:
            logger.warning("知识库目录为空：%s", settings.knowledge_dir)
            return 0
        # 第 3 步：清空旧数据，保证全量重建而非追加重复
        self.store.clear()
        # 第 4 步：store 内部对文本块批量 embedding 后写入 pgvector/内存
        self.store.add_texts(
            [c["text"] for c in chunks], [c["metadata"] for c in chunks]
        )
        logger.info("知识库导入完成：%d 个文本块", len(chunks))
        return len(chunks)


@lru_cache
def get_retriever() -> KnowledgeRetriever:
    """返回进程级单例检索器。

    embedder 与 vector_store 各自也是单例，确保维度探测结果、数据库
    连接和内存数据在全局复用。
    """
    embedder = get_embedder()
    store = get_vector_store(embedder)
    return KnowledgeRetriever(embedder, store)
