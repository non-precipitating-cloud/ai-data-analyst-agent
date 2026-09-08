"""Retriever：封装向量存储，提供知识检索能力。"""

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
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or get_settings().rag_top_k
        if self.store.count() == 0:
            self.ingest()
        return self.store.similarity_search(query, k)

    def ingest(self) -> int:
        """从 knowledge/ 导入知识库，返回文本块数量。"""
        settings = get_settings()
        docs = load_knowledge_documents(settings.knowledge_dir)
        chunks = chunk_documents(docs)
        if not chunks:
            logger.warning("知识库目录为空：%s", settings.knowledge_dir)
            return 0
        self.store.clear()
        self.store.add_texts(
            [c["text"] for c in chunks], [c["metadata"] for c in chunks]
        )
        logger.info("知识库导入完成：%d 个文本块", len(chunks))
        return len(chunks)


@lru_cache
def get_retriever() -> KnowledgeRetriever:
    """返回单例检索器。"""
    embedder = get_embedder()
    store = get_vector_store(embedder)
    return KnowledgeRetriever(embedder, store)
