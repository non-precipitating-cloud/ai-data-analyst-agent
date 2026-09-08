"""RAG 模块：文档加载 → 切分 → Embedding → 向量存储 → Retriever。"""

from src.rag.retriever import KnowledgeRetriever, get_retriever

__all__ = ["KnowledgeRetriever", "get_retriever"]
