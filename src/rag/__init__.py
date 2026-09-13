"""RAG（检索增强生成）模块。

完整链路：
    Markdown 知识文档（loader 加载）
        → 字符窗口切分为带重叠的文本块（loader.chunk_documents）
        → 文本块转向量（embeddings：真实 API embedding 或本地哈希回退）
        → 写入向量存储（vector_store：pgvector 优先，内存实现回退）
        → 检索时 query 向量化后做余弦相似度 Top-K 查询（retriever）

本包对外只暴露检索器 KnowledgeRetriever 及其单例 get_retriever，
Agent/MCP 工具通过它完成知识库的导入与检索。
"""

# 检索器：串联“加载→切分→向量化→入库→相似度检索”整条链路的统一入口
from src.rag.retriever import KnowledgeRetriever, get_retriever

__all__ = ["KnowledgeRetriever", "get_retriever"]
