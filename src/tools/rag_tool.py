"""RAG 工具：从数据分析知识库检索方法论知识。"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config.settings import get_settings
from src.rag import get_retriever


@tool
def retrieve_knowledge(query: str, top_k: int = 0) -> str:
    """从数据分析知识库检索相关方法论知识。

    当需要方法论指导时使用，例如：异常检测方法与阈值、相关性解读、销售/财务归因框架、
    数据清洗策略、统计方法选择等。返回相关知识片段、所属分类与相似度。
    """
    try:
        k = top_k or get_settings().rag_top_k
        results = get_retriever().retrieve(query, k)
    except Exception as e:  # noqa: BLE001
        return f"知识库检索失败：{type(e).__name__}: {e}"

    if not results:
        return "知识库为空或未检索到相关内容。请先运行 scripts/ingest_knowledge.py。"

    lines = []
    for r in results:
        cat = r["metadata"].get("category", "?")
        lines.append(f"[{cat}] (相似度 {r['score']}) {r['text'][:240]}")
    return "\n\n".join(lines)
