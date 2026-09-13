"""RAG 工具：从数据分析知识库检索方法论知识（检索增强生成）。

输入输出契约：
- 输入：自然语言查询 query（分析方法/业务框架类问题），可选 top_k（0 表示用配置默认值）。
- 输出：按相似度排序的知识片段纯文本，每行格式
  "[分类] (相似度 x.xxxx) 片段前 240 字"；
  知识库为空、未入库或检索异常时返回中文提示（不抛异常）。
- 知识库由 scripts/ingest_knowledge.py 预先切分、向量化写入 pgvector 或内存存储。
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config.settings import get_settings
from src.rag import get_retriever


@tool
def retrieve_knowledge(query: str, top_k: int = 0) -> str:
    """从数据分析知识库检索相关方法论知识（LangChain 工具入口）。

    当需要方法论指导时使用，例如：异常检测方法与阈值、相关性解读、销售/财务归因框架、
    数据清洗策略、统计方法选择等。返回相关知识片段、所属分类与相似度。

    参数：
        query: 检索问题文本。
        top_k: 返回片段数量；传 0 时取 settings.rag_top_k 默认值。
    返回值：
        str: 拼接后的多段知识文本；无结果时提示先运行入库脚本；
        检索异常时返回 "知识库检索失败：..." 提示字符串。
    """
    try:
        # top_k 为 0/缺省时回落到全局配置的检索条数
        k = top_k or get_settings().rag_top_k
        # 检索器为单例：内部封装 embedding 与向量库（pgvector 或内存）
        retriever = get_retriever()
        results = retriever.retrieve(query, k)
    except Exception as e:  # noqa: BLE001
        return f"知识库检索失败：{type(e).__name__}: {e}"

    if not results:
        return "知识库为空或未检索到相关内容。请先运行 scripts/ingest_knowledge.py。"

    # 把结构化检索结果压成 LLM 易读的文本，片段截断防止上下文膨胀
    lines = []
    for r in results:
        cat = r["metadata"].get("category", "?")
        lines.append(f"[{cat}] (相似度 {r['score']}) {r['text'][:240]}")

    # 如实标注检索方式：离线词法向量只做字面匹配，结论不应被当作语义检索的结果。
    # 没有这行提示，模型容易把「恰好字面重合」当成「语义相关」。
    embedder = getattr(retriever, "embedder", None)
    if embedder is not None and not getattr(embedder, "is_semantic", True):
        lines.append(
            f"\n[检索方式] 当前使用 {getattr(embedder, 'name', '未知')} "
            "离线词法向量（非语义模型），仅按字面重合度排序。"
            "若结果与问题明显无关，请忽略这些片段，不要强行引用。"
        )
    return "\n\n".join(lines)
