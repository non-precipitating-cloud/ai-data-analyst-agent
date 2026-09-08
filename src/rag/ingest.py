"""知识库导入逻辑（供脚本与代码复用）。"""

from __future__ import annotations

import logging

from src.rag import get_retriever

logger = logging.getLogger(__name__)


def ingest_knowledge(force: bool = False) -> int:
    """导入知识库到向量存储，返回文本块数量。

    force=True 时强制清空并重新导入；否则存储非空则跳过。
    """
    retriever = get_retriever()
    if not force and retriever.store.count() > 0:
        logger.info("知识库已有 %d 条，跳过导入（force=False）", retriever.store.count())
        return retriever.store.count()
    return retriever.ingest()
