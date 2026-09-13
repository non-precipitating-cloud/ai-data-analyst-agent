"""知识库导入入口（供命令行脚本与应用代码复用）。

RAG 入库链路的“一键触发层”：通过单例 retriever 完成
加载 knowledge/*.md → 切分 → embedding → 写入向量存储的全过程，
并通过 force 参数支持“跳过已导入”与“强制全量重建”两种模式。
"""

from __future__ import annotations

import logging

from src.rag import get_retriever

logger = logging.getLogger(__name__)


def ingest_knowledge(force: bool = False) -> int:
    """导入知识库到向量存储，返回文本块数量。

    Args:
        force: True 时强制清空旧数据并重新导入；
            False（默认）时若向量存储非空则跳过，避免重复入库。

    Returns:
        向量存储中最终的文本块数量。
    """
    retriever = get_retriever()
    # 非强制模式下做幂等保护：已有内容则直接返回现有条数
    if not force and retriever.store.count() > 0:
        logger.info("知识库已有 %d 条，跳过导入（force=False）", retriever.store.count())
        return retriever.store.count()
    # 强制重建或首次导入：内部会 clear 后重新走完整入库链路
    return retriever.ingest()
