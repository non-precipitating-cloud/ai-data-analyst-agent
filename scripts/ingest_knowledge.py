"""知识库导入脚本。

用法：
    python scripts/ingest_knowledge.py

把 knowledge/ 目录下的 Markdown 文档切分、向量化后写入向量存储。
- PostgreSQL + pgvector 可用时：持久化写入数据库（推荐）。
- pgvector 不可用时：写入内存存储（不持久化，仅用于演示/测试）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许以脚本方式直接运行时导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.rag import get_retriever  # noqa: E402
from src.rag.vector_store import InMemoryVectorStore  # noqa: E402


def main() -> None:
    setup_logging()
    get_settings().ensure_dirs()

    retriever = get_retriever()
    n = retriever.ingest()

    store_type = type(retriever.store).__name__
    print(f"知识库导入完成：{n} 个文本块")
    print(f"向量存储类型：{store_type}")

    if isinstance(retriever.store, InMemoryVectorStore):
        print("⚠ 当前使用内存存储，导入结果不会持久化。")
        print("  请先启动 PostgreSQL + pgvector（docker compose up -d）后重新运行本脚本。")


if __name__ == "__main__":
    main()
