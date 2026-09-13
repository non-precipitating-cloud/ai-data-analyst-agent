"""知识库导入脚本（RAG 离线建库）。

用法：
    python scripts/ingest_knowledge.py

脚本用途：
- 把 knowledge/ 目录下的 Markdown 方法论文档读取、切分为文本块，
  调用 embedding 模型向量化后写入向量存储，供 retrieve_knowledge 工具检索。
- PostgreSQL + pgvector 可用时：持久化写入数据库（推荐，重启不丢）。
- pgvector 不可用时：写入内存存储（不持久化，进程结束即失效，仅用于演示/测试）。
- 导入完成后打印文本块数量与实际使用的向量存储类型，并在内存模式下给出告警。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许以脚本方式直接运行（python scripts/xxx.py）时把项目根目录加入模块搜索路径，
# 否则下面的 `from src.xxx import ...` 无法解析
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.rag import get_retriever  # noqa: E402
from src.rag.vector_store import InMemoryVectorStore  # noqa: E402


def main() -> None:
    """脚本入口：执行一次知识库全量导入并打印结果摘要。

    返回值：
        None；副作用是把 knowledge/ 文档的文本块与向量写入向量存储，
        并向终端打印块数量、存储类型与持久化告警。
    """
    setup_logging()
    # 确保 knowledge/、reports/ 等运行所需目录存在
    get_settings().ensure_dirs()

    # 获取全局检索器单例：内部按配置选择 pgvector 或内存存储
    retriever = get_retriever()
    # ingest() 完成“读取 → 切分 → 向量化 → 写入”，返回入库文本块数
    n = retriever.ingest()

    store_type = type(retriever.store).__name__
    print(f"知识库导入完成：{n} 个文本块")
    print(f"向量存储类型：{store_type}")

    # 内存模式提醒：本次导入随进程消失，需启动 pgvector 后重跑才能持久化
    if isinstance(retriever.store, InMemoryVectorStore):
        print("⚠ 当前使用内存存储，导入结果不会持久化。")
        print("  请先启动 PostgreSQL + pgvector（docker compose up -d）后重新运行本脚本。")


if __name__ == "__main__":
    main()
