"""向量存储：PostgreSQL + pgvector（真实持久化）与内存实现（离线回退）。

两套实现对外接口一致（add_texts / similarity_search / count / clear）：
- PgVectorStore：向量落库到 PostgreSQL 的 pgvector 扩展，表结构维度随
  embedder 实际维度动态创建，检索用数据库端余弦距离算子 <=>；
- InMemoryVectorStore：进程内列表 + numpy 余弦计算，无需任何外部依赖，
  供离线开发、单元测试及数据库不可用时降级使用。
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

import numpy as np
from sqlalchemy import create_engine, text

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# 知识库文本块在 pgvector 中的统一表名
_TABLE = "knowledge_chunks"


def _cosine(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度（内存实现用）。

    Args:
        a: 向量 A（如 query 向量）。
        b: 向量 B（如文档块向量）。

    Returns:
        余弦相似度，范围约 [-1, 1]，越接近 1 越相似；
        任一向量为零向量时返回 0.0，避免除零。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _vec_to_str(vec: list[float]) -> str:
    """把浮点向量转成 pgvector 字面量格式 ``[v1,v2,...]``。

    pgvector 的文本输入要求方括号包裹、逗号分隔；保留 6 位小数
    兼顾精度与 SQL 报文长度。
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class InMemoryVectorStore:
    """内存向量存储（离线 / 测试用）。

    所有文本块、元数据与向量保存在进程内列表中，重启即丢失；
    检索时用 numpy 暴力计算 query 与全量块的余弦相似度并排序。
    """

    def __init__(self, embedder) -> None:
        """
        Args:
            embedder: 用于写入与检索时向量化文本的 embedder。
        """
        self.embedder = embedder
        self._items: list[dict] = []

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """批量 embedding 文本并追加到内存列表。

        Args:
            texts: 待入库的文本块列表。
            metadatas: 与 texts 等长的元数据列表；为 None 时用空字典占位。

        Returns:
            实际写入的文本块数量。
        """
        metadatas = metadatas or [{}] * len(texts)
        embeddings = self.embedder.embed_documents(texts)
        # 注意：循环变量不能命名为 text，否则会遮蔽模块顶部从 sqlalchemy 导入的 text()
        for chunk, meta, emb in zip(texts, metadatas, embeddings):
            self._items.append({"text": chunk, "metadata": meta, "embedding": emb})
        return len(texts)

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        """暴力余弦相似度检索 Top-K。

        Args:
            query: 检索问题。
            k: 返回条数。

        Returns:
            命中列表（text/metadata/score），按相似度降序。
        """
        q = self.embedder.embed_query(query)
        # 与库内每个块计算余弦相似度
        scored = [(_cosine(q, it["embedding"]), it) for it in self._items]
        # 按分数降序排列（取负号实现降序 key）
        scored.sort(key=lambda x: -x[0])
        return [
            {"text": it["text"], "metadata": it["metadata"], "score": round(s, 4)}
            for s, it in scored[:k]
        ]

    def count(self) -> int:
        """返回当前内存中的文本块数量。"""
        return len(self._items)

    def clear(self) -> None:
        """清空全部内存文本块。"""
        self._items = []


class PgVectorStore:
    """PostgreSQL + pgvector 向量存储。

    表 knowledge_chunks 保存正文、JSONB 元数据与 vector 类型向量；
    向量列的维度在建表时依据 embedder.dimension 动态确定。
    """

    def __init__(self, embedder, connection_string: str) -> None:
        """创建数据库引擎并初始化扩展与表。

        Args:
            embedder: 向量化实现；其 dimension 决定表向量列维度。
            connection_string: SQLAlchemy 数据库连接串。
        """
        self.embedder = embedder
        # 3 秒连接超时：数据库不可达时快速失败，以便上层回退内存实现
        self.engine = create_engine(
            connection_string, connect_args={"connect_timeout": 3}
        )
        self._init_table()

    def _existing_vector_dim(self, conn) -> int | None:
        """读取已存在的 knowledge_chunks.embedding 列的维度。

        用途：检测「换了 Embedding 实现」的情况——例如先用离线词法（384 维）
        入库，之后配上真实语义模型（1536 维）。此时旧表的列维度不匹配，
        不仅插入会报错，即便维度凑巧相同，不同模型产出的向量也不在同一空间，
        检索结果毫无意义。

        Args:
            conn: 已打开的数据库连接。

        Returns:
            现有向量列的维度；表或列不存在时返回 None。
        """
        # to_regclass 在表不存在时返回 NULL（而不是像 '表名'::regclass 那样抛错）
        row = conn.execute(
            text(
                """
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = to_regclass(:tbl)
                  AND attname = 'embedding'
                  AND NOT attisdropped
                """
            ),
            {"tbl": _TABLE},
        ).fetchone()
        if not row or not row[0]:
            return None
        # 列类型形如 "vector(384)"，取出括号内的维度
        m = re.search(r"\((\d+)\)", str(row[0]))
        return int(m.group(1)) if m else None

    def _init_table(self) -> None:
        """确保 pgvector 扩展已启用、知识块表与向量索引存在（幂等）。

        额外处理维度变更：若已存在的向量列维度与当前 embedder 不一致，
        说明 Embedding 实现被换过，旧向量无法复用——此时重建表并明确告警，
        而不是留着它让后续插入/检索在运行时报出难以理解的错误。
        """
        # 维度由 embedder 动态决定：离线词法回退 384 维，真实模型可能是 1536 等
        dim = self.embedder.dimension
        with self.engine.begin() as conn:
            # 启用向量扩展（docker/init.sql 也会执行，这里保证非容器环境同样可用）
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            existing = self._existing_vector_dim(conn)
            if existing is not None and existing != dim:
                logger.warning(
                    "知识库向量维度不一致（库中 %d 维，当前 Embedding 产出 %d 维，"
                    "实现=%s）：旧向量来自另一套 Embedding，无法与本模型的结果比较，"
                    "将重建知识块表。请随后重新运行 scripts/ingest_knowledge.py 导入。",
                    existing, dim, getattr(self.embedder, "name", "unknown"),
                )
                # 表与索引一并删除重建，避免残留的 HNSW 索引与新列维度冲突
                conn.execute(text(f"DROP TABLE IF EXISTS {_TABLE}"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_TABLE} (
                        id SERIAL PRIMARY KEY,
                        doc_id TEXT,
                        category TEXT,
                        chunk_index INTEGER,
                        content TEXT,
                        metadata JSONB,
                        embedding vector({dim})
                    )
                    """  # 向量列维度随 embedder 动态生成 DDL
                )
            )
            # 在 embedding 列上建 HNSW 近似最近邻索引（余弦距离算子族 vector_cosine_ops，
            # 与检索使用的 <=> 对应）。不建索引时 ORDER BY <=> 是全表精确排序，
            # 知识块上千条后检索明显变慢；HNSW 以极小精度损失换取亚线性检索性能。
            # IF NOT EXISTS：表已存在/索引已建时跳过，重复执行不报错
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS ix_{_TABLE}_embedding_hnsw
                    ON {_TABLE} USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """批量 embedding 文本并逐行插入 pgvector 表。

        Args:
            texts: 文本块列表。
            metadatas: 对应的元数据列表（source/category/chunk_index 等）。

        Returns:
            插入的行数。
        """
        metadatas = metadatas or [{}] * len(texts)
        # 先批量调用 embedding，再在同一事务中逐行插入
        embeddings = self.embedder.embed_documents(texts)
        with self.engine.begin() as conn:
            for chunk, meta, emb in zip(texts, metadatas, embeddings):
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {_TABLE}
                            (doc_id, category, chunk_index, content, metadata, embedding)
                        VALUES
                            (:doc_id, :category, :chunk_index, :content,
                             CAST(:metadata AS jsonb), CAST(:embedding AS vector))
                        """  # JSON 文本显式 CAST 为 jsonb；[..] 文本 CAST 为 vector
                    ),
                    {
                        "doc_id": meta.get("source", ""),
                        "category": meta.get("category", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "content": chunk,
                        "metadata": json.dumps(meta, ensure_ascii=False),
                        # 向量以 pgvector 字面量字符串传入并强转
                        "embedding": _vec_to_str(emb),
                    },
                )
        return len(texts)

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        """用 pgvector 余弦距离算子做数据库端相似度检索。

        Args:
            query: 检索问题（先经 embedder 向量化）。
            k: 返回条数。

        Returns:
            命中列表（text/metadata/score），按相似度降序。
        """
        q_str = _vec_to_str(self.embedder.embed_query(query))
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT content, metadata, 1 - (embedding <=> CAST(:q AS vector)) AS similarity
                    FROM {_TABLE}
                    ORDER BY embedding <=> CAST(:q AS vector)
                    LIMIT :k
                    """
                    # <=> 是 pgvector 的余弦距离算子（取值 0~2，越小越相似），
                    # 因此 1 - distance 即相似度；ORDER BY 距离升序 = 相似度降序
                ),
                {"q": q_str, "k": k},
            ).fetchall()
        results = []
        for content, metadata, similarity in rows:
            # 不同驱动下 JSONB 可能已被解析成 dict，也可能仍是字符串
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
            results.append(
                {"text": content, "metadata": meta, "score": round(float(similarity), 4)}
            )
        return results

    def count(self) -> int:
        """返回知识块表中的总行数（也用于连通性自检）。"""
        with self.engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {_TABLE}")).scalar())

    def clear(self) -> None:
        """删除知识块表中的全部行（全量重建前调用，表结构保留）。"""
        with self.engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {_TABLE}"))


@lru_cache
def get_vector_store(embedder):
    """返回向量存储单例：优先 pgvector，连接失败则回退到内存实现。

    Args:
        embedder: 向量化实现，传给具体存储构造函数。

    Returns:
        PgVectorStore（数据库可用）或 InMemoryVectorStore（降级）。
    """
    s = get_settings()
    try:
        store = PgVectorStore(embedder, s.database_url)
        store.count()  # 建表成功不等于可查询，额外执行一次 COUNT 校验连接真正可用
        logger.info(
            "向量存储：PostgreSQL + pgvector（Embedding 实现=%s，语义=%s）",
            getattr(embedder, "name", "unknown"),
            getattr(embedder, "is_semantic", False),
        )
        return store
    except Exception as e:  # noqa: BLE001
        # 数据库不可用（未启动/无扩展/鉴权失败等）时降级，保证 RAG 离线仍可运行。
        # 注意内存存储**不持久化**：进程重启后需要重新导入知识库。
        logger.warning("pgvector 不可用（%s），回退到内存向量存储（不持久化）", e)
        return InMemoryVectorStore(embedder)
