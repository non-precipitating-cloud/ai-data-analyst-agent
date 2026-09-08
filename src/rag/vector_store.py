"""向量存储：PostgreSQL + pgvector（真实），内存实现（离线回退）。"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np
from sqlalchemy import create_engine, text

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_TABLE = "knowledge_chunks"


def _cosine(a: list[float], b: list[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _vec_to_str(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class InMemoryVectorStore:
    """内存向量存储（离线 / 测试用）。"""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._items: list[dict] = []

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        metadatas = metadatas or [{}] * len(texts)
        embeddings = self.embedder.embed_documents(texts)
        for text, meta, emb in zip(texts, metadatas, embeddings):
            self._items.append({"text": text, "metadata": meta, "embedding": emb})
        return len(texts)

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        q = self.embedder.embed_query(query)
        scored = [(_cosine(q, it["embedding"]), it) for it in self._items]
        scored.sort(key=lambda x: -x[0])
        return [
            {"text": it["text"], "metadata": it["metadata"], "score": round(s, 4)}
            for s, it in scored[:k]
        ]

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []


class PgVectorStore:
    """PostgreSQL + pgvector 向量存储。"""

    def __init__(self, embedder, connection_string: str) -> None:
        self.embedder = embedder
        self.engine = create_engine(
            connection_string, connect_args={"connect_timeout": 3}
        )
        self._init_table()

    def _init_table(self) -> None:
        dim = self.embedder.dimension
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
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
                    """
                )
            )

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        metadatas = metadatas or [{}] * len(texts)
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
                        """
                    ),
                    {
                        "doc_id": meta.get("source", ""),
                        "category": meta.get("category", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "content": chunk,
                        "metadata": json.dumps(meta, ensure_ascii=False),
                        "embedding": _vec_to_str(emb),
                    },
                )
        return len(texts)

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
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
                ),
                {"q": q_str, "k": k},
            ).fetchall()
        results = []
        for content, metadata, similarity in rows:
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
            results.append(
                {"text": content, "metadata": meta, "score": round(float(similarity), 4)}
            )
        return results

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {_TABLE}")).scalar())

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {_TABLE}"))


@lru_cache
def get_vector_store(embedder):
    """返回向量存储：优先 pgvector，连接失败则回退到内存实现。"""
    s = get_settings()
    try:
        store = PgVectorStore(embedder, s.database_url)
        store.count()  # 校验连接真正可用
        logger.info("向量存储：PostgreSQL + pgvector")
        return store
    except Exception as e:  # noqa: BLE001
        logger.warning("pgvector 不可用（%s），回退到内存向量存储", e)
        return InMemoryVectorStore(embedder)
