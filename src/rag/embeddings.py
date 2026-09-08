"""Embedding 层。

- 配置了 EMBEDDING_API_KEY 时使用 OpenAI 兼容 Embedding（真实语义向量）。
- 未配置时回退到本地确定性哈希向量，保证离线可运行、可测试。
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from src.config.settings import get_settings


class HashingEmbedder:
    """本地确定性字符 n-gram 哈希向量（离线回退，非语义）。"""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        text = text.lower().strip()
        vec = [0.0] * self._dim
        grams = [text[i : i + 2] for i in range(len(text) - 1)] + list(text)
        for g in grams:
            h = hashlib.md5(g.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAICompatibleEmbedder:
    """OpenAI 兼容 Embedding（真实语义向量）。"""

    def __init__(self, model: str, api_key: str, base_url: str = "") -> None:
        kwargs: dict = {"model": model, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._emb = OpenAIEmbeddings(**kwargs)
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self._emb.embed_query("dim"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_query(text)


@lru_cache
def get_embedder():
    """返回单例 embedder（有 API Key 用真实语义，否则本地哈希回退）。"""
    s = get_settings()
    if s.embedding_api_key:
        return OpenAICompatibleEmbedder(
            model=s.embedding_model,
            api_key=s.embedding_api_key,
            base_url=s.embedding_base_url,
        )
    return HashingEmbedder(dim=s.embedding_dim)
