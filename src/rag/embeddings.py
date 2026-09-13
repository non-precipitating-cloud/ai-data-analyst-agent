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
    """本地确定性字符 n-gram 哈希向量（离线回退，非语义）。

    用 MD5 把字符二元组/单字哈希到固定维度并做 L2 归一化：相同文本
    永远得到相同向量，共享词汇的文本余弦相似度更高。它不理解语义，
    仅保证无 API Key 时 RAG 链路在离线/测试环境仍可端到端跑通。
    """

    def __init__(self, dim: int = 384) -> None:
        """
        Args:
            dim: 向量维度（需与 pgvector 建表维度保持一致）。
        """
        self._dim = dim

    @property
    def dimension(self) -> int:
        """返回向量维度，供向量存储建表时动态确定 vector(n) 的 n。"""
        return self._dim

    def _embed(self, text: str) -> list[float]:
        """单条文本 → 定长归一化哈希向量（核心实现）。"""
        text = text.lower().strip()
        vec = [0.0] * self._dim
        # 特征 = 相邻两字的二元组（bigram）+ 单字，兼顾局部词组与单字匹配
        grams = [text[i : i + 2] for i in range(len(text) - 1)] + list(text)
        for g in grams:
            # MD5 摘要前 4 字节转整数再对维度取模，得到该特征命中的桶下标
            h = hashlib.md5(g.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            vec[idx] += 1.0  # 词袋式计数，碰撞时累加
        # L2 归一化，使余弦相似度退化为向量点积；全零时用 1.0 防止除零
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本块（入库时使用）。

        Args:
            texts: 文本块列表。

        Returns:
            与输入等长、等维度的向量列表。
        """
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """编码单条检索 query（与文档使用同一套特征，保证同空间可比）。"""
        return self._embed(text)


class OpenAICompatibleEmbedder:
    """OpenAI 兼容 Embedding（真实语义向量）。

    通过 langchain_openai 调用任意 OpenAI /v1/embeddings 兼容服务，
    得到真正表达语义的稠密向量，检索质量显著优于哈希回退。
    """

    def __init__(self, model: str, api_key: str, base_url: str = "") -> None:
        """
        Args:
            model: embedding 模型名。
            api_key: 服务 API Key。
            base_url: 可选的自定义服务地址（第三方/自建网关），为空用官方地址。
        """
        kwargs: dict = {"model": model, "api_key": api_key}
        # 仅在显式配置了自定义地址时传入，避免空字符串覆盖 SDK 默认端点
        if base_url:
            kwargs["base_url"] = base_url
        self._emb = OpenAIEmbeddings(**kwargs)
        # 维度懒探测缓存（不同模型维度不同，如 1536/768）
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        """返回模型实际输出维度（首次访问时发一次探测请求并缓存）。"""
        if self._dim is None:
            # 用任意短文本换取一条向量，其长度即模型维度
            self._dim = len(self._emb.embed_query("dim"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量调用 embedding API 编码文档文本块。"""
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """调用 embedding API 编码单条检索 query。"""
        return self._emb.embed_query(text)


@lru_cache
def get_embedder():
    """返回单例 embedder（有 API Key 用真实语义，否则本地哈希回退）。

    Returns:
        HashingEmbedder 或 OpenAICompatibleEmbedder，二者接口一致；
        lru_cache 保证全进程复用同一实例及其实测维度。
    """
    s = get_settings()
    # 分支：配置了 EMBEDDING_API_KEY → 走真实语义 embedding 服务
    if s.embedding_api_key:
        return OpenAICompatibleEmbedder(
            model=s.embedding_model,
            api_key=s.embedding_api_key,
            base_url=s.embedding_base_url,
        )
    # 回退分支：离线确定性哈希向量，维度取配置 embedding_dim（默认 384）
    return HashingEmbedder(dim=s.embedding_dim)
