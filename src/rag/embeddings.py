"""Embedding Provider 层：把「谁来把文本变成向量」这件事显式区分开。

架构（三者接口一致，可互换）::

    EmbeddingProvider（抽象基类）
    ├── OpenAICompatibleEmbedder   真实语义向量：调用 OpenAI 兼容 /v1/embeddings
    ├── LexicalHashEmbedder        离线词法回退：字符 n-gram 哈希，**非语义**
    └── DeterministicTestEmbedder  测试替身：固定维度、可预测，**仅供测试**

为什么要分这么清楚：哈希向量与语义向量**长得一样**（都是定长浮点数组），
但检索质量天差地别——哈希向量只认字面重合，问「营收下滑原因」检索不到
标题写着「销售额下降归因」的文档。若不加区分，很容易把「离线能跑」误当成
「检索可用」，进而在报告里引用到完全不相关的知识。

因此这里做三件事：
1. 每个实现都带 ``name`` 与 ``is_semantic`` 两个属性，调用方可据此判断
   当前检索质量等级，并如实展示（见 src/tools/rag_tool.py 的返回文本）；
2. 回退到非语义实现时记录**警告级**日志，而不是静默降级；
3. 用 ``EMBEDDING_PROVIDER`` 配置项显式选择实现，默认 ``auto``
   （有 Key 用真实语义，无 Key 用离线词法），可强制为 ``hash`` 以便
   完全离线运行与测试——**测试永远不会因为缺 Key 而被迫联网**。
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Embedding 实现的统一接口。

    不继承 LangChain 的 Embeddings 基类，是为了让「离线回退」这种非语义实现
    也能合法存在而不被误当作语义模型；调用方只依赖下面这四个成员。
    """

    #: 供日志与界面展示的实现名
    name: str = "unknown"
    #: 是否为**语义**向量。False 表示只做字面匹配，检索质量显著更低
    is_semantic: bool = False

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度（决定 pgvector 建表时 vector(n) 的 n）。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本块（入库时使用）。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """编码单条检索 query（必须与文档使用同一空间，结果才可比）。"""


class LexicalHashEmbedder(EmbeddingProvider):
    """本地确定性字符 n-gram 哈希向量（**离线词法回退，不是语义模型**）。

    用 MD5 把字符二元组/单字哈希到固定维度并做 L2 归一化：相同文本永远得到
    相同向量，共享词汇的文本余弦相似度更高。

    它**不理解语义**：同义词（"销售额" vs "营收"）、改写句式都匹配不上。
    存在的意义只有一个——在没有 Embedding API Key 的环境（离线开发、CI、
    单元测试）里让整条 RAG 链路仍可端到端跑通，而不是让进程启动就失败。
    生产环境请配置 EMBEDDING_API_KEY 使用真实语义向量。
    """

    name = "lexical-hash"
    is_semantic = False

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


# 向后兼容别名：历史代码/测试里以 HashingEmbedder 引用该实现。
# 新代码请使用语义更明确的 LexicalHashEmbedder。
HashingEmbedder = LexicalHashEmbedder


class DeterministicTestEmbedder(EmbeddingProvider):
    """测试专用确定性向量（**仅供测试**，不具备任何检索能力）。

    与 LexicalHashEmbedder 的区别：本类刻意**不做任何真实特征提取**，
    只保证「同输入 → 同输出」「不同输入 → 不同输出」，用于验证向量存储、
    检索流程、维度处理等管道逻辑，避免测试依赖真实模型或网络。

    它返回的相似度分数**没有语义含义**，任何基于它做的检索质量断言都是
    无效的——需要评估检索效果时请用真实 Embedding。
    """

    name = "test-deterministic"
    is_semantic = False

    def __init__(self, dim: int = 8) -> None:
        """
        Args:
            dim: 向量维度，默认取很小的值以便测试断言。
        """
        self._dim = dim

    @property
    def dimension(self) -> int:
        """返回固定向量维度。"""
        return self._dim

    def _embed(self, text: str) -> list[float]:
        """用摘要字节填充固定维度向量（仅要求确定性与可区分性）。"""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 逐字节映射到 [-1, 1]，维度不足时循环复用摘要
        raw = [((digest[i % len(digest)] / 255.0) * 2 - 1) for i in range(self._dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码（测试用）。"""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """编码单条 query（测试用）。"""
        return self._embed(text)


class OpenAICompatibleEmbedder(EmbeddingProvider):
    """OpenAI 兼容 Embedding（真实语义向量）。

    通过 langchain_openai 调用任意 OpenAI /v1/embeddings 兼容服务，
    得到真正表达语义的稠密向量，检索质量显著优于词法回退。
    """

    name = "openai-compatible"
    is_semantic = True

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


def build_embedder(provider: str | None = None) -> EmbeddingProvider:
    """按配置构造 Embedding 实现（不做缓存，便于测试与显式切换）。

    Args:
        provider: 强制指定实现，取值 ``openai`` / ``hash`` / ``test``；
            为 None 时读取配置项 ``EMBEDDING_PROVIDER``（默认 ``auto``）。

    Returns:
        EmbeddingProvider 实例。

    Raises:
        ValueError: 显式要求 ``openai`` 但未配置 API Key（宁可快速失败，
            也不要在用户以为在用语义检索时悄悄退回词法匹配）。
    """
    s = get_settings()
    mode = (provider or s.embedding_provider or "auto").strip().lower()

    if mode == "test":
        # 测试替身：显式指定才使用，绝不自动选中
        return DeterministicTestEmbedder(dim=s.embedding_dim)
    if mode == "hash":
        # 强制离线词法模式：CI / 离线开发 / 无 Key 演示
        logger.info("Embedding 使用离线词法回退（EMBEDDING_PROVIDER=hash），检索仅做字面匹配")
        return LexicalHashEmbedder(dim=s.embedding_dim)
    if mode == "openai":
        if not s.embedding_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai 但未配置 EMBEDDING_API_KEY。"
                "请填写 Key，或改用 EMBEDDING_PROVIDER=hash 走离线词法回退。"
            )
        return OpenAICompatibleEmbedder(
            model=s.embedding_model,
            api_key=s.embedding_api_key,
            base_url=s.embedding_base_url,
        )

    # auto：有 Key 用真实语义，没有则回退离线词法并**明确告警**
    if s.embedding_api_key:
        return OpenAICompatibleEmbedder(
            model=s.embedding_model,
            api_key=s.embedding_api_key,
            base_url=s.embedding_base_url,
        )
    logger.warning(
        "未配置 EMBEDDING_API_KEY，RAG 回退到离线词法向量（非语义，检索质量有限）。"
        "生产环境请配置 Key 或显式设置 EMBEDDING_PROVIDER=hash 以表明这是有意为之。"
    )
    return LexicalHashEmbedder(dim=s.embedding_dim)


@lru_cache
def get_embedder() -> EmbeddingProvider:
    """返回单例 embedder（进程内复用，包含维度探测结果）。

    实现选择逻辑见 build_embedder；本函数只负责缓存，
    测试需要切换实现时调用 ``get_embedder.cache_clear()``。
    """
    return build_embedder()
