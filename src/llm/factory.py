"""LLM 工厂：基于 OpenAI 兼容接口构造 LangChain ChatModel。

通过 .env 的 LLM_BASE_URL / LLM_MODEL 可切换 DeepSeek / Qwen / Kimi / GLM / OpenAI 等。
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config.settings import get_settings


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """返回配置好的 ChatModel。"""
    s = get_settings()

    if not s.llm_api_key:
        raise RuntimeError(
            "缺少 LLM_API_KEY。请复制 .env.example 为 .env 并填写 API Key。"
        )

    kwargs: dict = {
        "model": s.llm_model,
        "api_key": s.llm_api_key,
        "base_url": s.llm_base_url,
        "temperature": s.llm_temperature if temperature is None else temperature,
    }
    return ChatOpenAI(**kwargs)
