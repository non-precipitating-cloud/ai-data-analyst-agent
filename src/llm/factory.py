"""LLM 工厂：基于 OpenAI 兼容接口构造 LangChain ChatModel。

通过 .env 的 LLM_BASE_URL / LLM_MODEL 可切换 DeepSeek / Qwen / Kimi / GLM / OpenAI 等。
之所以各家模型都能用 ChatOpenAI，是因为它们都提供了与 OpenAI 兼容的 HTTP 接口，
只需更换 base_url、model 与 api_key 三个参数即可。
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config.settings import get_settings


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """构造一个配置好的 LangChain ChatModel 实例。

    参数:
        temperature: 临时覆盖配置中的采样温度；
            None 时使用 .env 中 LLM_TEMPERATURE 的默认值。

    返回:
        可直接用于 LangGraph 节点/绑定工具的 ChatModel 实例（具体为 ChatOpenAI）。

    异常:
        RuntimeError: 未配置 LLM_API_KEY 时抛出，提示先填写 .env。
    """
    s = get_settings()

    # 前置校验：没有 API Key 无法调用任何模型，快速失败并给出明确指引
    if not s.llm_api_key:
        raise RuntimeError(
            "缺少 LLM_API_KEY。请复制 .env.example 为 .env 并填写 API Key。"
        )

    # 统一走 OpenAI 兼容协议：更换 base_url + model 即可在不同厂商间切换
    kwargs: dict = {
        # 模型名称（由 LLM_MODEL 注入）
        "model": s.llm_model,
        # API 密钥（由 LLM_API_KEY 注入）
        "api_key": s.llm_api_key,
        # 兼容接口地址（由 LLM_BASE_URL 注入）
        "base_url": s.llm_base_url,
        # 采样温度：本次调用显式传入则优先，否则回落到配置默认值
        "temperature": s.llm_temperature if temperature is None else temperature,
        # 单次 HTTP 请求超时（秒）：避免模型服务卡死时整个任务无限挂起
        "timeout": s.llm_timeout_seconds,
        # 瞬时错误自动重试次数：对 HTTP 429（限流）/5xx（服务端错误）/
        # 连接失败按指数退避重试，全部失败后才向上抛出
        "max_retries": s.llm_max_retries,
    }
    return ChatOpenAI(**kwargs)
