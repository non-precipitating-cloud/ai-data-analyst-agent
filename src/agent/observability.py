"""LLM 调用的可观测性与容错封装。

解决的问题：
1. **可追踪**：每次 LLM 调用都记录节点名、模型名、token 用量、耗时与成败，
   落进 AgentState.llm_calls 并最终写入数据库（见 src/db/service.py），
   使一次 Agent 运行可以被完整复盘。
2. **不崩溃**：单次 LLM 调用因超时/限流/连接抖动失败时，先按指数退避重试；
   仍失败则抛出 AgentLLMError 由节点决定降级策略，而不是让整张图直接崩掉。
3. **不重复**：把「调用 LLM 并记录用量」收敛到一个入口，避免各节点各写一份
   计时代码导致统计口径不一致。

关于费用：这里**只记录 token 用量与耗时，不换算金额**。不同厂商、不同模型、
不同时段的单价差异很大，且接口本身不返回计价信息；凭空乘一个单价会产出看似
精确、实则不可信的成本数字，因此本模块刻意不做这件事。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# 判定「瞬时故障」的异常类名（各家 SDK 命名不一，用名字匹配避免强耦合具体库）
_TRANSIENT_EXCEPTION_NAMES = {
    "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "ServiceUnavailableError", "APIError",
    "Timeout", "TimeoutError", "ConnectTimeout", "ReadTimeout",
    "ConnectionError", "RemoteProtocolError",
}


class AgentLLMError(RuntimeError):
    """LLM 调用在重试后仍然失败。

    单独定义异常类型，使节点能精确区分「模型不可用」与「工具执行出错」，
    从而采取不同的降级策略（前者走确定性兜底摘要，后者回传 LLM 自纠）。
    """


def _is_transient(exc: BaseException) -> bool:
    """判断异常是否属于「重试可能成功」的瞬时故障。

    参数：
        exc: 被捕获的异常。

    返回：
        True 表示值得重试（超时、限流、5xx、连接中断等）。
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if type(exc).__name__ in _TRANSIENT_EXCEPTION_NAMES:
        return True
    # HTTP 状态码：429 限流与 5xx 服务端错误都值得退避重试
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return False


def extract_usage(message: BaseMessage) -> dict[str, Any]:
    """从 LLM 返回消息中提取 token 用量。

    LangChain 会把用量标准化到 ``usage_metadata``；部分 OpenAI 兼容服务
    则只在 ``response_metadata['token_usage']`` 里给出。两者都尝试，
    都拿不到时返回空字典——**不猜测、不补零**，避免把未知当作 0 统计。

    参数：
        message: LLM 返回的消息对象。

    返回：
        {"input_tokens","output_tokens","total_tokens"} 的字典（可能为空）。
    """
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    meta = getattr(message, "response_metadata", None) or {}
    raw = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(raw, dict) and raw:
        return {
            "input_tokens": raw.get("prompt_tokens"),
            "output_tokens": raw.get("completion_tokens"),
            "total_tokens": raw.get("total_tokens"),
        }
    return {}


def _model_name(llm: BaseChatModel) -> str:
    """尽力取出模型标识（不同 LangChain 版本字段名略有差异）。"""
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(llm).__name__


def invoke_llm(
    messages: list[BaseMessage],
    *,
    llm: BaseChatModel,
    node: str,
    retries: int | None = None,
) -> tuple[BaseMessage, dict[str, Any]]:
    """调用 LLM 并返回 (响应消息, 调用记录)。

    失败时按指数退避重试（仅针对瞬时故障），最终仍失败则抛 AgentLLMError。
    无论成功失败都会产出一条记录，便于统计「调用了几次、失败了几次」。

    参数：
        messages: 传入模型的完整消息列表。
        llm: 已构造好的 ChatModel 实例（由各节点用 get_llm() 获取）。
        node: 发起调用的节点名，用于区分任务理解/规划/洞察/报告等环节。
        retries: 覆盖默认重试次数；None 时取配置 agent_llm_retries。

    返回：
        二元组 (响应消息, 记录字典)。记录含 node/model/duration_ms/success
        以及 token 用量字段。

    异常：
        AgentLLMError: 重试耗尽后仍失败。
    """
    # 延迟导入，避免模块级循环依赖（settings 会被多处引用）
    from src.config.settings import get_settings

    max_retries = get_settings().agent_llm_retries if retries is None else retries
    model = _model_name(llm)
    started = time.monotonic()
    last_error: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
        except Exception as e:  # noqa: BLE001 —— 需要区分瞬时/永久故障
            last_error = e
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if attempt < max_retries and _is_transient(e):
                # 指数退避：0.5s、1s、2s…… 给限流/抖动留出恢复时间
                delay = 0.5 * (2 ** attempt)
                logger.warning(
                    "LLM 调用失败（节点=%s，第 %d 次），%.1fs 后重试：%s",
                    node, attempt + 1, delay, e,
                )
                time.sleep(delay)
                continue
            logger.error("LLM 调用最终失败（节点=%s）：%s: %s", node, type(e).__name__, e)
            raise AgentLLMError(
                f"LLM 调用失败（节点={node}，模型={model}）：{type(e).__name__}: {e}"
            ) from e
        else:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            record: dict[str, Any] = {
                "node": node,
                "model": model,
                "duration_ms": elapsed_ms,
                "success": True,
                "attempts": attempt + 1,
            }
            record.update(extract_usage(response))
            logger.debug(
                "LLM 调用成功：节点=%s 模型=%s 耗时=%dms tokens=%s",
                node, model, elapsed_ms, record.get("total_tokens"),
            )
            return response, record

    # 理论上不可达（循环内必然 return 或 raise），保留以满足类型检查
    raise AgentLLMError(f"LLM 调用失败（节点={node}）：{last_error}")
