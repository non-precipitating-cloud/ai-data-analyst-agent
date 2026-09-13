"""Redis 客户端单例与可用性检测（优雅降级的基础）。

- ``get_redis`` 懒加载并复用全局唯一的 redis.Redis 实例，避免重复建连；
- ``is_redis_available`` 通过 PING 探测一次并缓存结果，供 Cache /
  SessionStore 在每次读写前快速判断是否走降级路径；
- ``reset_redis`` 仅供测试重置全局状态，保证用例之间相互隔离。
"""

from __future__ import annotations

import logging

import redis

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# 全局 Redis 客户端单例（首次调用 get_redis 时创建）
_redis = None
# 可用性探测结果缓存：None=尚未探测，True/False=探测结论
_redis_available: bool | None = None


def get_redis():
    """获取全局唯一的 Redis 客户端（懒加载单例）。

    Returns:
        redis.Redis: 按配置 redis_url 创建的客户端实例，
        decode_responses=True 表示读写值均为 str 而非 bytes。
    """
    global _redis
    if _redis is None:
        # 从配置的连接 URL 创建客户端；3 秒连接超时，避免 Redis 宕机时长时间阻塞
        _redis = redis.Redis.from_url(
            get_settings().redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )
    return _redis


def is_redis_available() -> bool:
    """检测 Redis 是否可用（结果只探测一次并缓存）。

    Returns:
        True 表示 PING 成功；连接/探测异常时记录告警并返回 False，
        上层据此走“无缓存/无会话持久化”的降级逻辑。
    """
    global _redis_available
    # 已探测过则直接复用结论，避免每次操作都 PING
    if _redis_available is not None:
        return _redis_available
    try:
        # PING 是最轻量的连通性 + 鉴权校验
        get_redis().ping()
        _redis_available = True
    except Exception as e:  # noqa: BLE001
        logger.warning("Redis 不可用（%s），会话/缓存降级", e)
        _redis_available = False
    return _redis_available


def reset_redis() -> None:
    """重置客户端实例与可用性缓存（主要用于测试隔离）。

    下次调用 get_redis / is_redis_available 时会重新建连、重新探测。
    """
    global _redis, _redis_available
    _redis = None
    _redis_available = None
