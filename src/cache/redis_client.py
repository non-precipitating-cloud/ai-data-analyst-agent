"""Redis 客户端与可用性检测（优雅降级）。"""

from __future__ import annotations

import logging

import redis

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_redis = None
_redis_available: bool | None = None


def get_redis():
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(
            get_settings().redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )
    return _redis


def is_redis_available() -> bool:
    """检测 Redis 是否可用（缓存结果）。"""
    global _redis_available
    if _redis_available is not None:
        return _redis_available
    try:
        get_redis().ping()
        _redis_available = True
    except Exception as e:  # noqa: BLE001
        logger.warning("Redis 不可用（%s），会话/缓存降级", e)
        _redis_available = False
    return _redis_available


def reset_redis() -> None:
    """重置客户端与可用性缓存（测试隔离用）。"""
    global _redis, _redis_available
    _redis = None
    _redis_available = None
