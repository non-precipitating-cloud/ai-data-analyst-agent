"""通用 Redis Cache（JSON 序列化 + TTL）。"""

from __future__ import annotations

import json
import logging

from src.cache.redis_client import get_redis, is_redis_available
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class Cache:
    """通用缓存。key: cache:{key}，JSON 序列化，带 TTL。"""

    def __init__(self, prefix: str = "cache") -> None:
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def set_cache(self, key: str, value, ttl: int | None = None) -> bool:
        if not is_redis_available():
            return False
        try:
            ex = ttl if ttl is not None else get_settings().redis_cache_ttl
            get_redis().set(
                self._key(key), json.dumps(value, ensure_ascii=False, default=str), ex=ex
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 写入缓存失败：%s", e)
            return False

    def get_cache(self, key: str):
        if not is_redis_available():
            return None
        try:
            raw = get_redis().get(self._key(key))
            return json.loads(raw) if raw is not None else None
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 读取缓存失败：%s", e)
            return None

    def delete_cache(self, key: str) -> bool:
        if not is_redis_available():
            return False
        try:
            get_redis().delete(self._key(key))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 删除缓存失败：%s", e)
            return False
