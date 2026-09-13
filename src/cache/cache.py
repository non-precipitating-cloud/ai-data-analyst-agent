"""通用 Redis 缓存（JSON 序列化 + TTL 过期）。

在 Agent 中用于缓存开销较大、可复用的结果（如 embedding、检索结果等）：
- key 统一加 ``cache:`` 前缀，value 以 JSON 文本存储；
- 支持按条设置 TTL，未指定时使用配置中的默认过期时间；
- Redis 不可用或读写异常时一律“静默降级”（返回 False/None），
  缓存失效不应影响业务主流程。
"""

from __future__ import annotations

import json
import logging

from src.cache.redis_client import get_redis, is_redis_available
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class Cache:
    """通用缓存。key: cache:{key}，JSON 序列化，带 TTL。

    不同业务可通过不同 prefix 隔离命名空间，避免 key 冲突。
    """

    def __init__(self, prefix: str = "cache") -> None:
        """初始化缓存实例。

        Args:
            prefix: Redis key 前缀，最终 key 形如 ``{prefix}:{key}``。
        """
        self.prefix = prefix

    def _key(self, key: str) -> str:
        """拼接带前缀的完整 Redis key。

        Args:
            key: 业务侧使用的相对 key。

        Returns:
            带命名空间前缀的实际 Redis key。
        """
        return f"{self.prefix}:{key}"

    def set_cache(self, key: str, value, ttl: int | None = None) -> bool:
        """写入缓存（JSON 序列化，带过期时间）。

        Args:
            key: 缓存 key（不含前缀）。
            value: 任意可 JSON 序列化的 Python 对象。
            ttl: 过期秒数；为 None 时使用配置项 redis_cache_ttl 的默认值。

        Returns:
            True 表示写入成功；Redis 不可用或序列化/写入异常时返回 False。
        """
        # Redis 不可用时直接降级为“不缓存”，不抛异常
        if not is_redis_available():
            return False
        try:
            # 优先使用调用方指定的 TTL，否则取全局默认 TTL
            ex = ttl if ttl is not None else get_settings().redis_cache_ttl
            get_redis().set(
                # ensure_ascii=False 保留中文；default=str 兜底序列化 datetime 等对象
                self._key(key), json.dumps(value, ensure_ascii=False, default=str), ex=ex
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 写入缓存失败：%s", e)
            return False

    def get_cache(self, key: str):
        """读取缓存并反序列化为 Python 对象。

        Args:
            key: 缓存 key（不含前缀）。

        Returns:
            反序列化后的对象；key 不存在、Redis 不可用或解析失败时返回 None。
        """
        if not is_redis_available():
            return None
        try:
            raw = get_redis().get(self._key(key))
            # decode_responses=True 时拿到的是字符串，None 表示 key 不存在或已过期
            return json.loads(raw) if raw is not None else None
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 读取缓存失败：%s", e)
            return None

    def delete_cache(self, key: str) -> bool:
        """删除指定缓存。

        Args:
            key: 缓存 key（不含前缀）。

        Returns:
            True 表示删除命令执行成功；Redis 不可用或异常时返回 False。
        """
        if not is_redis_available():
            return False
        try:
            get_redis().delete(self._key(key))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 删除缓存失败：%s", e)
            return False
