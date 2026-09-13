"""Redis 会话 / 缓存模块。

职责：
- 基于 Redis 提供两类能力：带 TTL 的通用 JSON 缓存（Cache）、
  Agent 会话状态的持久化存取（SessionStore）。
- 通过 redis_client 维护全局唯一的 Redis 连接，并做可用性检测；
  Redis 不可用时上层自动降级（缓存失效/会话不落库），不阻断主流程。
"""

# 通用 TTL 缓存
from src.cache.cache import Cache
# 全局 Redis 客户端单例与可用性探测（供本模块内部与外部复用）
from src.cache.redis_client import get_redis, is_redis_available, reset_redis
# Agent 会话状态存储
from src.cache.session_store import SessionStore

__all__ = [
    "Cache",
    "SessionStore",
    "get_redis",
    "is_redis_available",
    "reset_redis",
]
