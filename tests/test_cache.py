"""Redis 会话 / 缓存测试（src.cache）。

覆盖两大组件：
- SessionStore：分析会话的创建、读取、更新、删除；
- Cache：通用键值缓存的写入、读取、删除；
并专门验证 Redis 不可用时的优雅降级（不抛异常，返回 None / False）。
测试全程使用 conftest 中的 FakeRedis 内存替身，无需真实 Redis 服务。
"""

from __future__ import annotations

from src.cache import Cache, SessionStore
from src.cache.redis_client import get_redis


def test_redis_client_created() -> None:
    """get_redis 应惰性返回一个客户端对象（获取阶段不要求真正建立连接）。"""
    # get_redis 返回客户端对象（惰性，不连接）
    assert get_redis() is not None


def test_session_create_get_update(fake_redis) -> None:
    """创建会话后可按 ID 原样读回；更新状态与步骤后应持久化并刷新 updated_at。"""
    store = SessionStore()
    # 构造一个关联任务 ID 和用户需求的会话
    store.create_session("s1", task_id=1, user_request="分析销售额")
    s = store.get_session("s1")
    assert s is not None
    assert s["session_id"] == "s1"
    assert s["task_id"] == 1
    assert s["user_request"] == "分析销售额"

    # 更新执行阶段与当前步骤号
    store.update_session("s1", current_status="planning", current_step=3)
    s2 = store.get_session("s1")
    assert s2["current_status"] == "planning"
    assert s2["current_step"] == 3
    # 更新时必须写入 updated_at 时间戳
    assert "updated_at" in s2


def test_session_delete(fake_redis) -> None:
    """删除会话后再次读取应返回 None。"""
    store = SessionStore()
    store.create_session("s1")
    assert store.get_session("s1") is not None
    store.delete_session("s1")
    assert store.get_session("s1") is None


def test_cache_set_get_delete(fake_redis) -> None:
    """通用缓存写入后可原样读回，删除后读不到。"""
    c = Cache()
    # 写入成功返回 True，且能读回相同的数据结构
    assert c.set_cache("k1", {"a": 1}) is True
    assert c.get_cache("k1") == {"a": 1}
    c.delete_cache("k1")
    assert c.get_cache("k1") is None


def test_redis_unavailable_degradation(monkeypatch) -> None:
    """Redis 不可用时各操作应优雅降级：会话写后读不到、缓存操作返回 False/None 且不崩溃。"""
    # 直接把模块级可用性标记置为 False、连接对象置空，模拟服务宕机
    monkeypatch.setattr("src.cache.redis_client._redis_available", False)
    monkeypatch.setattr("src.cache.redis_client._redis", None)

    store = SessionStore()
    # create_session 仍返回本地 dict（不崩溃），但读不到（降级）
    data = store.create_session("s1")
    assert data["session_id"] == "s1"
    assert store.get_session("s1") is None

    c = Cache()
    # 降级时：写返回 False、读返回 None、删除返回 False
    assert c.set_cache("k", 1) is False
    assert c.get_cache("k") is None
    assert c.delete_cache("k") is False
