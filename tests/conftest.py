"""pytest 共享夹具。"""

from __future__ import annotations

import pytest

from src.config.settings import get_settings

SALES_CSV = "datasets/sales.csv"


@pytest.fixture(scope="session", autouse=True)
def _ensure_dirs() -> None:
    get_settings()


@pytest.fixture
def sqlite_db(tmp_path):
    """用 SQLite 测试数据库（每个测试隔离），用于 ORM / repository / 持久化测试。"""
    from src.db.database import configure_engine, init_db, reset_engine

    reset_engine()
    configure_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db()
    yield
    reset_engine()


class FakeRedis:
    """内存版 Redis，用于测试 Session / Cache 逻辑（不依赖真实 Redis）。"""

    def __init__(self) -> None:
        self.store: dict = {}

    def ping(self) -> bool:
        return True

    def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)
        return len(keys)


@pytest.fixture
def fake_redis(monkeypatch):
    """用 FakeRedis 替换真实 Redis，测试会话/缓存逻辑（含 TTL 忽略）。"""
    from src.cache import redis_client

    redis_client.reset_redis()
    fake = FakeRedis()
    monkeypatch.setattr(redis_client, "_redis", fake)
    monkeypatch.setattr(redis_client, "_redis_available", None)
    yield fake
    redis_client.reset_redis()
