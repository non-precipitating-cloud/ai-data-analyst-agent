"""pytest 全局共享夹具（fixture）模块。

本文件不放具体测试用例，只为 tests/ 下所有测试提供公共依赖：
- _ensure_dirs：会话级自动夹具，触发配置加载并确保输出目录存在；
- sqlite_db：每个用例独立的临时 SQLite 数据库，供 ORM / Repository / 持久化测试使用；
- FakeRedis / fake_redis：用内存字典模拟 Redis，隔离会话与缓存逻辑，无需真实 Redis 服务。
"""

from __future__ import annotations

import pytest

from src.config.settings import get_settings

# 测试统一使用的样例数据集相对路径（UTF-8 BOM 的中文销售数据，共 2160 行 12 列）
SALES_CSV = "datasets/sales.csv"


@pytest.fixture(scope="session", autouse=True)
def _ensure_dirs() -> None:
    """会话级自动夹具：尽早加载一次配置，借此确保 reports/charts/datasets 等目录已创建。"""
    get_settings()


@pytest.fixture
def sqlite_db(tmp_path):
    """用 SQLite 测试数据库（每个测试隔离），用于 ORM / repository / 持久化测试。"""
    from src.db.database import configure_engine, init_db, reset_engine

    # 先清空可能存在的全局引擎，避免用例之间串数据
    reset_engine()
    # 把全局数据库引擎切换到 tmp_path 下的独立 SQLite 文件
    configure_engine(f"sqlite:///{tmp_path / 'test.db'}")
    # 按 ORM 模型创建全部业务表
    init_db()
    yield
    # 用例结束后复位引擎，恢复全局初始状态
    reset_engine()


class FakeRedis:
    """内存版 Redis，用于测试 Session / Cache 逻辑（不依赖真实 Redis）。"""

    def __init__(self) -> None:
        # 用普通字典充当 Redis 的键值存储
        self.store: dict = {}

    def ping(self) -> bool:
        """模拟连通性检查：始终返回可用。"""
        return True

    def set(self, key, value, ex=None):
        """模拟 SET：忽略过期时间 ex，直接写入字典并返回成功。"""
        self.store[key] = value
        return True

    def get(self, key):
        """模拟 GET：键不存在时返回 None（与 redis-py 行为一致）。"""
        return self.store.get(key)

    def delete(self, *keys):
        """模拟 DELETE：批量移除键，返回被删除的键数量。"""
        for k in keys:
            self.store.pop(k, None)
        return len(keys)


@pytest.fixture
def fake_redis(monkeypatch):
    """用 FakeRedis 替换真实 Redis，测试会话/缓存逻辑（含 TTL 忽略）。"""
    from src.cache import redis_client

    # 重置模块内单例状态，保证从干净状态开始
    redis_client.reset_redis()
    fake = FakeRedis()
    # 把真实连接对象与可用性标记替换为内存替身
    monkeypatch.setattr(redis_client, "_redis", fake)
    monkeypatch.setattr(redis_client, "_redis_available", None)
    yield fake
    # 用例结束复位，避免污染其他测试
    redis_client.reset_redis()
