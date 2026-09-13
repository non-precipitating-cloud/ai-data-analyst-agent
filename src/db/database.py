"""数据库连接与 Session 管理模块（基于 SQLAlchemy 2.0）。

职责：
- 懒加载并缓存进程级全局 Engine（连接池）与 Session 工厂；
- 提供短生命周期的 session_scope() 上下文管理器，自动 commit/rollback/close；
- 提供建表 init_db() 与可用性探测 is_db_available()，支撑上层优雅降级。

模块级变量采用单例模式，整个进程共享同一个数据库连接池。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config.settings import get_settings
from src.db.models import Base

logger = logging.getLogger(__name__)

# 全局引擎（连接池）；None 表示尚未创建，首次使用时懒加载
_engine = None
# 全局 Session 工厂，绑定到 _engine
_session_factory = None
# 数据库可用性缓存：None=尚未探测，True/False=探测结果，避免每次调用都 SELECT 1
_db_available: bool | None = None


def _create_engine(url: str):
    """根据数据库连接串创建 SQLAlchemy 引擎。

    参数:
        url: 数据库连接串，如 postgresql+psycopg://user:pwd@host:5432/db。

    返回:
        配置好连接参数的 Engine 实例。
    """
    kwargs: dict = {
        # 取连接前先发一次 ping 探活，自动丢弃连接池中已断开的死连接
        "pool_pre_ping": True,
    }
    if url.startswith("sqlite"):
        # SQLite 专属参数：关闭「连接必须与创建线程相同」的限制，便于多线程复用
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL 等网络数据库：建连超时 3 秒，数据库宕机时不会长时间卡死
        kwargs["connect_args"] = {"connect_timeout": 3}
    return create_engine(url, **kwargs)


def get_engine():
    """获取全局引擎单例：首次调用时按配置中的 DATABASE_URL 懒加载创建。"""
    global _engine
    if _engine is None:
        _engine = _create_engine(get_settings().database_url)
    return _engine


def get_session_factory():
    """获取全局 Session 工厂单例。

    expire_on_commit=False 表示 commit 后对象属性不过期，
    因此 Repository 在 Session 关闭后仍能读取已查出的对象属性（脱离会话的游离对象）。
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def configure_engine(url: str) -> None:
    """显式配置数据库引擎（测试/程序化配置用）。

    参数:
        url: 要强制使用的数据库连接串（如测试用的 sqlite 内存库）。
    """
    global _engine, _session_factory
    _engine = _create_engine(url)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)


def reset_engine() -> None:
    """重置引擎与可用性缓存（测试隔离用）。

    释放连接池中的全部连接并清空模块级单例，下次调用会重新创建，
    确保不同测试用例之间互不污染。
    """
    global _engine, _session_factory, _db_available
    if _engine is not None:
        _engine.dispose()  # 关闭连接池持有的全部底层连接
    _engine = None
    _session_factory = None
    _db_available = None


@contextmanager
def session_scope():
    """短生命周期 Session 上下文管理器。

    用法::

        with session_scope() as s:
            s.add(obj)

    with 块正常退出时自动 commit；抛出异常时自动 rollback 并继续向上抛；
    无论成功与否最终都 close，保证数据库连接归还连接池。
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()  # 正常结束：提交事务
    except Exception:
        session.rollback()  # 出现异常：撤销本事务内所有未提交改动
        raise
    finally:
        session.close()  # 关闭 Session，连接归还连接池


# 轻量迁移语句：create_all 只会**新建缺失的表**，不会给已存在的表补列。
# 对于新增字段（如 tool_calls.duration_ms），老库升级后会因缺列而写库失败，
# 因此这里显式补一次。语句本身幂等（IF NOT EXISTS），可安全重复执行。
_LIGHT_MIGRATIONS = (
    "ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS duration_ms INTEGER DEFAULT 0",
    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS llm_call_count INTEGER DEFAULT 0",
    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS llm_failed_count INTEGER DEFAULT 0",
    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS total_tokens INTEGER",
    "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS duration_ms INTEGER DEFAULT 0",
)


def _apply_light_migrations(eng) -> None:
    """为已存在的表补齐新增列（仅 PostgreSQL；SQLite 由 create_all 全新建表）。

    说明：本项目没有引入 Alembic 这类完整迁移框架——那对当前的规模属于过度
    设计。这里只处理「加列」这一种向后兼容的变更；一旦出现改类型、拆表这类
    破坏性变更，应改用正式迁移工具。

    参数:
        eng: 目标数据库引擎。
    """
    # SQLite 等方言不支持 ADD COLUMN IF NOT EXISTS；测试库每次都是全新建表，
    # 因此这里只对 PostgreSQL 执行，避免制造方言兼容问题
    if eng.dialect.name != "postgresql":
        return
    try:
        with eng.begin() as conn:
            for stmt in _LIGHT_MIGRATIONS:
                conn.execute(text(stmt))
    except Exception as e:  # noqa: BLE001 —— 迁移失败不应阻断启动，交由后续写入报错暴露
        logger.warning("轻量迁移执行失败（可能是权限不足）：%s", e)


def init_db(engine=None) -> None:
    """创建数据库表并补齐新增列（幂等，不删除已有数据）。

    参数:
        engine: 可选的目标引擎；不传则使用全局引擎。
            底层 create_all 只创建缺失的表，已存在的表与数据保持原样。
    """
    eng = engine or get_engine()
    Base.metadata.create_all(eng)
    _apply_light_migrations(eng)


def is_db_available() -> bool:
    """检测数据库是否可用（结果缓存到模块级变量）。

    返回:
        True 表示可连通并能执行简单查询；False 表示不可用。
        调用方可据此跳过持久化，保证 Agent 核心分析流程不依赖数据库。
    """
    global _db_available
    if _db_available is not None:
        return _db_available  # 命中缓存，不再重复探测
    try:
        with session_scope() as s:
            s.execute(text("SELECT 1"))  # 最简单的连通性校验 SQL
        _db_available = True
    except Exception as e:  # noqa: BLE001
        # 任何异常都视为不可用：仅告警并降级，不向上抛出中断主流程
        logger.warning("数据库不可用（%s），业务持久化将跳过", e)
        _db_available = False
    return _db_available
