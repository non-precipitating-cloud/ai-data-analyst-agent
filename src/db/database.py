"""数据库连接与 Session 管理（SQLAlchemy 2.0）。"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config.settings import get_settings
from src.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None
_db_available: bool | None = None


def _create_engine(url: str):
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["connect_args"] = {"connect_timeout": 3}
    return create_engine(url, **kwargs)


def get_engine():
    global _engine
    if _engine is None:
        _engine = _create_engine(get_settings().database_url)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def configure_engine(url: str) -> None:
    """显式配置数据库引擎（测试/程序化配置用）。"""
    global _engine, _session_factory
    _engine = _create_engine(url)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)


def reset_engine() -> None:
    """重置引擎与可用性缓存（测试隔离用）。"""
    global _engine, _session_factory, _db_available
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _db_available = None


@contextmanager
def session_scope():
    """短生命周期 session：正确 commit / rollback / close。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine=None) -> None:
    """创建数据库表（幂等，不删除已有数据）。"""
    eng = engine or get_engine()
    Base.metadata.create_all(eng)


def is_db_available() -> bool:
    """检测数据库是否可用（缓存结果）。"""
    global _db_available
    if _db_available is not None:
        return _db_available
    try:
        with session_scope() as s:
            s.execute(text("SELECT 1"))
        _db_available = True
    except Exception as e:  # noqa: BLE001
        logger.warning("数据库不可用（%s），业务持久化将跳过", e)
        _db_available = False
    return _db_available
