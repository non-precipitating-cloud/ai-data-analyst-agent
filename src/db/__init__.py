"""数据库持久化包：汇总导出引擎管理函数与 Repository 数据访问对象。

业务层只需从 ``src.db`` 导入所需能力，不必分别引用 database / repository
等子模块，例如::

    from src.db import Repository, session_scope, init_db
"""

from src.db.database import (
    configure_engine,
    get_engine,
    init_db,
    is_db_available,
    reset_engine,
    session_scope,
)
# Repository：封装各张表增删改查的数据访问对象
from src.db.repository import Repository

# 公开 API 白名单：兼作包接口文档，并约束 from src.db import * 的导出范围
__all__ = [
    "Repository",
    "init_db",
    "is_db_available",
    "session_scope",
    "get_engine",
    "configure_engine",
    "reset_engine",
]
