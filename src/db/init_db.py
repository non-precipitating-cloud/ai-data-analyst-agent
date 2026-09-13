"""数据库初始化脚本（独立可执行）。

用法：
    python -m src.db.init_db

创建业务表（幂等，不删除已有数据，不影响 pgvector extension）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径，兼容「直接按文件路径运行本脚本」的场景
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import get_settings  # noqa: E402
from src.db.database import init_db  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402


def main() -> None:
    """脚本入口：初始化日志并创建全部业务表。

    建表失败时打印异常类型与信息，并以退出码 1 结束进程，
    便于容器编排或 Shell 脚本感知初始化失败。
    """
    setup_logging()
    try:
        init_db()
    except Exception as e:  # noqa: BLE001
        print(f"数据库初始化失败：{type(e).__name__}: {e}")
        sys.exit(1)
    # 成功后回显已创建的表清单与实际使用的连接串，方便人工核对
    print("数据库表创建成功（datasets / analysis_tasks / agent_runs / tool_calls / analysis_results / reports）")
    print(f"DATABASE_URL: {get_settings().database_url}")


if __name__ == "__main__":
    main()
