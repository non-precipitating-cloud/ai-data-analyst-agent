"""数据库初始化脚本。

用法：
    python -m src.db.init_db

创建业务表（幂等，不删除已有数据，不影响 pgvector extension）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import get_settings  # noqa: E402
from src.db.database import init_db  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402


def main() -> None:
    setup_logging()
    try:
        init_db()
    except Exception as e:  # noqa: BLE001
        print(f"数据库初始化失败：{type(e).__name__}: {e}")
        sys.exit(1)
    print("数据库表创建成功（datasets / analysis_tasks / agent_runs / tool_calls / analysis_results / reports）")
    print(f"DATABASE_URL: {get_settings().database_url}")


if __name__ == "__main__":
    main()
