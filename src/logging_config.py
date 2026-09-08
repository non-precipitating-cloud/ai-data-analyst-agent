"""日志配置：控制台 + 文件。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.settings import get_settings

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根日志器（幂等）。"""
    root = logging.getLogger()
    if root.handlers:  # 已配置过则跳过
        return

    settings = get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path = settings.logs_dir / "agent.log"

    root.setLevel(level)

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))

    # 文件（滚动）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))

    root.addHandler(console)
    root.addHandler(file_handler)

    # 降低第三方库日志噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
