"""日志配置模块：为整个应用提供「控制台 + 滚动文件」两路日志输出。

- 控制台：输出 INFO 及以上级别到标准输出，前台交互或 docker logs 可实时查看；
- 文件：输出 DEBUG 及以上级别到 logs/agent.log，按大小自动滚动，
  便于事后排查完整的调试细节。

调用一次 setup_logging() 后，项目中任何 logging.getLogger(__name__)
取到的日志器都会继承根日志器上的统一配置。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.settings import get_settings

# 日志行格式：时间 | 级别（左对齐占 7 字符）| 日志器名 | 消息正文
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
# 时间戳格式：年-月-日 时:分:秒
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根日志器（幂等，可安全重复调用）。

    参数:
        level: 控制台输出的最低日志级别，默认为 INFO；
            文件处理器固定记录 DEBUG 及以上级别，不受此参数影响。
    """
    root = logging.getLogger()
    if root.handlers:  # 已配置过则直接返回，避免重复挂 handler 导致同一条日志打印多遍
        return

    settings = get_settings()
    # 确保日志目录存在：parents=True 连带创建父目录，exist_ok=True 时目录已存在也不报错
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path = settings.logs_dir / "agent.log"

    root.setLevel(level)

    # 控制台处理器：日志写到标准输出，终端与 docker logs 可实时看到
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))

    # 滚动文件处理器：单文件最大 5MB，超出后切分，最多保留 3 个历史文件（agent.log.1/.2/.3）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    # 文件记录更细的 DEBUG 级别，线上排查时能看到完整调试信息
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))

    # 两个处理器都挂到根日志器：每条日志会同时分发给控制台与文件
    root.addHandler(console)
    root.addHandler(file_handler)

    # 降低第三方库日志噪音：这些库默认 INFO 输出很多，统一抬到 WARNING 级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
