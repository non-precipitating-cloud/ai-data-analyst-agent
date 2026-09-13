"""日志配置模块：为整个应用提供「控制台 + 滚动文件」两路日志输出。

- 控制台：输出 INFO 及以上级别到标准输出，前台交互或 docker logs 可实时查看；
- 文件：输出 DEBUG 及以上级别到 logs/agent.log，按大小自动滚动，
  便于事后排查完整的调试细节。

调用一次 setup_logging() 后，项目中任何 logging.getLogger(__name__)
取到的日志器都会继承根日志器上的统一配置。

本模块同时负责**终端输出编码兜底**（见 configure_stdio）：CLI 会打印
✓/✗/⚠ 一类符号，而 Windows 默认控制台编码（cp936/GBK）无法表示它们，
不处理会直接抛 UnicodeEncodeError 把程序打挂。
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


def configure_stdio() -> None:
    """让标准输出/错误在任意终端编码下都不会因字符不可表示而崩溃。

    问题背景：Windows 上 Python 默认按控制台代码页（通常是 cp936）编码输出，
    而本项目 CLI 会打印 ``✓``/``✗``/``⚠`` 等符号——它们在 GBK 里没有对应
    字符，于是 ``print`` 直接抛 UnicodeEncodeError。

    处理策略刻意保守：**保留终端原有编码**，只把错误处理改成 ``replace``。
    这样中文仍然正常显示（GBK 覆盖中文），个别符号降级为 ``?``，
    既不会崩溃，也不会因为在 GBK 控制台上强切 UTF-8 而让中文变成乱码。

    兼容性：``reconfigure`` 是 Python 3.7+ 的流接口；被重定向到管道、
    或已被测试框架替换过的流可能没有该方法，此时静默跳过。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            # 流不可重配置（如已被关闭或替换）时忽略：输出编码问题是尽力而为，
            # 绝不能因为兜底本身失败而影响主流程
            pass


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根日志器（幂等，可安全重复调用）。

    参数:
        level: 控制台输出的最低日志级别，默认为 INFO；
            文件处理器固定记录 DEBUG 及以上级别，不受此参数影响。
    """
    # 先兜底终端编码，避免后续任何 print/日志因字符不可表示而崩溃
    configure_stdio()

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
