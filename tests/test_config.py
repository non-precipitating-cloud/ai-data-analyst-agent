"""配置加载测试（src.config.settings）。

验证全局配置能正常读取、关键参数（最大步数、Python 超时）为正数，
reports / charts / datasets 目录真实存在，且数据集相对路径可被解析到真实文件。
"""

from __future__ import annotations

from src.config.settings import get_settings


def test_settings_loads() -> None:
    """配置应成功加载，且最大步数与沙箱超时时间均为正数。"""
    s = get_settings()
    assert s.max_steps > 0
    assert s.python_timeout_seconds > 0


def test_dirs_exist() -> None:
    """配置中声明的报告、图表、数据集目录都应在磁盘上存在。"""
    s = get_settings()
    assert s.reports_dir.exists()
    assert s.charts_dir.exists()
    assert s.datasets_dir.exists()


def test_resolve_dataset_path() -> None:
    """相对路径 datasets/sales.csv 经解析后应指向真实存在的文件。"""
    s = get_settings()
    p = s.resolve_dataset_path("datasets/sales.csv")
    assert p.exists()
