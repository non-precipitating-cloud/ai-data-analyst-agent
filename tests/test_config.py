"""配置测试。"""

from __future__ import annotations

from src.config.settings import get_settings


def test_settings_loads() -> None:
    s = get_settings()
    assert s.max_steps > 0
    assert s.python_timeout_seconds > 0


def test_dirs_exist() -> None:
    s = get_settings()
    assert s.reports_dir.exists()
    assert s.charts_dir.exists()
    assert s.datasets_dir.exists()


def test_resolve_dataset_path() -> None:
    s = get_settings()
    p = s.resolve_dataset_path("datasets/sales.csv")
    assert p.exists()
