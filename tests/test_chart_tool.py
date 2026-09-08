"""图表工具测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import SALES_CSV
from src.tools.chart_tool import SUPPORTED_TYPES, generate_chart_file


def test_generate_bar_chart() -> None:
    out = generate_chart_file(SALES_CSV, "bar", "region", "sales", "region sales")
    assert Path(out).exists()
    Path(out).unlink()  # 清理测试产物


def test_generate_scatter_chart() -> None:
    out = generate_chart_file(SALES_CSV, "scatter", "sales", "profit")
    assert Path(out).exists()
    Path(out).unlink()


def test_unsupported_type() -> None:
    with pytest.raises(ValueError):
        generate_chart_file(SALES_CSV, "pie", "sales")


def test_supported_types_complete() -> None:
    assert SUPPORTED_TYPES == {"bar", "line", "scatter", "hist", "box"}
