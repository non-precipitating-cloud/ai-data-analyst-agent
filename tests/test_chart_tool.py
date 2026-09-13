"""图表生成工具测试（src.tools.chart_tool）。

验证 generate_chart_file 能基于样例 CSV 生成柱状图、散点图等图片文件；
非法图表类型应抛 ValueError；同时锁定受支持的图表类型集合，防止误增删。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import SALES_CSV
from src.tools.chart_tool import SUPPORTED_TYPES, generate_chart_file


def test_generate_bar_chart() -> None:
    """以 region 为分类轴、sales 为数值轴生成柱状图，输出文件应真实落盘。"""
    out = generate_chart_file(SALES_CSV, "bar", "region", "sales", "region sales")
    assert Path(out).exists()
    Path(out).unlink()  # 清理测试产物


def test_generate_scatter_chart() -> None:
    """以 sales、profit 两列生成散点图，输出文件应真实落盘。"""
    out = generate_chart_file(SALES_CSV, "scatter", "sales", "profit")
    assert Path(out).exists()
    Path(out).unlink()


def test_unsupported_type() -> None:
    """传入不支持的图表类型（pie）应抛 ValueError。"""
    with pytest.raises(ValueError):
        generate_chart_file(SALES_CSV, "pie", "sales")


def test_supported_types_complete() -> None:
    """受支持类型应恰好为这 5 种，构成对外契约。"""
    assert SUPPORTED_TYPES == {"bar", "line", "scatter", "hist", "box"}
