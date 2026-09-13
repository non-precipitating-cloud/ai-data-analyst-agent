"""统计分析工具测试（src.tools.stats_tools）。

覆盖三个 LangChain 工具：
- calculate_statistics：单列描述性统计，以及列不存在时的错误提示；
- calculate_correlation：两列相关系数与全量相关矩阵；
- detect_outliers：zscore / iqr 两种异常检测方法及非法方法的报错。
"""

from __future__ import annotations

from tests.conftest import SALES_CSV
from src.tools.stats_tools import (
    calculate_correlation,
    calculate_statistics,
    detect_outliers,
)


def test_calculate_statistics() -> None:
    """对 sales 列做描述性统计，结果应包含 mean/std/min/max。"""
    r = calculate_statistics.invoke({"path": SALES_CSV, "column": "sales"})
    assert "mean" in r
    assert "std" in r
    assert "min" in r
    assert "max" in r


def test_calculate_statistics_missing_column() -> None:
    """列不存在时不应崩溃，而应在返回文本中提示该列名（不存在）。"""
    r = calculate_statistics.invoke({"path": SALES_CSV, "column": "不存在"})
    assert "不存在" in r


def test_calculate_correlation_two_cols() -> None:
    """指定 sales、profit 两列时应返回它们之间的 correlation 系数。"""
    r = calculate_correlation.invoke(
        {"path": SALES_CSV, "column_x": "sales", "column_y": "profit"}
    )
    assert "correlation" in r


def test_calculate_correlation_matrix() -> None:
    """不指定列时应返回所有数值列的 correlation_matrix 相关矩阵。"""
    r = calculate_correlation.invoke({"path": SALES_CSV})
    assert "correlation_matrix" in r


def test_detect_outliers_zscore() -> None:
    """zscore 方法检测 sales 异常值，结果应包含 outlier_count 字段。"""
    r = detect_outliers.invoke({"path": SALES_CSV, "column": "sales", "method": "zscore"})
    assert "outlier_count" in r


def test_detect_outliers_iqr() -> None:
    """iqr 方法检测 sales 异常值，结果同样应包含 outlier_count 字段。"""
    r = detect_outliers.invoke({"path": SALES_CSV, "column": "sales", "method": "iqr"})
    assert "outlier_count" in r


def test_detect_outliers_bad_method() -> None:
    """传入不支持的方法名时应返回“仅支持”开头的错误提示。"""
    r = detect_outliers.invoke({"path": SALES_CSV, "column": "sales", "method": "bad"})
    assert "仅支持" in r
