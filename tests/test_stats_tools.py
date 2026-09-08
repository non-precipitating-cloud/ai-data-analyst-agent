"""统计工具测试。"""

from __future__ import annotations

from tests.conftest import SALES_CSV
from src.tools.stats_tools import (
    calculate_correlation,
    calculate_statistics,
    detect_outliers,
)


def test_calculate_statistics() -> None:
    r = calculate_statistics.invoke({"path": SALES_CSV, "column": "sales"})
    assert "mean" in r
    assert "std" in r
    assert "min" in r
    assert "max" in r


def test_calculate_statistics_missing_column() -> None:
    r = calculate_statistics.invoke({"path": SALES_CSV, "column": "不存在"})
    assert "不存在" in r


def test_calculate_correlation_two_cols() -> None:
    r = calculate_correlation.invoke(
        {"path": SALES_CSV, "column_x": "sales", "column_y": "profit"}
    )
    assert "correlation" in r


def test_calculate_correlation_matrix() -> None:
    r = calculate_correlation.invoke({"path": SALES_CSV})
    assert "correlation_matrix" in r


def test_detect_outliers_zscore() -> None:
    r = detect_outliers.invoke({"path": SALES_CSV, "column": "sales", "method": "zscore"})
    assert "outlier_count" in r


def test_detect_outliers_iqr() -> None:
    r = detect_outliers.invoke({"path": SALES_CSV, "column": "sales", "method": "iqr"})
    assert "outlier_count" in r


def test_detect_outliers_bad_method() -> None:
    r = detect_outliers.invoke({"path": SALES_CSV, "column": "sales", "method": "bad"})
    assert "仅支持" in r
