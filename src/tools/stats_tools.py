"""统计工具：描述统计、相关性、异常值检测。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.tools.file_tools import load_dataframe


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


@tool
def calculate_statistics(path: str, column: str) -> str:
    """计算指定数值列的描述统计：count/mean/std/min/25%/50%/75%/max。"""
    df = load_dataframe(path)
    if column not in df.columns:
        return f"错误：字段 '{column}' 不存在。可选字段: {list(df.columns)}"

    series = pd.to_numeric(df[column], errors="coerce")
    stats = {
        "column": column,
        "count": int(series.count()),
        "mean": _float(series.mean()),
        "std": _float(series.std()),
        "min": _float(series.min()),
        "25%": _float(series.quantile(0.25)),
        "50%": _float(series.quantile(0.50)),
        "75%": _float(series.quantile(0.75)),
        "max": _float(series.max()),
    }
    return _json_dumps(stats)


@tool
def calculate_correlation(path: str, column_x: str = "", column_y: str = "") -> str:
    """计算数值列之间的皮尔逊相关系数。

    同时提供 column_x 和 column_y 时返回两列相关系数；留空则返回所有数值列的相关系数矩阵。
    """
    df = load_dataframe(path)
    numeric = df.select_dtypes(include="number")

    if column_x and column_y:
        if column_x not in df.columns or column_y not in df.columns:
            return f"错误：字段不存在。可选数值字段: {list(numeric.columns)}"
        corr = float(df[column_x].corr(df[column_y]))
        strength = _strength(corr)
        return _json_dumps({
            "column_x": column_x,
            "column_y": column_y,
            "correlation": round(corr, 4),
            "strength": strength,
        })

    corr_matrix = numeric.corr().round(4)
    return _json_dumps({"correlation_matrix": corr_matrix.to_dict()})


@tool
def detect_outliers(path: str, column: str = "", method: str = "zscore") -> str:
    """检测数值列中的异常值。

    method: "zscore"（|z|>3）或 "iqr"（超出 1.5 倍 IQR）。
    column 留空时对所有数值列检测。返回异常值数量与代表性示例。
    """
    df = load_dataframe(path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if column:
        if column not in df.columns:
            return f"错误：字段 '{column}' 不存在。可选数值字段: {numeric_cols}"
        columns = [column]
    else:
        columns = numeric_cols

    if method not in ("zscore", "iqr"):
        return "错误：method 仅支持 'zscore' 或 'iqr'"

    result: dict[str, Any] = {"method": method, "columns": {}}
    total_outliers = 0
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        if method == "zscore":
            z = (s - s.mean()) / s.std()
            mask = z.abs() > 3
        else:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

        outlier_values = s[mask]
        total_outliers += int(mask.sum())
        result["columns"][col] = {
            "outlier_count": int(mask.sum()),
            "pct": round(float(mask.sum() / len(s) * 100), 2),
            "examples": [round(float(v), 2) for v in outlier_values.head(5)],
        }

    result["total_outliers"] = total_outliers
    return _json_dumps(result)


def _float(v: Any) -> float | None:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _strength(r: float) -> str:
    a = abs(r)
    if a >= 0.8:
        return "极强"
    if a >= 0.6:
        return "强"
    if a >= 0.4:
        return "中等"
    if a >= 0.2:
        return "弱"
    return "极弱/无"
