"""统计工具：描述统计、皮尔逊相关性、异常值检测（Z-Score / IQR）。

输入输出契约：
- 输入：数据文件路径 path + 字段名（column / column_x / column_y），
  异常检测另支持 method（"zscore" 或 "iqr"）。
- 输出：三个 @tool 均返回 JSON 字符串；字段不存在或方法非法时返回中文错误提示
  （普通字符串而非异常），便于 Agent 自我纠正参数。
- 计算基于 pandas/numpy；非数值内容会被强转为数值（不可解析记为 NaN 后忽略）。
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.tools.file_tools import load_dataframe


def _json_dumps(obj: Any) -> str:
    """把统计结果序列化为中文友好的 JSON 字符串。

    参数：
        obj: 统计结果 dict（可能含 numpy 标量）。
    返回值：
        str: JSON 文本；无法原生序列化的对象用 str() 兜底。
    """
    return json.dumps(obj, ensure_ascii=False, default=str)


@tool
def calculate_statistics(path: str, column: str) -> str:
    """计算指定数值列的描述统计：count/mean/std/min/25%/50%/75%/max。

    参数：
        path: 数据文件路径。
        column: 待统计字段名。
    返回值：
        str: 描述统计的 JSON 字符串；字段不存在时返回错误提示（含可选字段列表）。
    """
    df = load_dataframe(path)
    if column not in df.columns:
        return f"错误：字段 '{column}' 不存在。可选字段: {list(df.columns)}"

    # 强转数值：非数字内容变为 NaN，后续 count/mean 等会自动忽略
    series = pd.to_numeric(df[column], errors="coerce")
    stats = {
        # 非缺失值计数（NaN 不计入）
        "column": column,
        "count": int(series.count()),
        "mean": _float(series.mean()),
        "std": _float(series.std()),
        "min": _float(series.min()),
        # 三个分位数：下四分位/中位数/上四分位
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

    参数：
        path: 数据文件路径。
        column_x: 第一个字段名（可空）。
        column_y: 第二个字段名（可空）。
    返回值：
        str: 两列模式返回 {column_x, column_y, correlation, strength}；
        留空模式返回 {correlation_matrix: 嵌套字典}；字段缺失时返回错误提示。
    """
    df = load_dataframe(path)
    # 仅取数值列参与相关性计算，字符串列会被 pandas 自动排除
    numeric = df.select_dtypes(include="number")

    if column_x and column_y:
        if column_x not in df.columns or column_y not in df.columns:
            return f"错误：字段不存在。可选数值字段: {list(numeric.columns)}"
        # 皮尔逊相关系数 ∈ [-1, 1]：绝对值越大线性关系越强，符号表示方向
        corr = float(df[column_x].corr(df[column_y]))
        strength = _strength(corr)
        return _json_dumps({
            "column_x": column_x,
            "column_y": column_y,
            "correlation": round(corr, 4),
            "strength": strength,
        })

    # 未指定列：一次性给出全部数值列两两之间的相关系数矩阵
    corr_matrix = numeric.corr().round(4)
    return _json_dumps({"correlation_matrix": corr_matrix.to_dict()})


@tool
def detect_outliers(path: str, column: str = "", method: str = "zscore") -> str:
    """检测数值列中的异常值。

    method: "zscore"（|z|>3）或 "iqr"（超出 1.5 倍 IQR）。
    column 留空时对所有数值列检测。返回异常值数量与代表性示例。

    参数：
        path: 数据文件路径。
        column: 目标字段；留空表示扫描全部数值列。
        method: 检测方法，zscore（3σ 准则）或 iqr（箱线图四分位距准则）。
    返回值：
        str: {method, columns: {字段: {outlier_count, pct, examples}}, total_outliers}
        的 JSON 字符串；字段/方法非法时返回错误提示。
    """
    df = load_dataframe(path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if column:
        if column not in df.columns:
            return f"错误：字段 '{column}' 不存在。可选数值字段: {numeric_cols}"
        columns = [column]
    else:
        # 未指定列时对全部数值列逐一检测
        columns = numeric_cols

    if method not in ("zscore", "iqr"):
        return "错误：method 仅支持 'zscore' 或 'iqr'"

    result: dict[str, Any] = {"method": method, "columns": {}}
    total_outliers = 0
    for col in columns:
        # 强转数值并丢弃 NaN，避免污染均值/标准差
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        if method == "zscore":
            # 标准化得分 z = (x - 均值) / 标准差；|z| > 3 视为离群（约落在 99.7% 之外）
            z = (s - s.mean()) / s.std()
            mask = z.abs() > 3
        else:
            # IQR 法：落在 [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 区间之外即异常，对偏态分布更稳健
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

        outlier_values = s[mask]
        total_outliers += int(mask.sum())
        result["columns"][col] = {
            "outlier_count": int(mask.sum()),
            # 异常占比：帮助判断是零星脏数据还是系统性问题
            "pct": round(float(mask.sum() / len(s) * 100), 2),
            # 只给前 5 个示例值，避免输出过长
            "examples": [round(float(v), 2) for v in outlier_values.head(5)],
        }

    result["total_outliers"] = total_outliers
    return _json_dumps(result)


def _float(v: Any) -> float | None:
    """把统计标量安全地转为保留 4 位小数的 float。

    参数：
        v: pandas/numpy 标量（如 np.float64），可能为 NaN 或非法值。
    返回值：
        float | None: 四舍五入后的 float；无法转换（如全 NaN 列）时返回 None。
    """
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _strength(r: float) -> str:
    """按相关系数绝对值给出中文强度档位解读。

    参数：
        r: 皮尔逊相关系数（-1 ~ 1）。
    返回值：
        str: "极强" / "强" / "中等" / "弱" / "极弱/无"。
    """
    # 相关强度只看绝对值，正负号代表相关方向而不代表强弱
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
