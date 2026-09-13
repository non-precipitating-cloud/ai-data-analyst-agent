"""统计工具：描述统计、皮尔逊相关性、异常值检测（Z-Score / IQR）。

输入输出契约：
- 输入：数据文件路径 path + 字段名（column / column_x / column_y），
  异常检测另支持 method（"zscore" 或 "iqr"）。
- 输出：三个 @tool 均返回 JSON 字符串；参数非法、字段不存在、数据为空或
  读取失败时返回**结构化错误文本**（见 src/tools/errors.py），而非抛异常，
  便于 Agent 依据错误类型自我纠正参数。
- 计算基于 pandas/numpy；非数值内容会被强转为数值（不可解析记为 NaN 后忽略）。
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.tools.errors import (
    KIND_DATA_EMPTY,
    KIND_INVALID_ARGUMENT,
    KIND_NOT_FOUND,
    KIND_UNSUPPORTED,
    describe_exception,
    tool_error,
)
from src.tools.file_tools import load_dataframe


def _json_dumps(obj: Any) -> str:
    """把统计结果序列化为中文友好的 JSON 字符串。

    参数：
        obj: 统计结果 dict（可能含 numpy 标量）。
    返回值：
        str: JSON 文本；无法原生序列化的对象用 str() 兜底。
    """
    return json.dumps(obj, ensure_ascii=False, default=str)


def _load(path: str) -> tuple[pd.DataFrame | None, str | None]:
    """统一加载数据集并把异常转成结构化错误文本。

    三个统计工具共用同一套「读文件 → 失败则返回错误文本」的处理，
    避免各自重复 try/except 且语义不一致。

    参数：
        path: 数据文件路径。
    返回：
        二元组 (DataFrame, 错误文本)：成功时错误文本为 None，失败时 DataFrame 为 None。
    """
    try:
        return load_dataframe(path), None
    except Exception as e:  # noqa: BLE001 —— 工具层不抛异常，统一转文本
        kind, retryable = describe_exception(e)
        return None, tool_error(
            kind,
            f"无法读取数据集 {path}：{type(e).__name__}: {e}",
            hint="请确认 path 指向 datasets/ 目录下真实存在且格式受支持的文件。",
            retryable=retryable,
        )


@tool
def calculate_statistics(path: str, column: str) -> str:
    """计算指定数值列的描述统计：count/mean/std/min/25%/50%/75%/max。

    参数：
        path: 数据文件路径。
        column: 待统计字段名。
    返回值：
        str: 描述统计的 JSON 字符串；字段不存在时返回结构化错误（含可选字段列表）。
    """
    df, err = _load(path)
    if err:
        return err
    assert df is not None  # _load 成功时必然有 DataFrame

    if column not in df.columns:
        return tool_error(
            KIND_NOT_FOUND,
            f"字段 '{column}' 不存在",
            options=list(df.columns),
            hint="请从上列字段中选一个已存在的列名后重试。",
        )

    # 强转数值：非数字内容变为 NaN，后续 count/mean 等会自动忽略
    series = pd.to_numeric(df[column], errors="coerce")
    if int(series.count()) == 0:
        # 列存在但没有任何可解析的数值：这是「数据为空」而非参数错误，
        # 明确区分可避免 LLM 反复改列名做无效重试
        return tool_error(
            KIND_DATA_EMPTY,
            f"字段 '{column}' 没有可参与统计的数值（全部为空或非数值）",
            options=df.select_dtypes(include="number").columns.tolist(),
            hint="请改选一个数值型字段，或先用 execute_python 清洗该列。",
            retryable=False,
        )

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

    同时提供 column_x 和 column_y 时返回两列相关系数；两者都留空则返回所有数值列的相关系数矩阵。

    参数：
        path: 数据文件路径。
        column_x: 第一个字段名（可空）。
        column_y: 第二个字段名（可空，须与 column_x 同时提供）。
    返回值：
        str: 两列模式返回 {column_x, column_y, correlation, strength}；
        留空模式返回 {correlation_matrix: 嵌套字典}；字段缺失时返回结构化错误。
    """
    df, err = _load(path)
    if err:
        return err
    assert df is not None

    # 仅取数值列参与相关性计算，字符串列会被 pandas 自动排除
    numeric = df.select_dtypes(include="number")

    # 只给了一个列名属于「参数没写全」：明确报错，而不是悄悄退化成全量相关矩阵
    # （旧实现会静默返回矩阵，LLM 拿到一个没预期的巨大结果，容易得出错误结论）
    if bool(column_x) != bool(column_y):
        return tool_error(
            KIND_INVALID_ARGUMENT,
            "column_x 与 column_y 必须同时提供或同时留空"
            f"（当前 column_x={column_x!r}, column_y={column_y!r}）",
            options=numeric.columns.tolist(),
            hint="要算两列相关性就两个都填；要看全量相关矩阵就两个都留空。",
        )

    if column_x and column_y:
        if column_x not in df.columns or column_y not in df.columns:
            missing = [c for c in (column_x, column_y) if c not in df.columns]
            return tool_error(
                KIND_NOT_FOUND,
                f"字段不存在: {missing}",
                options=list(df.columns),
                hint="请从上列字段中选择已存在的数值列后重试。",
            )
        if numeric.shape[1] < 2:
            return tool_error(
                KIND_DATA_EMPTY,
                "数据集中的数值列少于 2 列，无法计算相关性",
                options=numeric.columns.tolist(),
                retryable=False,
            )
        # 皮尔逊相关系数 ∈ [-1, 1]：绝对值越大线性关系越强，符号表示方向
        corr = float(df[column_x].corr(df[column_y]))
        if pd.isna(corr):
            # 任一列常量（标准差为 0）时相关系数无定义
            return tool_error(
                KIND_DATA_EMPTY,
                f"'{column_x}' 与 '{column_y}' 的相关系数无定义"
                "（通常是其中一列取值恒定，标准差为 0）",
                hint="请改用其他有波动的数值列，或先检查该列取值分布。",
                retryable=False,
            )
        return _json_dumps({
            "column_x": column_x,
            "column_y": column_y,
            "correlation": round(corr, 4),
            "strength": _strength(corr),
        })

    if numeric.shape[1] < 2:
        return tool_error(
            KIND_DATA_EMPTY,
            "数据集中的数值列少于 2 列，无法计算相关矩阵",
            options=numeric.columns.tolist(),
            retryable=False,
        )

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
        的 JSON 字符串；字段/方法非法或数据为空时返回结构化错误。
    """
    df, err = _load(path)
    if err:
        return err
    assert df is not None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # 先校验 method：非法取值属于参数错误，应在扫描数据前就快速失败
    if method not in ("zscore", "iqr"):
        return tool_error(
            KIND_UNSUPPORTED,
            f"method 仅支持 'zscore' 或 'iqr'，收到 {method!r}",
            options=["zscore", "iqr"],
            hint="请把 method 改成两者之一后重试。",
        )

    if column:
        if column not in df.columns:
            return tool_error(
                KIND_NOT_FOUND,
                f"字段 '{column}' 不存在",
                options=list(df.columns),
                hint="请从上列字段中选择已存在的列名后重试。",
            )
        columns = [column]
    else:
        # 未指定列时对全部数值列逐一检测
        columns = numeric_cols

    if not columns:
        return tool_error(
            KIND_DATA_EMPTY,
            "数据集中没有可检测的数值列",
            options=list(df.columns),
            hint="请用 column 指定一个数值列，或先用 execute_python 做类型转换。",
            retryable=False,
        )

    result: dict[str, Any] = {"method": method, "columns": {}}
    skipped: list[str] = []  # 记录「列存在但无有效数值」的字段，避免被静默忽略
    total_outliers = 0
    for col in columns:
        # 强转数值并丢弃 NaN，避免污染均值/标准差
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            # 旧实现直接 continue，导致该列在结果里凭空消失，LLM 会以为已检测过
            skipped.append(col)
            continue
        if method == "zscore":
            # 标准化得分 z = (x - 均值) / 标准差；|z| > 3 视为离群（约落在 99.7% 之外）
            std = s.std()
            if std == 0 or pd.isna(std):
                # 常量列：z-score 无定义（分母为 0），标记跳过而不是产出 NaN
                skipped.append(col)
                continue
            z = (s - s.mean()) / std
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

    # 被跳过的列如实回传，而不是让它们从结果里消失
    if skipped:
        result["skipped_columns"] = skipped
    result["total_outliers"] = total_outliers

    if not result["columns"]:
        return tool_error(
            KIND_DATA_EMPTY,
            f"所有待检测字段都没有可用的数值数据（已跳过 {skipped}）",
            options=numeric_cols,
            hint="请确认字段类型或先用 execute_python 清洗数据。",
            retryable=False,
        )
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
