"""文件工具：读取 CSV / Excel / JSON 数据集并返回概览、schema、画像。

输入输出契约：
- 输入：数据文件路径（经 settings.resolve_dataset_path 做工作区边界的安全校验，
  禁止通过 ../ 等方式越界访问 workspace 之外的文件）。
- load_dataframe() 返回 pandas.DataFrame，供其他工具（统计/图表/Python 沙箱前导代码）复用。
- 三个 @tool 对外返回 JSON 字符串（ensure_ascii=False，中文不转义）：
  read_dataset → 行/列数 + 字段名 + 前 5 行样例；
  inspect_schema → 字段名与 dtype；
  profile_dataset → 完整画像（缺失值、唯一值、数值列分位数统计、前 3 行样例）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.config.settings import get_settings
from src.tools.errors import describe_exception, tool_error


def _native(value: Any) -> Any:
    """将 numpy 标量转换为原生 Python 类型，保证 JSON 可序列化。

    参数：
        value: describe() 等 pandas/numpy 计算结果中的标量（如 np.float64）。
    返回值：
        Any: 调用 .item() 后的原生 int/float；无该方法时原样返回。
    """
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_dumps(obj: Any) -> str:
    """把工具结果序列化为 JSON 字符串。

    参数：
        obj: 待序列化的 dict/list（可能含 numpy 类型、Timestamp 等）。
    返回值：
        str: UTF-8 友好的 JSON 文本；无法原生序列化的对象用 str() 兜底。
    """
    # ensure_ascii=False 保留中文可读性；default=str 兜底时间戳等特殊类型
    return json.dumps(obj, ensure_ascii=False, default=str)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """根据扩展名加载 DataFrame（csv / xlsx / xls / json）。

    参数：
        path: 数据文件相对路径（相对 workspace/datasets）或绝对路径。
    返回值：
        pd.DataFrame: 加载后的数据表。
    异常：
        FileNotFoundError: 安全校验后的路径不存在。
        ValueError: 文件扩展名不在支持列表内，或文件超过体积上限。
    """
    settings = get_settings()
    # 路径安全校验：解析后必须落在允许的数据目录内，阻断目录穿越
    p = settings.resolve_dataset_path(path)

    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

    # 体积上限：pandas 会把整个文件读进内存，超大文件（如几 GB CSV）
    # 会直接拖垮进程。这里在读取前快速失败，给出可操作的提示，
    # 而不是让 Agent 卡在工具调用上直到超时。
    size = p.stat().st_size
    if size > settings.max_dataset_bytes:
        raise ValueError(
            f"数据文件过大（{size / 1024 / 1024:.1f} MB，上限 "
            f"{settings.max_dataset_bytes / 1024 / 1024:.0f} MB）。"
            "请先抽样或裁剪后再分析。"
        )

    suffix = p.suffix.lower()
    if suffix == ".csv":
        # utf-8-sig 显式按 UTF-8 读取，并正确剥离 BOM（EF BB BF），
        # 避免在 Windows 中文环境下被默认按 GBK 解码成乱码。
        return pd.read_csv(p, encoding="utf-8-sig")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"不支持的文件类型: {suffix}（支持 .csv / .xlsx / .json）")


def _read_dataset(path: str) -> dict:
    """读取数据集并返回轻量概览（底层 dict 版本，供节点/测试直接使用）。

    参数：
        path: 数据文件路径。
    返回值：
        dict: {path 解析后的绝对路径, num_rows, num_columns, columns, head(前5行记录)}。
    """
    df = load_dataframe(path)
    return {
        "path": str(get_settings().resolve_dataset_path(path)),
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        # orient="records" → 每行一个 dict，方便 LLM 直观理解数据样貌
        "head": df.head(5).to_dict(orient="records"),
    }


def _inspect_schema(path: str) -> dict:
    """查看数据集结构（字段名与数据类型）。

    参数：
        path: 数据文件路径。
    返回值：
        dict: {columns 字段名列表, dtypes {字段: 类型字符串}, num_rows}。
    """
    df = load_dataframe(path)
    return {
        "columns": list(df.columns),
        # dtype 对象不能直接 JSON 序列化，统一转成字符串
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "num_rows": int(len(df)),
    }


def _profile_dataset(path: str) -> dict:
    """对数据集做完整画像（缺失/唯一/数值分布），供规划阶段判断数据质量。

    参数：
        path: 数据文件路径。
    返回值：
        dict: 含 path、行列数、columns、dtypes、missing_values（数量与占比）、
        unique_counts、numeric_columns、numeric_stats（均值/标准差/最值/四分位数）、
        sample（前 3 行）。
    """
    df = load_dataframe(path)
    num_rows = int(len(df))

    # 缺失值：逐列统计 NaN 数量与占比，仅保留有缺失的列，减少输出噪音
    missing = df.isna().sum()
    missing_info = {
        c: {"count": int(missing[c]), "pct": round(float(missing[c] / num_rows * 100), 2)}
        for c in df.columns
        if missing[c] > 0
    }

    # 唯一值数量：用于识别主键（≈行数）、二值标志、高基数类别列
    unique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    # 数值列基础统计：describe 一次给出计数值/均值/标准差/最值/四分位数
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_stats: dict[str, dict] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        for col in numeric_cols:
            row = desc.loc[col]
            # _native 把 np.float64 转成原生 float，保证结果可 JSON 序列化
            numeric_stats[col] = {
                "mean": _native(row["mean"]),
                "std": _native(row["std"]),
                "min": _native(row["min"]),
                "max": _native(row["max"]),
                "25%": _native(row["25%"]),
                "50%": _native(row["50%"]),
                "75%": _native(row["75%"]),
            }

    return {
        "path": str(get_settings().resolve_dataset_path(path)),
        "num_rows": num_rows,
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "missing_values": missing_info,
        "unique_counts": unique,
        "numeric_columns": numeric_cols,
        "numeric_stats": numeric_stats,
        "sample": df.head(3).to_dict(orient="records"),
    }


def _run_tool(handler, path: str) -> str:
    """统一执行文件类工具：成功返回 JSON，任何异常转成中文提示字符串。

    与其他工具（python_tool/chart_tool）保持一致的契约——工具不向外抛异常，
    保证 Agent 的工具调用循环不会因单个工具出错而中断。

    参数：
        handler: 真正干活的内部函数（_read_dataset / _inspect_schema / _profile_dataset）。
        path: 用户提供的数据文件路径（内部会做目录边界安全校验）。
    返回值：
        str: 正常结果的 JSON 字符串；出错时为结构化中文错误文本
        （含错误类型与修复建议，便于 Agent 自我纠正）。
    """
    try:
        return _json_dumps(handler(path))
    except (PermissionError, FileNotFoundError, ValueError) as e:
        # 路径越界、文件不存在、格式/体积问题：给出明确中文原因与类型
        kind, retryable = describe_exception(e)
        return tool_error(
            kind,
            f"{type(e).__name__}: {e}",
            hint="请确认路径位于 datasets/ 目录、文件存在且格式为 csv/xlsx/json。",
            retryable=retryable,
        )
    except Exception as e:  # 兜底：读取/解析阶段的其他意外
        kind, retryable = describe_exception(e)
        return tool_error(
            kind,
            f"读取失败：{type(e).__name__}: {e}",
            hint="文件可能已损坏或编码异常，可先用 execute_python 检查。",
            retryable=retryable,
        )


@tool
def read_dataset(path: str) -> str:
    """读取数据文件，返回行数、列数、字段名和前几行示例。支持 .csv / .xlsx / .json。

    返回值：
        str: 概览信息的 JSON 字符串；路径越界/文件缺失等错误返回中文提示，不抛异常。
    """
    return _run_tool(_read_dataset, path)


@tool
def inspect_schema(path: str) -> str:
    """查看数据字段名称和数据类型（schema）。支持 .csv / .xlsx / .json。

    返回值：
        str: schema 信息的 JSON 字符串；路径越界/文件缺失等错误返回中文提示，不抛异常。
    """
    return _run_tool(_inspect_schema, path)


@tool
def profile_dataset(path: str) -> str:
    """对数据集做完整画像：行数/列数、字段类型、缺失值、唯一值、数值列统计、示例。

    返回值：
        str: 完整画像的 JSON 字符串；路径越界/文件缺失等错误返回中文提示，不抛异常。
    """
    return _run_tool(_profile_dataset, path)
