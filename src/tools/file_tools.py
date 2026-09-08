"""文件工具：读取 CSV / Excel / JSON 数据集并返回概览、schema、画像。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.config.settings import get_settings


def _native(value: Any) -> Any:
    """将 numpy 标量转换为原生 Python 类型，保证 JSON 可序列化。"""
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """根据扩展名加载 DataFrame（csv / xlsx / xls / json）。"""
    settings = get_settings()
    p = settings.resolve_dataset_path(path)

    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

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
    df = load_dataframe(path)
    return {
        "path": str(get_settings().resolve_dataset_path(path)),
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "head": df.head(5).to_dict(orient="records"),
    }


def _inspect_schema(path: str) -> dict:
    df = load_dataframe(path)
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "num_rows": int(len(df)),
    }


def _profile_dataset(path: str) -> dict:
    df = load_dataframe(path)
    num_rows = int(len(df))

    # 缺失值
    missing = df.isna().sum()
    missing_info = {
        c: {"count": int(missing[c]), "pct": round(float(missing[c] / num_rows * 100), 2)}
        for c in df.columns
        if missing[c] > 0
    }

    # 唯一值数量
    unique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    # 数值列基础统计
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_stats: dict[str, dict] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        for col in numeric_cols:
            row = desc.loc[col]
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


@tool
def read_dataset(path: str) -> str:
    """读取数据文件，返回行数、列数、字段名和前几行示例。支持 .csv / .xlsx / .json。"""
    return _json_dumps(_read_dataset(path))


@tool
def inspect_schema(path: str) -> str:
    """查看数据字段名称和数据类型（schema）。支持 .csv / .xlsx / .json。"""
    return _json_dumps(_inspect_schema(path))


@tool
def profile_dataset(path: str) -> str:
    """对数据集做完整画像：行数/列数、字段类型、缺失值、唯一值、数值列统计、示例。"""
    return _json_dumps(_profile_dataset(path))
