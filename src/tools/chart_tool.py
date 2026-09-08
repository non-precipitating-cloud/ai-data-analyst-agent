"""图表工具：生成柱状图 / 折线图 / 散点图 / 直方图 / 箱线图，保存到 reports/charts/。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 非交互后端，避免 GUI 依赖

import matplotlib.pyplot as plt  # noqa: E402

# 中文字体，避免图表中文标签显示为方框
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.tools.file_tools import load_dataframe  # noqa: E402

SUPPORTED_TYPES = {"bar", "line", "scatter", "hist", "box"}


def _as_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series)
    except (ValueError, TypeError):
        return series


def _build_plot(df: pd.DataFrame, chart_type: str, x: str, y: str, title: str) -> None:
    plt.figure(figsize=(10, 6))

    if chart_type == "bar":
        if y and y in df.columns:
            grouped = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False).head(20)
            plt.bar(grouped[x].astype(str), grouped[y])
            plt.ylabel(y)
        else:
            counts = df[x].value_counts()
            plt.bar(counts.index.astype(str), counts.values)
            plt.ylabel("count")
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "line":
        data = df.copy()
        data[x] = _as_datetime(data[x])
        data = data.sort_values(x)
        plt.plot(data[x], data[y], marker="o", markersize=3)
        plt.ylabel(y)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "scatter":
        plt.scatter(df[x], df[y], alpha=0.5, s=12)
        plt.xlabel(x)
        plt.ylabel(y)

    elif chart_type == "hist":
        plt.hist(pd.to_numeric(df[x], errors="coerce").dropna(), bins=30)
        plt.xlabel(x)
        plt.ylabel("frequency")

    elif chart_type == "box":
        if y and y in df.columns:
            df.boxplot(column=y, by=x, rot=45, figsize=(10, 6))
            plt.title(title or f"{y} by {x}")
            plt.suptitle("")
        else:
            plt.boxplot(pd.to_numeric(df[x], errors="coerce").dropna())
            plt.xlabel(x)

    plt.title(title or f"{chart_type}: {x}" + (f" vs {y}" if y else ""))
    plt.tight_layout()


def generate_chart_file(
    path: str, chart_type: str, x: str, y: str = "", title: str = ""
) -> str:
    """生成图表文件，返回保存路径。"""
    settings = get_settings()
    chart_type = chart_type.lower()
    if chart_type not in SUPPORTED_TYPES:
        raise ValueError(f"不支持的图表类型: {chart_type}，支持 {sorted(SUPPORTED_TYPES)}")

    df = load_dataframe(path)
    if x not in df.columns:
        raise ValueError(f"字段 '{x}' 不存在: {list(df.columns)}")
    if y and y not in df.columns:
        raise ValueError(f"字段 '{y}' 不存在: {list(df.columns)}")

    _build_plot(df, chart_type, x, y, title)

    settings.charts_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{chart_type}_{x}_{y or 'all'}_{stamp}.png"
    out_path = settings.charts_dir / fname
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return str(out_path)


@tool
def generate_chart(path: str, chart_type: str, x: str, y: str = "", title: str = "") -> str:
    """生成图表并保存为 PNG，返回文件路径。

    chart_type: bar / line / scatter / hist / box。
    x 为横轴字段；y 为数值字段（hist 可省略 y）。
    """
    try:
        out = generate_chart_file(path, chart_type, x, y, title)
        return f"图表已生成: {out}"
    except Exception as e:
        return f"图表生成失败：{type(e).__name__}: {e}"
