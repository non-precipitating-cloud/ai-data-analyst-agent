"""图表工具：基于 matplotlib 生成静态分析图表并保存为 PNG。

输入输出契约：
- 输入：数据文件路径（csv/xlsx/json，由 file_tools 统一加载）、图表类型、
  横轴字段 x、可选数值字段 y、可选标题。
- 输出（@tool 层）：成功返回 "图表已生成: <绝对路径>" 字符串；
  失败返回 "图表生成失败：<异常类型>: <异常信息>"，不向上抛异常。
- 产物文件写入配置项 charts_dir（默认 reports/charts/），文件名带时间戳，避免覆盖。

支持图表类型：bar（柱状图）/ line（折线图）/ scatter（散点图）/ hist（直方图）/ box（箱线图）。
"""

from __future__ import annotations

from datetime import datetime

import matplotlib

# 必须在导入 pyplot 前切换到 Agg 非交互后端：服务端无显示器，避免 GUI 依赖与报错
matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _available_cjk_fonts() -> list[str]:
    """探测当前系统实际安装的中文字体，按优先级返回可用回退链。

    背景：不同环境的中文字体名完全不同——
    - Linux/Docker：fonts-noto-cjk 注册为 "Noto Sans CJK SC/JP"，或文泉驿；
    - Windows：Microsoft YaHei / SimHei；
    - macOS：PingFang SC / Heiti SC / Arial Unicode MS。
    若直接把字体名写死，matplotlib 在找不到时会静默回退 DejaVu Sans，
    中文标签就渲染成方框。因此这里用 fontManager 枚举已安装字体做交集。

    返回值：
        list[str]: 本机可用的中文字体名（按优先级排列），末尾追加
        matplotlib 自带的 "DejaVu Sans" 作为英文/数字兜底。
    """
    # 跨平台候选字体名，按显示效果与常见程度排序
    candidates = [
        "Noto Sans CJK SC",      # Linux：fonts-noto-cjk（思源黑体，本项目 Docker 镜像安装）
        "Noto Sans CJK JP",      # 部分发行版只注册日文字形名，但其字形同样覆盖常用汉字
        "Source Han Sans SC",    # 思源黑体的另一发行名称
        "WenQuanYi Zen Hei",     # 文泉驿正黑（常见 Linux 发行版）
        "WenQuanYi Micro Hei",   # 文泉驿微米黑
        "Microsoft YaHei",       # Windows 微软雅黑
        "SimHei",                # Windows 黑体
        "SimSun",                # Windows 宋体
        "PingFang SC",           # macOS 苹方
        "Heiti SC",              # macOS 黑体-简
        "Arial Unicode MS",      # macOS / 装过 Office 的系统
    ]
    # fontManager.ttflist 是 matplotlib 启动时扫描系统字体目录得到的已安装字体清单
    installed = {f.name for f in fm.fontManager.ttflist}
    available = [name for name in candidates if name in installed]
    # DejaVu Sans 随 matplotlib 自带，不含中文字形但保证英文、数字、负号总能渲染
    return available + ["DejaVu Sans"]


# 设置中文字体回退链：只放系统真正存在的字体，避免图表中文标签显示为方框
plt.rcParams["font.sans-serif"] = _available_cjk_fonts()
# 用中文字体后负号会显示异常，关闭 unicode_minus 让坐标轴负号正常渲染
plt.rcParams["axes.unicode_minus"] = False

import re  # noqa: E402

import pandas as pd  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.tools.errors import (  # noqa: E402
    KIND_INVALID_ARGUMENT,
    KIND_NOT_FOUND,
    KIND_UNSUPPORTED,
    describe_exception,
    tool_error,
)
from src.tools.file_tools import load_dataframe  # noqa: E402

# 支持的图表类型白名单（小写）
SUPPORTED_TYPES = {"bar", "line", "scatter", "hist", "box"}

# 各图表类型对 y 字段的要求：
# - required：必须提供 y，否则画不出有意义的图（折线/散点本质是 x-y 关系）
# - optional：y 可省略，省略时退化为计数/单变量分布
_Y_REQUIREMENT = {
    "bar": "optional",
    "line": "required",
    "scatter": "required",
    "hist": "optional",
    "box": "optional",
}


class ChartArgumentError(ValueError):
    """图表参数不合法（类型不支持、字段不存在、缺少必需参数等）。

    与普通 ValueError 的区别：它额外携带错误类型、可选值与修复建议，
    由 @tool 层渲染成结构化的中文错误文本回传给 LLM，便于其自我纠正。
    """

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        options: list | None = None,
        hint: str = "",
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.options = options
        self.hint = hint
        self.retryable = retryable

    def to_text(self) -> str:
        """渲染成与其他工具一致的结构化错误文本。"""
        return tool_error(
            self.kind,
            str(self),
            options=self.options,
            hint=self.hint,
            retryable=self.retryable,
        )


def _safe_name_part(value: str) -> str:
    """把字段名清洗为可安全用于文件名的片段。

    字段名可能含空格、斜杠、中文标点等，直接拼进文件名会产生非法路径或
    目录穿越。这里只保留单词字符、中文与连字符，其余替换为下划线。

    :param value: 原始字段名（或 y 的占位符）
    :return: 清洗后的文件名片段；清洗后为空时返回 "field"
    """
    cleaned = re.sub(r"[^\w一-鿿\-]+", "_", value or "").strip("_")
    return cleaned or "field"


def _as_datetime(series: pd.Series) -> pd.Series:
    """尝试把一列转换为日期时间类型，转换失败时原样返回。

    参数：
        series: 待转换的 pandas 列（折线图的横轴通常是日期字符串）。
    返回值：
        pd.Series: 转换后的 datetime 列；原值无法解析时返回原列。
    """
    try:
        return pd.to_datetime(series)
    except (ValueError, TypeError):
        # 非日期内容（如普通分类标签）保持原样，交给 matplotlib 按原值绘制
        return series


def _build_plot(df: pd.DataFrame, chart_type: str, x: str, y: str, title: str) -> None:
    """在当前 matplotlib 画布上按类型绘图（不负责保存与关闭画布）。

    参数：
        df: 已加载的数据表，且 x/y 字段存在性已由调用方校验。
        chart_type: 图表类型，取值见 SUPPORTED_TYPES。
        x: 横轴字段名。
        y: 纵轴数值字段名，可为空（柱状图退化为计数图，直方图/单箱线图可省略）。
        title: 图表标题，为空时自动用字段名拼装。
    返回值：
        None；图形绘制在 matplotlib 全局状态中，由调用方 savefig。
    """
    # 固定画布尺寸，保证不同图表风格统一
    plt.figure(figsize=(10, 6))

    if chart_type == "bar":
        if y and y in df.columns:
            # 有数值列：按 x 分组求和，取 Top20 避免类别过多挤爆横轴
            grouped = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False).head(20)
            plt.bar(grouped[x].astype(str), grouped[y])
            plt.ylabel(y)
        else:
            # 无 y：对 x 做值计数（频数分布）
            counts = df[x].value_counts()
            plt.bar(counts.index.astype(str), counts.values)
            plt.ylabel("count")
        # 横轴标签倾斜 45° 并右对齐，防止长标签互相重叠
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "line":
        data = df.copy()
        # 折线图通常沿时间展开：尝试转日期后按时间升序排列
        data[x] = _as_datetime(data[x])
        data = data.sort_values(x)
        plt.plot(data[x], data[y], marker="o", markersize=3)
        plt.ylabel(y)
        plt.xticks(rotation=45, ha="right")

    elif chart_type == "scatter":
        # alpha 半透明 + 小点，点密集时也能看出分布密度
        plt.scatter(df[x], df[y], alpha=0.5, s=12)
        plt.xlabel(x)
        plt.ylabel(y)

    elif chart_type == "hist":
        # 强转数值并丢弃无法解析的行；固定 30 个分箱
        plt.hist(pd.to_numeric(df[x], errors="coerce").dropna(), bins=30)
        plt.xlabel(x)
        plt.ylabel("frequency")

    elif chart_type == "box":
        if y and y in df.columns:
            # 按 x 分组画 y 的箱线图（观察不同类别下数值分布差异）
            df.boxplot(column=y, by=x, rot=45, figsize=(10, 6))
            plt.title(title or f"{y} by {x}")
            # pandas boxplot 会自带一个分组总标题，清空避免与主标题重复
            plt.suptitle("")
        else:
            # 单变量箱线图：快速观察中位数、四分位与离群点
            plt.boxplot(pd.to_numeric(df[x], errors="coerce").dropna())
            plt.xlabel(x)

    # 统一补标题；自动调整布局，避免轴标签被画布边缘裁掉
    plt.title(title or f"{chart_type}: {x}" + (f" vs {y}" if y else ""))
    plt.tight_layout()


def generate_chart_file(
    path: str, chart_type: str, x: str, y: str = "", title: str = ""
) -> str:
    """生成图表文件，返回保存路径。

    参数：
        path: 数据文件路径（相对 workspace 的路径或绝对路径）。
        chart_type: 图表类型（不区分大小写）。
        x: 横轴字段名，必须存在。
        y: 纵轴字段名，可空；提供时必须存在。
        title: 可选图表标题。
    返回值：
        str: 生成的 PNG 文件绝对路径。
    异常：
        ChartArgumentError: 图表类型不支持、x/y 字段不存在或缺少必需参数
            （携带错误类型与修复建议，由 @tool 层转成结构化错误文本）。
    """
    settings = get_settings()
    chart_type = (chart_type or "").strip().lower()
    if chart_type not in SUPPORTED_TYPES:
        raise ChartArgumentError(
            KIND_UNSUPPORTED,
            f"不支持的图表类型: {chart_type!r}",
            options=sorted(SUPPORTED_TYPES),
            hint="请从上述类型中选择一种后重试。",
        )

    # 复用文件工具的统一加载与路径安全校验逻辑
    try:
        df = load_dataframe(path)
    except Exception as e:  # noqa: BLE001 —— 转成带类型的参数错误，便于 LLM 分辨
        kind, retryable = describe_exception(e)
        raise ChartArgumentError(
            kind,
            f"无法读取数据集 {path}：{type(e).__name__}: {e}",
            hint="请确认 path 指向 datasets/ 目录下真实存在的文件。",
            retryable=retryable,
        ) from e

    if not x:
        raise ChartArgumentError(
            KIND_INVALID_ARGUMENT,
            "缺少横轴字段 x",
            options=list(df.columns),
            hint="请指定一个已存在的字段作为横轴。",
        )
    if x not in df.columns:
        raise ChartArgumentError(
            KIND_NOT_FOUND,
            f"横轴字段 '{x}' 不存在",
            options=list(df.columns),
            hint="请从上列字段中选择已存在的列名。",
        )
    if y and y not in df.columns:
        raise ChartArgumentError(
            KIND_NOT_FOUND,
            f"纵轴字段 '{y}' 不存在",
            options=list(df.columns),
            hint="请从上列字段中选择已存在的数值列，或留空 y 改画分布图。",
        )
    # 折线图/散点图必须给出 y：否则绘制时会抛 KeyError('')，
    # LLM 只能看到一句含糊的报错，无法判断该怎么改
    if _Y_REQUIREMENT[chart_type] == "required" and not y:
        raise ChartArgumentError(
            KIND_INVALID_ARGUMENT,
            f"{chart_type} 图必须提供纵轴字段 y",
            options=df.select_dtypes(include="number").columns.tolist(),
            hint="请补上数值型 y；若只想看分布，可改用 hist 或 box。",
        )

    _build_plot(df, chart_type, x, y, title)

    # 确保图表输出目录存在（首次运行时目录可能尚未创建）
    settings.charts_dir.mkdir(parents=True, exist_ok=True)
    # 时间戳命名：同一数据集多次出图不会互相覆盖
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 字段名先清洗再拼进文件名，避免空格/斜杠造成非法路径
    fname = f"{chart_type}_{_safe_name_part(x)}_{_safe_name_part(y) if y else 'all'}_{stamp}.png"
    out_path = settings.charts_dir / fname
    # bbox_inches="tight" 自动裁掉多余白边，保留倾斜的横轴标签
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    # 关闭全部画布释放内存，Agent 长循环多图生成时防止内存膨胀
    plt.close("all")
    return str(out_path)


@tool
def generate_chart(path: str, chart_type: str, x: str, y: str = "", title: str = "") -> str:
    """生成图表并保存为 PNG，返回文件路径（LangChain 工具入口）。

    chart_type: bar / line / scatter / hist / box。
    x 为横轴字段；y 为数值字段（hist 可省略 y）。

    返回值：
        str: 成功为 "图表已生成: <路径>"；任何异常都被捕获并转成结构化错误文本
        （含错误类型与修复建议），保证工具调用不会中断 Agent 的执行链路。
    """
    try:
        out = generate_chart_file(path, chart_type, x, y, title)
        return f"图表已生成: {out}"
    except ChartArgumentError as e:
        # 参数类错误：直接给出错误类型 + 可选值 + 建议，LLM 通常一次就能改对
        return e.to_text()
    except Exception as e:  # noqa: BLE001 —— 绘制期异常兜底，仍不抛出
        kind, retryable = describe_exception(e)
        return tool_error(
            kind,
            f"图表生成失败：{type(e).__name__}: {e}",
            hint="请检查字段类型与数据分布后重试，或改用其他图表类型。",
            retryable=retryable,
        )
