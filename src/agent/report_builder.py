"""报告构建辅助模块：为 report 节点准备结构化上下文并做报告质量把关。

在 Agent 架构中，本模块位于 report 节点与原始状态（AgentState）之间，
承担「报告生成前的数据整理 + 生成后的校验清洗」职责，使节点本身只关注
调用 LLM：
- build_report_context：把状态里分散的画像、工具结果、图表、洞察等
  汇总为一个上下文字典（全部来自真实执行结果，杜绝幻觉素材）；
- validate_chart_paths / sanitize_report / validate_report：
  负责图表白名单校验、虚假图片引用清理、占位符与章节完整性检查。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

# 长文本截断工具，防止工具结果拼进提示词后造成 token 爆炸
from src.agent.utils import truncate

logger = logging.getLogger(__name__)

# 允许在报告中引用的图表文件扩展名（白名单，防引用任意文件）
VALID_CHART_EXTS = {".png", ".jpg", ".jpeg", ".svg"}

# 常见数据字段英文名 → 中文名映射，用于把图表文件名转成友好中文标题
_FIELD_CN = {
    "sales": "销售额", "profit": "利润", "quantity": "销量", "month": "月度",
    "year": "年度", "region": "区域", "category": "品类", "product": "产品",
    "date": "时间", "discount": "折扣", "unit_price": "单价",
    "customer_segment": "客户", "department": "部门", "account": "科目", "amount": "金额",
}

# 图表类型前缀 → 中文语义（如 line 图表达「趋势」）
_CHART_TYPE_CN = {
    "line": "趋势", "bar": "对比", "scatter": "关系", "hist": "分布", "box": "分布",
}

# Markdown 图片引用：![alt](url)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]*)\)")
# 图表文件名（含扩展名）
_CHART_FILE_RE = re.compile(r"[\w\-]+\.(?:png|jpg|jpeg|svg)")

# 报告中引用图表时使用的相对目录（报告落在 reports/，图表落在 reports/charts/）
CHART_REL_DIR = "charts"


def _match_columns(tokens: list[str], columns: list[str]) -> list[str] | None:
    """用数据集真实列名把文件名片段贪心还原成字段名。

    为什么需要：图表文件名形如 ``bar_region_sales_20260913_120000.png``，
    字段之间也是下划线，因此 ``unit_price`` 这类含下划线的列名按位置切分
    必然切错（会得到 "unit" 与 "price" 两个不存在的字段）。有了真实列名，
    就能按「最长匹配」正确还原。

    :param tokens: 去掉图表类型与时间戳后的文件名片段
    :param columns: 数据集真实列名列表
    :return: 还原出的字段名列表；无法完整还原时返回 None（由调用方退回旧策略）
    """
    if not tokens or not columns:
        return None
    known = set(columns)
    matched: list[str] = []
    rest = list(tokens)
    while rest:
        # 从最长片段开始尝试，优先匹配更具体的列名
        for size in range(len(rest), 0, -1):
            candidate = "_".join(rest[:size])
            if candidate in known:
                matched.append(candidate)
                rest = rest[size:]
                break
        else:
            # 整段都无法匹配到已知列名，说明文件名不符合预期
            return None
    return matched or None


def is_valid_chart_path(path: str | Path) -> bool:
    """判断单个图表路径是否真实有效。

    有效需同时满足：路径在磁盘上存在、是普通文件、扩展名属于白名单。

    :param path: 待检查的图表路径（字符串或 Path 对象）
    :return: 有效返回 True，否则 False
    """
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VALID_CHART_EXTS


def validate_chart_paths(chart_paths: list[str]) -> tuple[list[str], list[str]]:
    """批量校验图表路径，把有效与无效路径分开。

    :param chart_paths: 待校验的图表路径列表
    :return: 二元组 (有效路径列表, 无效路径列表)；无效路径同时写告警日志
    """
    valid: list[str] = []
    invalid: list[str] = []
    for p in chart_paths:
        if is_valid_chart_path(p):
            valid.append(p)
        else:
            invalid.append(p)
            # 无效图表不会进入白名单，报告中对应引用稍后会被清理
            logger.warning("图表路径无效，将从报告排除：%s", p)
    return valid, invalid


def chart_title_from_path(path: str | Path, columns: list[str] | None = None) -> str:
    """根据图表文件名推导友好中文标题。

    文件名约定为「图表类型_X轴_Y轴_日期_时间.png」，例如
    line_month_sales.png → 月度销售额趋势。

    当提供 ``columns``（数据集真实列名）时，会优先按列名做最长匹配来还原
    字段，从而正确处理 ``unit_price`` 这类自身含下划线的列名；否则退回
    按位置切分（兼容历史文件名与无画像信息的场景）。

    :param path: 图表文件路径
    :param columns: 数据集真实列名；为 None 时使用按位置切分的旧策略
    :return: 中文化的图表标题；无法识别类型时退回原始文件名
    """
    name = Path(path).stem
    # 以下划线拆分：第一段是图表类型，其余段是字段名
    parts = name.split("_")
    chart_type = parts[0] if parts else ""
    core = parts[1:]
    # 去掉末尾时间戳（YYYYMMDD + HHMMSS 两段）
    if len(core) >= 2 and core[-2].isdigit() and len(core[-2]) == 8:
        core = core[:-2]

    # 优先用真实列名还原（可正确处理含下划线的字段名）
    matched = _match_columns(core, columns or []) if columns else None
    if matched is not None and len(matched) <= 2:
        core = matched

    x = core[0] if core else ""
    y = core[1] if len(core) > 1 else ""
    # 字段名翻译为中文；映射表中没有的保留原名
    x_cn = _FIELD_CN.get(x, x)
    y_cn = _FIELD_CN.get(y, y)

    # 按图表类型拼装符合中文阅读习惯的标题
    if chart_type == "scatter":
        return f"{x_cn} 与 {y_cn} 的关系"
    if chart_type == "line":
        return f"{x_cn}{y_cn}趋势" if y else f"{x_cn}趋势"
    if chart_type == "bar":
        return f"{x_cn}{y_cn}对比" if y else f"{x_cn}对比"
    if chart_type == "box":
        return f"{x_cn}{y_cn}分布" if y else f"{x_cn}分布"
    if chart_type == "hist":
        return f"{x_cn}分布"
    return name


def build_tool_call_summary(tool_calls: list[dict]) -> str:
    """把工具调用记录渲染成 Markdown 摘要表格。

    :param tool_calls: 工具调用字典列表，每项含 name/status 等键
    :return: Markdown 表格字符串；无调用时返回占位说明
    """
    if not tool_calls:
        return "（本次分析未调用任何工具）"
    # 表头 + 分隔行（Markdown 表格语法）
    rows = ["| 工具 | 类型 | 状态 |", "| --- | --- | --- |"]
    for tc in tool_calls:
        name = tc.get("name", "?")
        # MCP 工具统一以 mcp__ 前缀注册，据此区分本地工具与 MCP 工具
        tool_type = "MCP" if name.startswith("mcp__") else "local"
        status = tc.get("status", "?")
        rows.append(f"| {name} | {tool_type} | {status} |")
    return "\n".join(rows)


def build_report_context(state: dict) -> dict:
    """从 Agent 最终状态构建统一的结构化报告上下文。

    所有字段均取自 Agent 的实际执行结果（画像、工具返回、图表文件等），
    不引入任何外部假设，是报告「可追溯、防编造」的数据来源。

    :param state: LangGraph 共享状态（AgentState）
    :return: 供 _build_prompt 使用的上下文字典，含图表白名单等衍生字段
    """
    meta = state.get("dataset_metadata") or {}
    tool_calls = state.get("tool_calls") or []
    tool_results = state.get("tool_results") or []
    generated_charts = state.get("generated_charts") or []

    # 图表校验：只保留磁盘上真实存在且扩展名合法的图表
    valid_charts, _ = validate_chart_paths(generated_charts)
    # 数据集的真实列名：用于把图表文件名准确还原成字段名（含下划线的情况）
    known_columns = list(meta.get("columns") or [])
    # 为每张图表整理路径、文件名、中文标题及报告中使用的相对引用路径
    chart_items = [
        {
            "path": p,
            "name": Path(p).name,
            "title": chart_title_from_path(p, known_columns),
            "relative": f"{CHART_REL_DIR}/{Path(p).name}",
        }
        for p in valid_charts
    ]

    # 工具结果（截断，保留真实数字）
    truncated_results = [
        {
            "name": tr.get("name", "?"),
            "status": tr.get("status", "?"),
            "result": truncate(str(tr.get("result", "")), 2000),
        }
        for tr in tool_results
    ]

    # RAG 结果（retrieve_knowledge 工具）——单独抽出作为「分析方法依据」章节素材
    rag_results = [
        tr for tr in tool_results
        if tr.get("name") in ("retrieve_knowledge", "mcp__retrieve_knowledge")
    ]

    return {
        "user_request": state.get("user_request", ""),
        "dataset_metadata": meta,
        "selected_skills": state.get("selected_skills") or [],
        "skills_context": state.get("skills_context", ""),
        "tool_calls": tool_calls,
        "tool_call_summary": build_tool_call_summary(tool_calls),
        "tool_results": truncated_results,
        "rag_results": rag_results,
        "insights": state.get("insights") or [],
        "observations": state.get("observations") or [],
        "generated_charts": chart_items,
        # 有效图表文件名集合：报告中只允许引用这些名字
        "valid_chart_names": {Path(p).name for p in valid_charts},
        "errors": state.get("errors") or [],
        # 降级原因：报告「分析局限性」章节必须如实呈现
        "degraded": state.get("degraded") or [],
    }


def validate_report(content: str, valid_chart_names: set[str] | None = None) -> tuple[bool, list[str]]:
    """对 LLM 生成的报告做轻量级质量检查。

    检查项：非空、无模板占位符/JSON 残留、含 Markdown 标题、
    未引用白名单之外的图表文件。

    :param content: LLM 生成的 Markdown 报告正文
    :param valid_chart_names: 实际存在的图表文件名集合；为 None 时跳过图表检查
    :return: 二元组 (是否全部通过, 问题描述列表)
    """
    issues: list[str] = []
    if not content or not content.strip():
        return False, ["报告为空"]

    text = content.strip()

    # 模板占位符 / 未填充残留。
    # 注意 `}}` 单独出现并不代表占位符——报告里引用一段 JSON 数据（如
    # `{"a":{"b":1}}`）就会产生连续的右花括号。因此只把 Jinja 风格的
    # `{{ ... }}` 成对形式判定为占位符，避免对正常报告产生误报。
    if re.search(r"\{\{.*?\}\}", text, flags=re.DOTALL):
        issues.append("发现模板占位符: {{ ... }}")
    for placeholder in ("<chart_path>", "{chart", "TODO", "FIXME"):
        if placeholder in text:
            issues.append(f"发现占位符/残留: {placeholder}")

    # 至少有一个标题 + 若干章节
    if not text.startswith("#") and "## " not in text:
        issues.append("缺少 Markdown 标题/章节")

    # 虚假图表引用：报告提到的图表文件名减去白名单，差集即「幻觉图表」
    if valid_chart_names is not None:
        mentioned = set(_CHART_FILE_RE.findall(text))
        fake = mentioned - valid_chart_names
        if fake:
            issues.append(f"引用了不存在的图表: {sorted(fake)}")

    return (not issues), issues


def _normalize_chart_url(url: str, fname: str) -> str:
    """把图表图片地址规范化为报告中可用的相对路径 ``charts/<文件名>``。

    报告保存在 ``reports/`` 下、图表保存在 ``reports/charts/`` 下，因此只有
    ``charts/<文件名>`` 这种相对写法才能在 Markdown 预览里正确加载。LLM 可能
    写成绝对路径、``./charts/x.png``、甚至只有裸文件名——它们指向的文件确实
    存在，但链接是坏的。既然白名单已确认文件真实存在，这里统一改写为规范路径。

    :param url: LLM 写入的原始图片地址
    :param fname: 该地址对应的文件名（已确认在白名单内）
    :return: 规范化后的相对路径
    """
    # 已经规范的写法原样返回，避免无谓改写
    if url == f"{CHART_REL_DIR}/{fname}":
        return url
    return f"{CHART_REL_DIR}/{fname}"


def sanitize_report(content: str, valid_chart_names: set[str]) -> str:
    """清理报告中引用了不存在图表的图片标签，并规范化保留项的路径。

    两步处理：
    1. 图片文件名不在白名单 → 整条标签删除（防止报告引用不存在/幻觉出的图表）；
    2. 文件名在白名单 → 把地址改写为 ``charts/<文件名>``，保证链接真的能打开。

    :param content: 原始 Markdown 报告
    :param valid_chart_names: 允许保留的图表文件名集合（白名单）
    :return: 清理并规范化后的报告
    """
    def _repl(m: re.Match) -> str:
        """正则替换回调：按图片 URL 的文件名是否在白名单决定去留。

        :param m: 匹配到的图片标签正则 Match 对象
        :return: 文件名合法则返回路径规范化后的标签，否则返回空串将其删除
        """
        url = m.group(1)
        fname = Path(url).name
        if fname not in valid_chart_names:
            # 引用了不存在的图表：整条删除，而不是留下一个打不开的链接
            return ""
        normalized = _normalize_chart_url(url, fname)
        # 只替换 URL 部分，保留作者写的 alt 文本
        return m.group(0).replace(f"]({url})", f"]({normalized})")

    # 用回调逐个处理所有 ![...](...) 图片引用
    return _IMAGE_RE.sub(_repl, content)


def build_degraded_report(ctx: dict, reason: str) -> str:
    """LLM 不可用时，用确定性模板把真实结果整理成 Markdown 报告。

    设计原则：**只搬运，不推断**。模板里出现的每一个数字都直接来自工具结果
    原文，不生成任何归纳性结论。这样即使模型不可用，用户仍能拿到一份数据可
    追溯的报告，且不会把「未经分析的原始数据」伪装成分析结论。

    :param ctx: build_report_context 产出的上下文字典
    :param reason: 降级原因（会写入报告开头）
    :return: Markdown 报告文本
    """
    meta = ctx.get("dataset_metadata") or {}
    lines: list[str] = [
        "# AI 数据分析报告",
        "",
        "> ⚠️ **本报告为降级输出**：报告生成阶段大模型不可用，以下内容由确定性模板"
        "直接汇总工具原始结果，**未经过模型归纳**，因此不含核心发现与原因推断。",
        "",
        f"- 降级原因：{reason}",
        f"- 用户需求：{ctx.get('user_request', '')}",
        "",
        "## 分析概览",
        f"- 数据集行数：{meta.get('num_rows', '未获取')}",
        f"- 数据集列数：{meta.get('num_columns', '未获取')}",
        f"- 字段：{meta.get('columns', [])}",
        "",
        "## 工具调用摘要",
        ctx.get("tool_call_summary", "（无）"),
        "",
        "## 工具结果（原始数据）",
    ]
    results = ctx.get("tool_results") or []
    if results:
        for tr in results:
            lines += [
                f"### {tr.get('name', '?')}（{tr.get('status', '?')}）",
                "```",
                str(tr.get("result", "")),
                "```",
                "",
            ]
    else:
        lines += ["（本次未取得任何工具结果）", ""]

    charts = ctx.get("generated_charts") or []
    lines += ["## 图表"]
    if charts:
        lines += [f"![{c['title']}]({c['relative']})" for c in charts]
    else:
        lines += ["本次分析未生成图表。"]
    lines.append("")

    lines += ["## 分析局限性", "- 报告由确定性模板生成，未包含模型提炼的洞察与因果推断。"]
    for d in ctx.get("degraded") or []:
        lines.append(f"- {d}")
    if ctx.get("errors"):
        lines.append("- 过程错误：")
        lines += [f"  - {e}" for e in ctx["errors"][:10]]
    return "\n".join(lines)
