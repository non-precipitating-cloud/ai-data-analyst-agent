"""Report 构建辅助：结构化上下文、图表校验、标题、工具摘要、质量检查。"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.agent.utils import truncate

logger = logging.getLogger(__name__)

VALID_CHART_EXTS = {".png", ".jpg", ".jpeg", ".svg"}

_FIELD_CN = {
    "sales": "销售额", "profit": "利润", "quantity": "销量", "month": "月度",
    "year": "年度", "region": "区域", "category": "品类", "product": "产品",
    "date": "时间", "discount": "折扣", "unit_price": "单价",
    "customer_segment": "客户", "department": "部门", "account": "科目", "amount": "金额",
}

_CHART_TYPE_CN = {
    "line": "趋势", "bar": "对比", "scatter": "关系", "hist": "分布", "box": "分布",
}

# Markdown 图片引用：![alt](url)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]*)\)")
# 图表文件名（含扩展名）
_CHART_FILE_RE = re.compile(r"[\w\-]+\.(?:png|jpg|jpeg|svg)")


def is_valid_chart_path(path: str | Path) -> bool:
    """图表路径是否真实有效：存在、是文件、扩展名合法。"""
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VALID_CHART_EXTS


def validate_chart_paths(chart_paths: list[str]) -> tuple[list[str], list[str]]:
    """校验图表路径，返回 (有效路径列表, 无效路径列表)。"""
    valid: list[str] = []
    invalid: list[str] = []
    for p in chart_paths:
        if is_valid_chart_path(p):
            valid.append(p)
        else:
            invalid.append(p)
            logger.warning("图表路径无效，将从报告排除：%s", p)
    return valid, invalid


def chart_title_from_path(path: str | Path) -> str:
    """根据图表文件名生成友好标题（如 line_month_sales.png → 月度销售额趋势）。"""
    name = Path(path).stem
    parts = name.split("_")
    chart_type = parts[0] if parts else ""
    core = parts[1:]
    # 去掉末尾时间戳（YYYYMMDD + HHMMSS 两段）
    if len(core) >= 2 and core[-2].isdigit() and len(core[-2]) == 8:
        core = core[:-2]
    x = core[0] if core else ""
    y = core[1] if len(core) > 1 else ""
    x_cn = _FIELD_CN.get(x, x)
    y_cn = _FIELD_CN.get(y, y)

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
    """生成工具调用摘要表格（local/mcp 类型）。"""
    if not tool_calls:
        return "（本次分析未调用任何工具）"
    rows = ["| 工具 | 类型 | 状态 |", "| --- | --- | --- |"]
    for tc in tool_calls:
        name = tc.get("name", "?")
        tool_type = "MCP" if name.startswith("mcp__") else "local"
        status = tc.get("status", "?")
        rows.append(f"| {name} | {tool_type} | {status} |")
    return "\n".join(rows)


def build_report_context(state: dict) -> dict:
    """构建统一的结构化报告上下文（全部来自 Agent 实际执行结果）。"""
    meta = state.get("dataset_metadata") or {}
    tool_calls = state.get("tool_calls") or []
    tool_results = state.get("tool_results") or []
    generated_charts = state.get("generated_charts") or []

    # 图表校验
    valid_charts, _ = validate_chart_paths(generated_charts)
    chart_items = [
        {
            "path": p,
            "name": Path(p).name,
            "title": chart_title_from_path(p),
            "relative": f"charts/{Path(p).name}",
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

    # RAG 结果（retrieve_knowledge 工具）
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
        "valid_chart_names": {Path(p).name for p in valid_charts},
        "errors": state.get("errors") or [],
    }


def validate_report(content: str, valid_chart_names: set[str] | None = None) -> tuple[bool, list[str]]:
    """轻量报告质量检查。返回 (是否通过, 问题列表)。"""
    issues: list[str] = []
    if not content or not content.strip():
        return False, ["报告为空"]

    text = content.strip()

    # 模板占位符 / JSON 残留
    for placeholder in ("{{", "}}", "<chart_path>", "{chart", "TODO", "FIXME"):
        if placeholder in text:
            issues.append(f"发现占位符/残留: {placeholder}")

    # 至少有一个标题 + 若干章节
    if not text.startswith("#") and "## " not in text:
        issues.append("缺少 Markdown 标题/章节")

    # 虚假图表引用
    if valid_chart_names is not None:
        mentioned = set(_CHART_FILE_RE.findall(text))
        fake = mentioned - valid_chart_names
        if fake:
            issues.append(f"引用了不存在的图表: {sorted(fake)}")

    return (not issues), issues


def sanitize_report(content: str, valid_chart_names: set[str]) -> str:
    """删除引用不存在图表的 Markdown 图片标签。"""
    def _repl(m: re.Match) -> str:
        url = m.group(1)
        fname = Path(url).name
        return m.group(0) if fname in valid_chart_names else ""

    return _IMAGE_RE.sub(_repl, content)
