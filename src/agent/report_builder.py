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


def chart_title_from_path(path: str | Path) -> str:
    """根据图表文件名推导友好中文标题。

    文件名约定为「图表类型_X轴_Y轴_日期_时间.png」，例如
    line_month_sales.png → 月度销售额趋势。

    :param path: 图表文件路径
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
    # 为每张图表整理路径、文件名、中文标题及报告中使用的相对引用路径
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

    # 模板占位符 / JSON 残留：出现任一即说明 LLM 没把模板填完
    for placeholder in ("{{", "}}", "<chart_path>", "{chart", "TODO", "FIXME"):
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


def sanitize_report(content: str, valid_chart_names: set[str]) -> str:
    """清理报告中引用了不存在图表的 Markdown 图片标签。

    :param content: 原始 Markdown 报告
    :param valid_chart_names: 允许保留的图表文件名集合（白名单）
    :return: 清理后的报告；非法图片标签整体替换为空字符串
    """
    def _repl(m: re.Match) -> str:
        """正则替换回调：按图片 URL 的文件名是否在白名单决定去留。

        :param m: 匹配到的图片标签正则 Match 对象
        :return: 文件名合法则原样返回整个标签，否则返回空串将其删除
        """
        url = m.group(1)
        fname = Path(url).name
        return m.group(0) if fname in valid_chart_names else ""

    # 用回调逐个处理所有 ![...](...) 图片引用
    return _IMAGE_RE.sub(_repl, content)
