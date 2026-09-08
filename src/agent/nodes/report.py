"""报告节点：基于结构化上下文生成 Markdown 报告，含图表校验与质量检查。"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.prompts import REPORT_SYSTEM
from src.agent.report_builder import (
    build_report_context,
    sanitize_report,
    validate_report,
)
from src.agent.state import AgentState
from src.agent.utils import truncate
from src.llm import get_llm
from src.tools.report_tool import save_report_file

logger = logging.getLogger(__name__)


def _format_charts(chart_paths: list[str]) -> str:
    """将实际生成的图表路径格式化为清单文本（仅文件名）。"""
    if not chart_paths:
        return "（无——本次分析未生成任何图表）"
    names = [Path(p).name for p in chart_paths]
    return "\n".join(f"- {n}" for n in names)


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("markdown"):
            text = text[len("markdown"):]
        text = text.strip()
    return text


def _build_prompt(ctx: dict) -> str:
    meta = ctx["dataset_metadata"]
    parts: list[str] = []

    parts.append(f"# 用户分析需求\n{ctx['user_request']}")

    parts.append(
        "# 数据集\n"
        f"- 行数: {meta.get('num_rows')}\n"
        f"- 列数: {meta.get('num_columns')}\n"
        f"- 字段: {meta.get('columns', [])}\n"
        f"- 数据类型: {meta.get('dtypes', {})}\n"
        f"- 数值字段: {meta.get('numeric_columns', [])}\n"
        f"- 缺失值: {meta.get('missing_values', {})}"
    )

    if ctx["selected_skills"]:
        parts.append(f"# 本次使用的 Skills\n{', '.join(ctx['selected_skills'])}")

    parts.append(f"# 工具调用摘要\n{ctx['tool_call_summary']}")

    tr_text = "\n\n".join(
        f"[{tr['name']}] ({tr['status']})\n{tr['result']}" for tr in ctx["tool_results"]
    )
    parts.append(f"# 工具结果（真实数据）\n{tr_text or '（无）'}")

    if ctx["rag_results"]:
        rag_text = "\n\n".join(
            f"[{r.get('name')}]\n{truncate(str(r.get('result', '')), 800)}"
            for r in ctx["rag_results"]
        )
        parts.append(f"# 知识库检索结果（分析方法依据）\n{rag_text}")

    if ctx["insights"]:
        parts.append("# 洞察\n" + "\n".join(ctx["insights"]))

    if ctx["observations"]:
        parts.append("# 分析过程观察\n" + truncate("\n".join(ctx["observations"]), 3000))

    if ctx["generated_charts"]:
        chart_text = "\n".join(
            f"- {c['title']} → {c['relative']}" for c in ctx["generated_charts"]
        )
        parts.append(f"# 实际生成的图表清单（只能引用这些）\n{chart_text}")
    else:
        parts.append("# 实际生成的图表清单\n（无）")

    return "\n\n".join(parts)


def report_node(state: AgentState) -> dict:
    """根据完整分析过程生成结构化 Markdown 报告并落盘。"""
    llm = get_llm()
    ctx = build_report_context(state)

    prompt = HumanMessage(content=_build_prompt(ctx))
    resp = llm.invoke([SystemMessage(content=REPORT_SYSTEM), prompt])
    content = _strip_code_fence(resp.content)

    # 清理虚假图表引用 + 质量检查（不中断）
    content = sanitize_report(content, ctx["valid_chart_names"])
    ok, issues = validate_report(content, ctx["valid_chart_names"])
    if not ok:
        logger.warning("报告质量检查发现问题：%s", issues)

    try:
        path = save_report_file(content, title="analysis_report")
    except Exception as e:  # noqa: BLE001
        path = f"(保存失败: {e})"

    return {
        "final_report": content,
        "report_path": path,
        "status": "done",
        "messages": [AIMessage(content=f"[报告已生成] {path}")],
    }
