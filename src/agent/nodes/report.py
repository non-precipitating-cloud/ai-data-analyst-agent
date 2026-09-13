"""报告节点模块：工作流的最后一环，生成并落盘结构化 Markdown 报告。

在 Agent 架构中位于 insight 之后、END 之前。节点把整个分析过程中
积累的真实数据（画像、工具结果、RAG 检索、洞察、图表白名单）组织成
提示词交给 LLM 撰写报告，随后做两道后处理：
1) sanitize_report 删除引用了不存在图表的图片标签；
2) validate_report 做占位符/章节/虚假图表的质量检查（仅告警不中断）；
最后调用 save_report_file 把报告写入磁盘。
"""

from __future__ import annotations

import logging
from pathlib import Path

# LangChain 消息类型
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# LLM 调用的容错与用量记录
from src.agent.observability import AgentLLMError, invoke_llm
# 报告节点专用系统提示词（17 个章节与防编造约束）
from src.agent.prompts import REPORT_SYSTEM
# 报告辅助函数：上下文构建、虚假图片清理、质量校验、降级兜底
from src.agent.report_builder import (
    build_degraded_report,
    build_report_context,
    sanitize_report,
    validate_report,
)
# 图共享状态类型
from src.agent.state import AgentState
# 长文本截断
from src.agent.utils import truncate
# LLM 工厂
from src.llm import get_llm
# 报告落盘工具：把 Markdown 内容写入报告目录并返回文件路径
from src.tools.report_tool import save_report_file

logger = logging.getLogger(__name__)


def _format_charts(chart_paths: list[str]) -> str:
    """将实际生成的图表路径格式化为清单文本（仅文件名）。

    :param chart_paths: 图表文件的完整路径列表
    :return: 每行一个 `- 文件名` 的清单文本；无图表时返回占位说明
    """
    if not chart_paths:
        return "（无——本次分析未生成任何图表）"
    # 报告提示词中只需文件名（白名单按文件名匹配），无需完整路径
    names = [Path(p).name for p in chart_paths]
    return "\n".join(f"- {n}" for n in names)


def _strip_code_fence(content: str) -> str:
    """去掉 LLM 可能误加的 Markdown 代码块围栏。

    报告提示词要求只输出 Markdown 正文，但模型仍可能用 ```markdown 包裹。
    本函数在检测到围栏时剥离首尾反引号及可选的 language 标记。

    :param content: LLM 返回的原始文本
    :return: 去除围栏后的纯 Markdown 文本；无围栏时基本原样返回
    """
    text = content.strip()
    if text.startswith("```"):
        # 去掉首尾连续的反引号字符
        text = text.strip("`")
        # 剥围栏后开头可能残留语言标识 markdown
        if text.startswith("markdown"):
            text = text[len("markdown"):]
        text = text.strip()
    return text


def _build_prompt(ctx: dict) -> str:
    """把结构化报告上下文拼接成分章节的人类提示词文本。

    各小节均来自 build_report_context 整理的真实执行结果，顺序上与
    REPORT_SYSTEM 要求的报告章节相呼应，确保 LLM「按事实填空」。

    :param ctx: build_report_context 产出的上下文字典
    :return: 拼接完成、可直接作为 HumanMessage 内容的字符串
    """
    meta = ctx["dataset_metadata"]
    parts: list[str] = []

    # 第 1 节：用户原始分析需求
    parts.append(f"# 用户分析需求\n{ctx['user_request']}")

    # 第 2 节：数据集画像（规模、字段、类型、数值列、缺失值）
    parts.append(
        "# 数据集\n"
        f"- 行数: {meta.get('num_rows')}\n"
        f"- 列数: {meta.get('num_columns')}\n"
        f"- 字段: {meta.get('columns', [])}\n"
        f"- 数据类型: {meta.get('dtypes', {})}\n"
        f"- 数值字段: {meta.get('numeric_columns', [])}\n"
        f"- 缺失值: {meta.get('missing_values', {})}"
    )

    # 第 3 节（可选）：本次实际选用的 Skill 名称
    if ctx["selected_skills"]:
        parts.append(f"# 本次使用的 Skills\n{', '.join(ctx['selected_skills'])}")

    # 第 4 节：工具调用摘要表格（local/MCP + 状态）
    parts.append(f"# 工具调用摘要\n{ctx['tool_call_summary']}")

    # 第 5 节：全部工具的真实返回结果（已在上游截断），是数字的唯一合法来源
    tr_text = "\n\n".join(
        f"[{tr['name']}] ({tr['status']})\n{tr['result']}" for tr in ctx["tool_results"]
    )
    parts.append(f"# 工具结果（真实数据）\n{tr_text or '（无）'}")

    # 第 6 节（可选）：RAG 知识库检索结果，作为「分析方法依据」
    if ctx["rag_results"]:
        rag_text = "\n\n".join(
            f"[{r.get('name')}]\n{truncate(str(r.get('result', '')), 800)}"
            for r in ctx["rag_results"]
        )
        parts.append(f"# 知识库检索结果（分析方法依据）\n{rag_text}")

    # 第 7 节（可选）：insight 节点产出的关键洞察
    if ctx["insights"]:
        parts.append("# 洞察\n" + "\n".join(ctx["insights"]))

    # 第 8 节（可选）：分析全过程的观察记录，截断后用于「分析过程」章节
    if ctx["observations"]:
        parts.append("# 分析过程观察\n" + truncate("\n".join(ctx["observations"]), 3000))

    # 第 9 节：图表白名单清单——明确告知 LLM 只能引用这些文件
    if ctx["generated_charts"]:
        chart_text = "\n".join(
            f"- {c['title']} → {c['relative']}" for c in ctx["generated_charts"]
        )
        parts.append(f"# 实际生成的图表清单（只能引用这些）\n{chart_text}")
    else:
        parts.append("# 实际生成的图表清单\n（无）")

    # 第 10 节（可选）：降级/未完成情况。
    # 必须显式告知模型，否则它会把「没分析出来」写成「没有问题」，
    # 这是数据分析里危害最大的一类误导。
    if ctx.get("degraded"):
        parts.append(
            "# 本次分析的降级/未完成情况（必须写入「分析局限性」章节）\n"
            + "\n".join(f"- {d}" for d in ctx["degraded"])
        )

    # 小节之间用空行分隔，拼成最终提示词
    return "\n\n".join(parts)


def report_node(state: AgentState) -> dict:
    """报告节点：根据完整分析过程生成结构化 Markdown 报告并落盘。

    :param state: 图最终共享状态（含画像、工具记录、洞察、图表等）
    :return: 状态增量：final_report 为报告正文，report_path 为保存路径
             （保存失败时为错误说明字符串），status 置为 "done"，
             messages 累加一条报告生成通知
    """
    llm = get_llm()
    # 先把原始状态整理成只含真实数据的结构化上下文
    ctx = build_report_context(state)

    # 用系统提示词（章节规范）+ 人类提示词（真实素材）调用 LLM 撰写报告
    prompt = HumanMessage(content=_build_prompt(ctx))
    extra: dict = {}
    try:
        resp, record = invoke_llm(
            [SystemMessage(content=REPORT_SYSTEM), prompt], llm=llm, node="report"
        )
        # 剥离模型可能误加的代码块围栏
        content = _strip_code_fence(resp.content)
        extra["llm_calls"] = [record]
    except AgentLLMError as e:
        # 报告阶段 LLM 不可用：用确定性模板把真实工具结果整理成报告落盘。
        # 宁可交付一份「没有模型润色、但每个数字都可追溯」的报告，
        # 也不要因为一次 LLM 故障丢失整轮分析成果。
        logger.error("报告 LLM 调用失败，改用确定性模板生成：%s", e)
        content = build_degraded_report(ctx, str(e))
        extra["llm_calls"] = [{"node": "report", "success": False, "error": str(e)}]
        extra["errors"] = [f"报告 LLM 调用失败：{e}"]
        extra["degraded"] = ["报告阶段 LLM 不可用，已改用确定性模板输出原始结果。"]

    # 清理虚假图表引用 + 质量检查（不中断）
    # 先删除引用白名单外图片的标签，再检查占位符/章节/残留虚假引用
    content = sanitize_report(content, ctx["valid_chart_names"])
    ok, issues = validate_report(content, ctx["valid_chart_names"])
    if not ok:
        # 质量问题仅记录告警，仍保留报告并继续落盘，避免整个流程失败
        logger.warning("报告质量检查发现问题：%s", issues)

    try:
        # 写入报告目录（默认标题 analysis_report），返回生成文件的路径
        path = save_report_file(content, title="analysis_report")
    except Exception as e:  # noqa: BLE001
        # 落盘失败不抛出：把错误信息作为路径占位返回，保证节点有完整输出
        path = f"(保存失败: {e})"

    return {
        "final_report": content,
        "report_path": path,
        # 标记工作流正常走到终点
        "status": "done",
        "messages": [AIMessage(content=f"[报告已生成] {path}")],
        **extra,
    }
