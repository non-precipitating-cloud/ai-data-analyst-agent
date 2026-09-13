"""Skill 选择节点模块：根据任务挑选相关方法论 Skill 并注入后续上下文。

在 Agent 架构中位于 task_understanding 之后、profiler 之前。Skill 是
预置的「数据分析方法论卡片」（如异常检测、归因分析）。本节点把用户需求
与任务理解文本拿去和全部 Skill 做匹配，命中的 Skill 正文会被格式化后
放入 skills_context，供 planner 与 agent 节点在规划和执行时遵循。
"""

from __future__ import annotations

# AI 消息类型：选择结果会写入对话历史
from langchain_core.messages import AIMessage

# 图共享状态类型
from src.agent.state import AgentState
# Skill 相关能力：取全部 Skill、按文本匹配选择、格式化为上下文文本
from src.skills import format_skills, get_all_skills, select_skills


def skill_selection_node(state: AgentState) -> dict:
    """Skill 选择节点：匹配相关 Skills 并把方法论正文注入状态。

    :param state: 图当前共享状态，需包含 user_request；
                  understanding 由上一节点写入（可能尚不存在）
    :return: 状态增量：selected_skills 覆盖为命中的 slug 列表，
             skills_context 覆盖为格式化后的方法论正文，
             messages/observations 各累加一条选择记录
    """
    # 加载本地（或注册中心中）全部可用 Skill
    skills = get_all_skills()
    # LLM 选择过程的调用记录：由 select_skills 追加，随后写入状态供落库统计
    llm_calls: list[dict] = []
    # 结合用户需求与任务理解文本做相关性匹配，返回命中 Skill 的 slug 列表
    slugs = select_skills(
        state["user_request"], state.get("understanding", ""), skills, usage_sink=llm_calls
    )
    # 用 slug 过滤出完整的 Skill 对象，供格式化使用
    selected = [s for s in skills if s.slug in slugs]
    # 把命中 Skill 的正文拼成一段文本，后续注入规划/分析提示词
    context = format_skills(selected)
    # 展示用标签：无命中时明确标注「（无）」
    label = ", ".join(slugs) if slugs else "（无）"

    return {
        "selected_skills": slugs,
        "skills_context": context,
        # 对话历史中同时保留名称标签与方法论正文
        "messages": [AIMessage(content=f"[已选择 Skills]\n{label}\n\n{context}")],
        "observations": [f"[Skill 选择]\n已选择: {label}"],
        "llm_calls": llm_calls,
    }
