"""规划节点模块：在正式分析前让 LLM 产出结构化的分析计划。

在 Agent 架构中位于 profiler（数据画像）之后、agent 工具调用循环之前：
节点拿到任务理解、数据集字段和已选 Skill 方法论后，要求 LLM 输出一个
JSON 数组形式的分步计划（每步含目标、建议工具、做法），为后续 Agent
的工具调用提供「行动路线图」。
"""

from __future__ import annotations

import json

# LangChain 消息类型
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# LLM 调用的容错与用量记录
from src.agent.observability import AgentLLMError, invoke_llm
# 规划节点专用系统提示词（规定输出 JSON 数组的字段结构）
from src.agent.prompts import PLANNER_SYSTEM
# 图共享状态类型
from src.agent.state import AgentState
# parse_json 容错解析 LLM 输出；truncate 限制 Skill 上下文长度
from src.agent.utils import parse_json, truncate
# LLM 工厂
from src.llm import get_llm


def planner_node(state: AgentState) -> dict:
    """规划节点：调用 LLM 生成分析计划并解析为步骤列表。

    :param state: 图当前共享状态，需包含 user_request、understanding、
                  skills_context 及 dataset_metadata（字段/数值列信息）
    :return: 状态增量：analysis_plan 覆盖为解析出的步骤列表，
             observations 与 messages 各累加一份易读的计划文本
    """
    llm = get_llm()
    # 从数据画像中取出全部字段与数值字段，帮助 LLM 选出可行的工具与列
    meta = state.get("dataset_metadata", {})
    columns = meta.get("columns", [])
    numeric = meta.get("numeric_columns", [])

    # 组装人类消息：需求 + 任务理解 + Skill 方法论（截断）+ 字段信息
    prompt = HumanMessage(
        content=(
            f"用户需求：{state['user_request']}\n"
            f"任务理解：{state.get('understanding', '')}\n"
            f"已选择的 Skills 方法论：\n{truncate(state.get('skills_context', ''), 2500)}\n\n"
            f"数据集字段：{columns}\n"
            f"数值字段：{numeric}"
        )
    )
    extra: dict = {}
    try:
        # 发起 LLM 调用，期望返回纯 JSON 数组文本
        resp, record = invoke_llm(
            [SystemMessage(content=PLANNER_SYSTEM), prompt], llm=llm, node="planner"
        )
        extra["llm_calls"] = [record]
    except AgentLLMError as e:
        # 降级：没有计划也能继续跑（agent 节点会自主决定调用哪些工具），
        # 但必须在状态里留痕，避免后续报告把「没规划」当成「已按计划完成」
        plan_text = "（规划阶段 LLM 不可用，未能生成分析计划）"
        return {
            "analysis_plan": [],
            "observations": [f"[分析计划]\n{plan_text}"],
            "messages": [AIMessage(content=f"[分析计划]\n{plan_text}")],
            "llm_calls": [{"node": "planner", "success": False, "error": str(e)}],
            "errors": [f"规划 LLM 调用失败：{e}"],
            "degraded": ["规划阶段 LLM 不可用，本次分析未生成结构化计划。"],
        }

    # 容错解析：LLM 若带围栏/废话也能抽出 JSON；解析失败统一降级为空计划
    plan = parse_json(resp.content)
    if not isinstance(plan, list):
        plan = []
        extra["degraded"] = ["规划输出无法解析为 JSON 数组，已按无计划继续。"]

    # 把计划重新序列化为缩进 JSON 文本，便于写入观察记录与对话历史
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
    return {
        # 普通字段（覆盖语义）：保存最终步骤列表供后续节点参考
        "analysis_plan": plan,
        # 累加字段：在过程观察与对话历史中各留一份可读计划
        "observations": [f"[分析计划]\n{plan_text}"],
        "messages": [AIMessage(content=f"[分析计划]\n{plan_text}")],
        **extra,
    }
