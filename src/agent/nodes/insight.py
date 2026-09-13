"""洞察节点模块：在工具调用循环结束后，基于真实工具结果提炼关键洞察。

在 Agent 架构中位于 agent ⇄ tools 循环之后、report 节点之前：
当 LLM 不再请求工具（或达到步数上限）时，图路由到本节点。本节点把
过程观察与全部工具结果汇总给 LLM，产出 3-7 条有数据支撑的洞察，
作为后续报告「核心发现」章节的重要素材。
"""

from __future__ import annotations

# LangChain 消息类型：分别承载系统指令、拼装的上下文与 AI 返回内容
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 洞察节点专用系统提示词
from src.agent.prompts import INSIGHT_SYSTEM
# 图共享状态类型
from src.agent.state import AgentState
# 长文本截断，控制送入 LLM 的工具结果长度
from src.agent.utils import truncate
# LLM 工厂：按全局配置返回聊天大模型实例
from src.llm import get_llm


def insight_node(state: AgentState) -> dict:
    """洞察节点：汇总工具执行结果，调用 LLM 提炼关键洞察。

    :param state: 图当前共享状态，需包含 user_request、observations、tool_results
    :return: 状态增量：insights 累加一条洞察文本，messages 累加一条
             带「[洞察]」前缀的 AI 消息（供对话历史与报告使用）
    """
    llm = get_llm()

    # 把各阶段过程观察拼成一段文本
    observations = "\n".join(state.get("observations", []))
    tool_results = state.get("tool_results", [])
    # 逐条渲染工具结果（工具名 + 截断后的返回内容），用空行分隔
    results_text = "\n\n".join(
        f"[{tr.get('name')}]\n{truncate(str(tr.get('result', '')), 2000)}"
        for tr in tool_results
    )

    # 组装给 LLM 的人类消息：需求 + 观察记录 + 真实工具结果
    prompt = HumanMessage(
        content=(
            f"分析需求：{state['user_request']}\n\n"
            f"观察记录：\n{truncate(observations)}\n\n"
            f"工具结果：\n{results_text}"
        )
    )
    # 系统提示词约束洞察格式与「必须有数字支撑」，随后发起 LLM 调用
    # 步数上限守卫：若最后一条消息仍想调用工具却因达到 MAX_STEPS 被强制结束循环，
    # 说明分析可能未完成，需要在洞察中显著标注，避免报告给出「看似完整」的误导性结论
    last_message = state.get("messages", [None])[-1] if state.get("messages") else None
    hit_step_limit = (
        state.get("step_count", 0) >= state.get("max_steps", 15)
        and bool(getattr(last_message, "tool_calls", None))
    )
    step_warning = (
        "\n\n【重要提醒】本次分析已达到最大步数上限，仍有分析动作未执行。"
        "请在洞察开头明确标注「分析受步数限制可能不完整」，并指出哪些计划步骤未完成。"
        if hit_step_limit
        else ""
    )
    resp = llm.invoke([SystemMessage(content=INSIGHT_SYSTEM + step_warning), prompt])
    insights = resp.content

    # 双保险：即使模型未按提醒开头标注，这里也程序化补一条醒目前缀
    if hit_step_limit and "步数" not in insights[:30]:
        insights = (
            "⚠️ 分析受最大步数限制可能不完整（部分计划步骤未执行）。\n\n" + insights
        )

    return {
        # insights 是累加字段：本轮洞察作为新元素追加
        "insights": [insights],
        # 同步写入对话历史，保持消息链完整
        "messages": [AIMessage(content=f"[洞察]\n{insights}")],
    }
