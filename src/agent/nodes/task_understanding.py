"""任务理解节点模块：工作流的第一个节点，用 LLM 解析用户自然语言需求。

在 Agent 架构中紧接 START 执行。它把用户那句模糊的「帮我分析下这份
销售数据」转化为结构化的任务理解（分析目标、关键指标、维度、建议分析
类型），结果写入 understanding，供后续 Skill 选择与规划使用。
"""

from __future__ import annotations

# LangChain 消息类型
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 任务理解节点专用系统提示词
from src.agent.prompts import TASK_UNDERSTANDING_SYSTEM
# LLM 调用的容错与用量记录
from src.agent.observability import AgentLLMError, invoke_llm
# 图共享状态类型
from src.agent.state import AgentState
# LLM 工厂
from src.llm import get_llm


def task_understanding_node(state: AgentState) -> dict:
    """任务理解节点：调用 LLM 解析用户需求，输出一段任务理解文本。

    本节点是「锦上添花」环节：任务理解失败不应让整次分析停下，
    因此 LLM 不可用时退化为直接使用用户的原始需求文本，并记录降级原因。

    :param state: 图当前共享状态，必须包含 user_request；
                  dataset_metadata 在本节点通常尚未生成（字段可能为空）
    :return: 状态增量：understanding 覆盖为理解文本，
             observations 与 messages 各累加一条带「[任务理解]」前缀的记录
    """
    llm = get_llm()
    request = state["user_request"]
    # 此时画像节点尚未执行，正常情况下 columns 为空；保留传入以便未来提前画像
    columns = state.get("dataset_metadata", {}).get("columns", [])

    # 组装人类消息：原始需求 + 已知字段信息
    prompt = HumanMessage(content=f"用户分析需求：{request}\n\n数据集字段：{columns}")

    try:
        # 系统提示词规定输出目标/指标/维度/分析类型四要素
        resp, record = invoke_llm(
            [SystemMessage(content=TASK_UNDERSTANDING_SYSTEM), prompt],
            llm=llm,
            node="task_understanding",
        )
        understanding = resp.content
        extra: dict = {"llm_calls": [record]}
    except AgentLLMError as e:
        # 降级：直接用原始需求作为任务理解，保证后续节点仍有可用输入
        understanding = request
        extra = {
            "llm_calls": [{"node": "task_understanding", "success": False, "error": str(e)}],
            "errors": [f"任务理解 LLM 调用失败：{e}"],
            "degraded": ["任务理解阶段 LLM 不可用，已退化为直接使用原始需求文本。"],
        }

    return {
        # 普通字段（覆盖语义）：保存任务理解供下游节点引用
        "understanding": understanding,
        # 累加字段：同步到过程观察与对话历史
        "observations": [f"[任务理解]\n{understanding}"],
        "messages": [AIMessage(content=f"[任务理解]\n{understanding}")],
        **extra,
    }
