"""LangGraph Agent 的共享状态（State）定义模块。

在 Agent 架构中，State 是流经图中所有节点的「唯一数据总线」：每个节点
接收当前状态、执行业务逻辑、返回需要更新的字段，LangGraph 负责合并。
本模块用 TypedDict 声明全部字段及其类型，并用 Annotated 指定合并方式。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

# LangChain 消息基类：对话历史中 System/Human/AI/Tool 消息的共同类型
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """Agent 全程共享状态。

    继承 TypedDict 以获得字段类型提示；total=False 表示所有字段都可选，
    这样初始状态可以只提供部分键，节点也只需返回它关心的增量字段。

    带 `Annotated[..., operator.add]` 的字段为“累加”字段：节点只返回新增元素，
    LangGraph 会自动追加到已有列表末尾（而不是整体覆盖）；
    未标注的普通字段则是「后写覆盖」语义。
    """

    # 对话 / 推理历史（累加）：承载系统提示、用户需求、AI 决策与工具回执
    messages: Annotated[list[BaseMessage], operator.add]

    # 用户输入：自然语言分析需求与数据集文件路径
    user_request: str
    dataset_path: str

    # 数据画像与分析规划：理解文本由任务理解节点写入，其余分别来自
    # profiler / skill_selection / planner 节点
    dataset_metadata: dict[str, Any]
    understanding: str
    analysis_plan: list[dict[str, Any]]
    selected_skills: list[str]
    skills_context: str

    # 工具调用记录（累加）：每次调用的入参/状态/耗时，以及工具返回的原始结果
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[dict[str, Any]], operator.add]

    # LLM 调用记录（累加）：节点名/模型/token 用量/耗时/成败，
    # 用于完整追踪一次 Agent 运行并落库（见 src/agent/observability.py）
    llm_calls: Annotated[list[dict[str, Any]], operator.add]

    # 实际生成的图表文件路径（累加）：仅记录 generate_chart 成功产出的文件
    generated_charts: Annotated[list[str], operator.add]

    # 观察结论 / 洞察 / 错误（累加）：
    # observations 为各阶段过程记录，insights 为最终洞察，errors 收集工具异常
    observations: Annotated[list[str], operator.add]
    insights: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # 结果：最终 Markdown 报告正文、落盘路径与整体状态（running/done）
    final_report: str
    report_path: str
    status: str

    # 降级标记（累加）：LLM 不可用、步数耗尽、工具反复失败等情况下记录原因，
    # 最终写入报告「分析局限性」章节，避免给出「看似完整」的误导性结论
    degraded: Annotated[list[str], operator.add]

    # 连续对话：由调用方（CLI）注入的历史上下文文本，空串表示无历史
    conversation_context: str

    # 控制：工具调用循环的当前步数与上限，用于防止 Agent 无限循环
    step_count: int
    max_steps: int
