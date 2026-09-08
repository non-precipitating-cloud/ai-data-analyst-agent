"""LangGraph Agent State 定义。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """Agent 全程共享状态。

    带 `Annotated[..., operator.add]` 的字段为“累加”字段：节点只返回新增元素，
    LangGraph 会自动追加到已有列表末尾。
    """

    # 对话 / 推理历史（累加）
    messages: Annotated[list[BaseMessage], operator.add]

    # 用户输入
    user_request: str
    dataset_path: str

    # 数据画像与分析规划
    dataset_metadata: dict[str, Any]
    understanding: str
    analysis_plan: list[dict[str, Any]]
    selected_skills: list[str]
    skills_context: str

    # 工具调用记录（累加）
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[dict[str, Any]], operator.add]

    # 实际生成的图表文件路径（累加）
    generated_charts: Annotated[list[str], operator.add]

    # 观察结论 / 洞察 / 错误（累加）
    observations: Annotated[list[str], operator.add]
    insights: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # 结果
    final_report: str
    report_path: str
    status: str

    # 控制
    step_count: int
    max_steps: int
