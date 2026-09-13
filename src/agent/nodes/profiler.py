"""数据画像节点模块：读取数据集并生成结构画像（确定性步骤，不调用 LLM）。

在 Agent 架构中位于 skill_selection 之后、planner 之前。与其他节点不同，
本节点不调用大模型，而是直接复用 file_tools 中的 _profile_dataset 做
确定性统计（行数、列名、类型、数值列、缺失值等），为规划与后续分析
提供「数据到底长什么样」的事实依据，避免 LLM 凭空猜测字段。
"""

from __future__ import annotations

# AI 消息类型：画像摘要会写入对话历史
from langchain_core.messages import AIMessage

# 图共享状态类型
from src.agent.state import AgentState
# 复用文件工具中的画像实现（下划线开头表示内部函数，这里跨模块直接复用）
from src.tools.file_tools import _profile_dataset


def profiler_node(state: AgentState) -> dict:
    """画像节点：读取数据集，生成画像并写入状态。

    :param state: 图当前共享状态，必须包含 dataset_path
    :return: 状态增量：dataset_metadata 覆盖为完整画像字典；
             observations 与 messages 各累加一条人类可读的画像摘要
    """
    # 执行确定性画像：返回含行列数、字段、类型、缺失值等的元数据字典
    meta = _profile_dataset(state["dataset_path"])

    # 缺失值映射形如 {列名: 缺失数量}，其键的数量即「有缺失的列数」
    missing = meta.get("missing_values") or {}
    # 拼一行概要文本，供对话历史与报告的「数据概况」使用
    summary = (
        f"数据概览：{meta['num_rows']} 行 × {meta['num_columns']} 列；"
        f"字段 {meta['columns']}；"
        f"数值列 {meta.get('numeric_columns', [])}；"
        f"缺失值列数 {len(missing)}。"
    )
    return {
        # 保存完整画像，planner / report 等节点会进一步取用其中细节
        "dataset_metadata": meta,
        "observations": [f"[数据画像]\n{summary}"],
        "messages": [AIMessage(content=f"[数据画像]\n{summary}")],
    }
