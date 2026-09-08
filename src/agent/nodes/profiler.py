"""数据画像节点：读取数据集并生成 profile（确定性，不调用 LLM）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.tools.file_tools import _profile_dataset


def profiler_node(state: AgentState) -> dict:
    """读取并画像数据集，写入 dataset_metadata 与观察。"""
    meta = _profile_dataset(state["dataset_path"])

    missing = meta.get("missing_values") or {}
    summary = (
        f"数据概览：{meta['num_rows']} 行 × {meta['num_columns']} 列；"
        f"字段 {meta['columns']}；"
        f"数值列 {meta.get('numeric_columns', [])}；"
        f"缺失值列数 {len(missing)}。"
    )
    return {
        "dataset_metadata": meta,
        "observations": [f"[数据画像]\n{summary}"],
        "messages": [AIMessage(content=f"[数据画像]\n{summary}")],
    }
