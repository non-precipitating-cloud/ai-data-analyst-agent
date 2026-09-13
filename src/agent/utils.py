"""Agent 通用工具函数模块。

存放与具体节点业务无关、被多处复用的小工具：LLM 输出的容错 JSON 解析、
长文本截断，以及送入 LLM 前的对话历史裁剪。task_understanding、planner、
insight、report、tool_calling 等节点都依赖这里的能力来应对大模型输出不
规范、上下文过长等常见问题。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


def parse_json(text: str) -> Any:
    """从 LLM 输出中稳健地解析 JSON（容忍 markdown 代码块与前后废话）。

    LLM 被要求只输出 JSON 时，仍可能用 ```json 围栏包裹或在前后加解释文字。
    本函数按「去围栏 → 整体解析 → 括号配平截取解析」三级策略逐级兜底。

    :param text: LLM 的原始文本输出
    :return: 解析成功返回对应 Python 对象（list/dict 等）；
             输入为空或三级尝试均失败时返回 None
    """
    if not text:
        return None

    text = text.strip()

    # 去掉 markdown 代码块围栏（```json ... ``` 或裸 ``` ... ```）
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 直接尝试：最理想情况——整段文本本身就是合法 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取第一个平衡的 [ ... ] 或 { ... }：用括号深度计数找到完整 JSON 片段，
    # 以容忍 LLM 在 JSON 前后输出的解释性文字
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                # 深度回到 0 说明找到了与起始括号配对的结束括号
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        # 截取到的片段仍非法，跳出尝试下一种括号
                        break
    return None


def truncate(text: str, limit: int = 6000) -> str:
    """截断长文本，避免单次请求携带过多内容导致 token 爆炸。

    :param text: 原始文本
    :param limit: 保留的最大字符数，默认 6000
    :return: 未超限则原样返回；超限则返回前 limit 个字符并追加截断说明
    """
    if len(text) <= limit:
        return text
    # 截断处附上原始长度，便于在日志/报告中意识到内容被裁过
    return text[:limit] + f"\n...(截断，原长度 {len(text)} 字符)"


def trim_messages_for_llm(
    messages: list[BaseMessage], *, keep_recent_tool_results: int = 6
) -> list[BaseMessage]:
    """裁剪送入 LLM 的对话历史，控制单次请求的上下文规模。

    为什么需要：agent ⇄ tools 循环每一轮都会把**完整历史**重新发给模型，
    而每条工具结果可达数千字符。跑到第十几轮时，请求体里绝大部分是重复的
    旧工具输出——既持续消耗 token，也逼近模型的上下文上限，反而挤掉了真正
    重要的近期信息。

    裁剪策略（保守，只动最冗余的部分）：
    - SystemMessage / HumanMessage（系统人设与用户需求）永远保留；
    - AIMessage 的**决策内容**全部保留：它记录了 Agent 每一步的选择，
      去掉会让模型对自己的推理链失去连续性；
    - ToolMessage 只保留最近 ``keep_recent_tool_results`` 条的原文，
      更早的替换为一行占位说明（保留工具名，便于模型知道「做过什么」）。

    这样既压缩了体积，又不会让模型误以为某些步骤从未执行过。

    :param messages: 原始消息列表
    :param keep_recent_tool_results: 保留原文的最近工具结果条数
    :return: 裁剪后的新列表（不修改入参）
    """
    if not messages:
        return []

    # 先定位所有 ToolMessage 的下标，只有它们会被替换
    tool_indexes = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    # 最近 N 条以内的保持原样
    keep_from = len(tool_indexes) - max(keep_recent_tool_results, 0)
    stale_indexes = set(tool_indexes[:keep_from]) if keep_from > 0 else set()

    if not stale_indexes:
        return list(messages)

    trimmed: list[BaseMessage] = []
    for i, m in enumerate(messages):
        if i in stale_indexes:
            # 占位消息保留工具名与原文长度，让模型知道该步骤存在过但内容已折叠
            name = getattr(m, "name", "") or "工具"
            length = len(str(getattr(m, "content", "")))
            trimmed.append(
                ToolMessage(
                    content=(
                        f"[历史工具结果已折叠] {name} 曾在更早的步骤返回约 {length} 字符的结果。"
                        "如需该数据，请重新调用对应工具。"
                    ),
                    tool_call_id=getattr(m, "tool_call_id", ""),
                    name=getattr(m, "name", None),
                )
            )
        else:
            trimmed.append(m)
    return trimmed


def build_degradation_note(degraded: list[str]) -> str:
    """把降级原因列表渲染成人类可读的说明段落。

    降级（LLM 不可用、步数耗尽、工具反复失败）会直接影响结论的完整性，
    必须显式写进报告，避免读者把「没分析出来」误读成「没有问题」。

    :param degraded: 降级原因列表
    :return: 每条一行的说明文本；无降级时返回空串
    """
    if not degraded:
        return ""
    lines = "\n".join(f"- {d}" for d in degraded)
    return f"本次分析存在以下降级/未完成情况：\n{lines}"
