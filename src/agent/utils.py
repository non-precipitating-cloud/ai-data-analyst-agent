"""Agent 通用工具函数模块。

存放与具体节点业务无关、被多处复用的小工具：LLM 输出的容错 JSON 解析
与长文本截断。planner、insight、report 等节点都依赖这里的能力来应对
大模型输出不规范、上下文过长等常见问题。
"""

from __future__ import annotations

import json
import re
from typing import Any


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
