"""Agent 通用工具函数。"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(text: str) -> Any:
    """从 LLM 输出中稳健地解析 JSON（容忍 markdown 代码块与前后废话）。"""
    if not text:
        return None

    text = text.strip()

    # 去掉 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取第一个平衡的 [ ... ] 或 { ... }
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
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def truncate(text: str, limit: int = 6000) -> str:
    """截断长文本，避免 token 爆炸。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(截断，原长度 {len(text)} 字符)"
