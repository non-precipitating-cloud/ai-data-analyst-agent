"""Skill 元数据模型。

loader 解析 SKILL.md 后产出该结构，selector 依据其中的
description / triggers 做选择，Agent 最终消费 content 正文。
使用 pydantic 做字段类型校验，frontmatter 缺字段时由加载侧补默认值。
"""

from __future__ import annotations

from pydantic import BaseModel


class SkillMetadata(BaseModel):
    """一个 Skill 的结构化元数据。"""

    slug: str                 # 目录名，如 "anomaly-detection"，选择器返回的稳定标识
    name: str                 # 中文名，如 "异常检测"，用于展示与提示词
    description: str          # 一句话描述（喂给 LLM 做语义选择的主要依据）
    triggers: list[str]       # 触发关键词（LLM 不可用时做确定性关键词匹配）
    tools: list[str]          # 推荐工具名列表（提示 Agent 优先使用哪些工具）
    content: str              # 完整正文（执行步骤/注意事项/示例等，选中后注入提示词）
    path: str                 # SKILL.md 文件的绝对路径，便于溯源与调试
