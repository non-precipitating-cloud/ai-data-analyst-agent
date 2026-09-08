"""Skill 元数据模型。"""

from __future__ import annotations

from pydantic import BaseModel


class SkillMetadata(BaseModel):
    """一个 Skill 的结构化元数据。"""

    slug: str                 # 目录名，如 "anomaly-detection"
    name: str                 # 中文名，如 "异常检测"
    description: str          # 一句话描述（用于选择）
    triggers: list[str]       # 触发关键词（用于关键词匹配）
    tools: list[str]          # 推荐工具
    content: str              # 完整正文（执行步骤/注意事项/示例等）
    path: str                 # SKILL.md 路径
