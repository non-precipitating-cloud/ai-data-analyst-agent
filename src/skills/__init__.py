"""Agent Skills（技能）包。

技能机制：每个技能是一个子目录中的 SKILL.md（YAML frontmatter + Markdown 正文），
描述某类分析任务的方法论、触发词与推荐工具。运行时先由 selector 根据用户
需求选出 1-3 个相关技能（LLM 语义选择，关键词匹配兜底），再由 loader 把
技能正文格式化注入提示词，引导 Agent 按既定方法论完成分析。
"""

# 加载与发现：扫描子目录、解析 SKILL.md frontmatter、格式化注入文本
from src.skills.loader import (
    discover_skills,
    format_skills,
    get_all_skills,
    get_skill,
    load_skill,
)
# 技能元数据结构（pydantic 模型）
from src.skills.models import SkillMetadata
# 技能选择：LLM 语义选择 + 关键词回退
from src.skills.selector import (
    select_skills,
    select_skills_by_keywords,
)

__all__ = [
    "SkillMetadata",
    "discover_skills",
    "load_skill",
    "get_all_skills",
    "get_skill",
    "format_skills",
    "select_skills",
    "select_skills_by_keywords",
]
