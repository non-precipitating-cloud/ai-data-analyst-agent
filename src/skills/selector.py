"""Skill Selector：根据用户任务选择相关 Skill。

优先用 LLM 做语义选择（能理解「销售额下降原因」需异常检测），
LLM 不可用/解析失败时回退到确定性关键词匹配。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.utils import parse_json
from src.llm import get_llm
from src.skills.loader import get_all_skills
from src.skills.models import SkillMetadata

SKILL_SELECTION_SYSTEM = """你是技能选择器。根据用户分析需求，从候选 Skills 中选择最相关的 1-3 个。

只输出一个 JSON 数组，元素是 Skill 的 slug 字符串。例如：["sales-analysis", "anomaly-detection"]。
不要输出任何其他文字。"""


def select_skills_by_keywords(
    request: str, skills: list[SkillMetadata] | tuple[SkillMetadata, ...]
) -> list[str]:
    """确定性关键词匹配：命中 trigger 即选中。"""
    text = request.lower()
    selected: list[str] = []
    for s in skills:
        if any(t.lower() in text for t in s.triggers):
            selected.append(s.slug)
    return selected


def select_skills(
    request: str,
    understanding: str = "",
    skills: list[SkillMetadata] | tuple[SkillMetadata, ...] | None = None,
) -> list[str]:
    """选择相关 Skill（LLM 语义优先 + 关键词回退）。"""
    skills = skills if skills is not None else get_all_skills()
    if not skills:
        return []

    # 1) LLM 语义选择
    try:
        llm = get_llm()
        skill_descs = "\n".join(
            f"- {s.slug}（{s.name}）: {s.description}" for s in skills
        )
        prompt = HumanMessage(
            content=(
                f"用户需求：{request}\n"
                f"任务理解：{understanding}\n\n"
                f"候选 Skills：\n{skill_descs}"
            )
        )
        resp = llm.invoke([SystemMessage(content=SKILL_SELECTION_SYSTEM), prompt])
        slugs = parse_json(resp.content)
        valid_slugs = {s.slug for s in skills}
        if isinstance(slugs, list):
            valid = [x for x in slugs if isinstance(x, str) and x in valid_slugs]
            if valid:
                return valid
    except Exception:  # noqa: BLE001 —— LLM 不可用则回退
        pass

    # 2) 关键词回退
    return select_skills_by_keywords(request, skills)
