"""Skill Selector：根据用户任务选择相关 Skill。

优先用 LLM 做语义选择（能理解「销售额下降原因」需异常检测），
LLM 不可用/解析失败时回退到确定性关键词匹配。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.observability import invoke_llm
from src.agent.utils import parse_json
from src.llm import get_llm
from src.skills.loader import get_all_skills
from src.skills.models import SkillMetadata

# 技能选择的系统提示词：约束 LLM 只输出 slug 的 JSON 数组，便于程序化解析
SKILL_SELECTION_SYSTEM = """你是技能选择器。根据用户分析需求，从候选 Skills 中选择最相关的 1-3 个。

只输出一个 JSON 数组，元素是 Skill 的 slug 字符串。例如：["sales-analysis", "anomaly-detection"]。
不要输出任何其他文字。"""


def select_skills_by_keywords(
    request: str, skills: list[SkillMetadata] | tuple[SkillMetadata, ...]
) -> list[str]:
    """确定性关键词匹配：任一 trigger 出现在请求文本中即选中该技能。

    Args:
        request: 用户的原始需求文本。
        skills: 候选技能集合。

    Returns:
        命中的技能 slug 列表（未命中任何 trigger 时为空列表）。
    """
    # 统一小写做子串匹配（trigger 中的英文关键词同样小写比较）
    text = request.lower()
    selected: list[str] = []
    for s in skills:
        # 技能匹配打分（确定性版本）：命中即 1 分，不做权重排序，按发现顺序返回
        if any(t.lower() in text for t in s.triggers):
            selected.append(s.slug)
    return selected


def select_skills(
    request: str,
    understanding: str = "",
    skills: list[SkillMetadata] | tuple[SkillMetadata, ...] | None = None,
    usage_sink: list | None = None,
) -> list[str]:
    """选择相关 Skill（LLM 语义优先 + 关键词回退）。

    Args:
        request: 用户的原始分析需求。
        understanding: Agent 对任务的理解文本（可选，辅助 LLM 判断）。
        skills: 候选技能集合；为 None 时使用已发现的全部技能。
        usage_sink: 可选的列表；LLM 调用记录（用量/耗时）会追加进去，
            供上层写入 AgentState 做可观测性统计。传 None 则不记录。

    Returns:
        选中技能的 slug 列表（通常 1-3 个）；无候选或均未命中时返回空列表。
    """
    skills = skills if skills is not None else get_all_skills()
    if not skills:
        return []

    # 1) LLM 语义选择：能理解“销售额下滑”这类不含显式关键词的隐含意图
    try:
        llm = get_llm()
        # 只把 slug/名称/一句话描述给 LLM，正文不参与选择（省 token）
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
        resp, record = invoke_llm(
            [SystemMessage(content=SKILL_SELECTION_SYSTEM), prompt],
            llm=llm,
            node="skill_selection",
        )
        if usage_sink is not None:
            usage_sink.append(record)
        # 容错解析 LLM 返回的 JSON 数组（可能带 markdown 代码块等噪声）
        slugs = parse_json(resp.content)
        # 白名单校验：丢弃 LLM 幻觉出的不存在 slug
        valid_slugs = {s.slug for s in skills}
        if isinstance(slugs, list):
            valid = [x for x in slugs if isinstance(x, str) and x in valid_slugs]
            if valid:
                return valid
    except Exception:  # noqa: BLE001 —— LLM 不可用/返回非法则静默回退
        if usage_sink is not None:
            # 记录一次失败，便于在可观测性统计里看到「LLM 选择未生效」
            usage_sink.append({"node": "skill_selection", "success": False})

    # 2) 关键词回退：不依赖 LLM 的确定性匹配，保证离线时技能机制仍可用
    return select_skills_by_keywords(request, skills)
