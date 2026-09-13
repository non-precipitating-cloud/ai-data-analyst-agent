"""Skills（分析技能）单元测试（src.skills）。

覆盖：
- discover_skills / get_all_skills：技能发现与全集数量；
- 每个技能的元数据完整性（name/description/triggers/tools 及正文必备小节）；
- get_skill：按 slug 取单个技能；
- _parse_frontmatter：YAML frontmatter 解析（含中英文逗号分隔）；
- 关键词选择器的命中与排除；
- format_skills：技能文本格式化；
- LLM 选择路径、LLM 失败时回退关键词；
- skill_selection_node 把选中技能注入状态与消息。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent.nodes.skill_selection import skill_selection_node
from src.skills import (
    discover_skills,
    format_skills,
    get_all_skills,
    get_skill,
    select_skills,
    select_skills_by_keywords,
)
from src.skills.loader import _parse_frontmatter

# 技能目录中应存在的全部 7 个 slug
ALL_SLUGS = {
    "data-cleaning",
    "exploratory-analysis",
    "sales-analysis",
    "financial-analysis",
    "anomaly-detection",
    "correlation-analysis",
    "report-generation",
}


def test_discover_skills_finds_all() -> None:
    """技能发现应找全 7 个 slug，且数量恰为 7。"""
    skills = discover_skills()
    assert {s.slug for s in skills} == ALL_SLUGS
    assert len(skills) == 7


def test_skill_metadata_complete() -> None:
    """每个技能的元数据与正文中的三个必备小节都不应缺失。"""
    for s in get_all_skills():
        assert s.name, f"{s.slug} 缺 name"
        assert s.description, f"{s.slug} 缺 description"
        assert s.triggers, f"{s.slug} 缺 triggers"
        assert s.tools, f"{s.slug} 缺 tools"
        assert "执行步骤" in s.content, f"{s.slug} 缺执行步骤"
        assert "注意事项" in s.content, f"{s.slug} 缺注意事项"
        assert "常见错误" in s.content, f"{s.slug} 缺常见错误"


def test_get_skill_anomaly() -> None:
    """按 slug 获取异常检测技能，名称、触发词与推荐工具都应正确。"""
    s = get_skill("anomaly-detection")
    assert s is not None
    assert s.name == "异常检测"
    assert "异常" in s.triggers
    assert "detect_outliers" in s.tools


def test_parse_frontmatter() -> None:
    """frontmatter 解析应正确切出元数据与正文，triggers/tools 支持中英文逗号。"""
    # tools 故意用中文逗号“，”分隔，验证解析器会统一拆分
    text = "---\nname: X\ntriggers: a, b\ntools: t1， t2\n---\nbody"
    meta, body = _parse_frontmatter(text)
    assert meta["name"] == "X"
    assert meta["triggers"] == ["a", "b"]
    assert meta["tools"] == ["t1", "t2"]
    assert body.strip() == "body"


def test_keyword_selector_relevant() -> None:
    """关键词选择器对各类需求应命中对应技能（清洗/异常/相关/销售）。"""
    skills = get_all_skills()
    assert select_skills_by_keywords("清洗这个 CSV 数据", skills) == ["data-cleaning"]
    assert "anomaly-detection" in select_skills_by_keywords("检测数据中的异常值", skills)
    assert "correlation-analysis" in select_skills_by_keywords("销售额和利润的关系", skills)
    assert "sales-analysis" in select_skills_by_keywords("分析销售额下降原因", skills)


def test_keyword_selector_excludes_irrelevant() -> None:
    """仅命中清洗意图时，不应附带异常检测、销售分析等无关技能。"""
    skills = get_all_skills()
    slugs = select_skills_by_keywords("清洗数据", skills)
    assert slugs == ["data-cleaning"]
    assert "anomaly-detection" not in slugs
    assert "sales-analysis" not in slugs


def test_format_skills_contains_content() -> None:
    """格式化后的技能文本应包含技能中文名与正文里的方法关键词（zscore）。"""
    skills = [s for s in get_all_skills() if s.slug == "anomaly-detection"]
    text = format_skills(skills)
    assert "异常检测" in text
    assert "zscore" in text


def test_select_skills_llm_path(monkeypatch) -> None:
    """LLM 正常返回 JSON slug 列表时，应直接采用 LLM 的选择结果。"""
    class Fake:
        def invoke(self, messages):
            # 返回一个 JSON 数组字符串，模拟技能选择器 LLM 输出
            return AIMessage(content='["anomaly-detection", "sales-analysis"]')

    monkeypatch.setattr("src.skills.selector.get_llm", lambda: Fake())
    slugs = select_skills("分析销售额下降原因", "理解", get_all_skills())
    assert slugs == ["anomaly-detection", "sales-analysis"]


def test_select_skills_fallback_to_keywords(monkeypatch) -> None:
    """LLM 不可用（如缺 API Key 抛错）时，应回退到关键词选择而非失败。"""
    def boom():
        # 模拟 get_llm 阶段就抛错（无 API Key）
        raise RuntimeError("no api key")

    monkeypatch.setattr("src.skills.selector.get_llm", boom)
    slugs = select_skills("检测异常值", "", get_all_skills())
    assert "anomaly-detection" in slugs


def test_skill_selection_node_injects_context(monkeypatch) -> None:
    """技能选择节点应输出选中 slug，并把技能正文注入 skills_context 与消息列表。"""
    class Fake:
        def invoke(self, messages):
            return AIMessage(content='["anomaly-detection"]')

    monkeypatch.setattr("src.skills.selector.get_llm", lambda: Fake())
    out = skill_selection_node({"user_request": "检测异常值", "understanding": "检测异常"})
    assert out["selected_skills"] == ["anomaly-detection"]
    # 上下文文本含技能中文名
    assert "异常检测" in out["skills_context"]
    # 同时以消息形式注入，供后续节点使用
    assert "异常检测" in out["messages"][0].content
