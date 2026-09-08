"""Skills 单元测试：发现、解析、元数据、选择器、上下文注入。"""

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
    skills = discover_skills()
    assert {s.slug for s in skills} == ALL_SLUGS
    assert len(skills) == 7


def test_skill_metadata_complete() -> None:
    for s in get_all_skills():
        assert s.name, f"{s.slug} 缺 name"
        assert s.description, f"{s.slug} 缺 description"
        assert s.triggers, f"{s.slug} 缺 triggers"
        assert s.tools, f"{s.slug} 缺 tools"
        assert "执行步骤" in s.content, f"{s.slug} 缺执行步骤"
        assert "注意事项" in s.content, f"{s.slug} 缺注意事项"
        assert "常见错误" in s.content, f"{s.slug} 缺常见错误"


def test_get_skill_anomaly() -> None:
    s = get_skill("anomaly-detection")
    assert s is not None
    assert s.name == "异常检测"
    assert "异常" in s.triggers
    assert "detect_outliers" in s.tools


def test_parse_frontmatter() -> None:
    text = "---\nname: X\ntriggers: a, b\ntools: t1， t2\n---\nbody"
    meta, body = _parse_frontmatter(text)
    assert meta["name"] == "X"
    assert meta["triggers"] == ["a", "b"]
    assert meta["tools"] == ["t1", "t2"]
    assert body.strip() == "body"


def test_keyword_selector_relevant() -> None:
    skills = get_all_skills()
    assert select_skills_by_keywords("清洗这个 CSV 数据", skills) == ["data-cleaning"]
    assert "anomaly-detection" in select_skills_by_keywords("检测数据中的异常值", skills)
    assert "correlation-analysis" in select_skills_by_keywords("销售额和利润的关系", skills)
    assert "sales-analysis" in select_skills_by_keywords("分析销售额下降原因", skills)


def test_keyword_selector_excludes_irrelevant() -> None:
    skills = get_all_skills()
    slugs = select_skills_by_keywords("清洗数据", skills)
    assert slugs == ["data-cleaning"]
    assert "anomaly-detection" not in slugs
    assert "sales-analysis" not in slugs


def test_format_skills_contains_content() -> None:
    skills = [s for s in get_all_skills() if s.slug == "anomaly-detection"]
    text = format_skills(skills)
    assert "异常检测" in text
    assert "zscore" in text


def test_select_skills_llm_path(monkeypatch) -> None:
    class Fake:
        def invoke(self, messages):
            return AIMessage(content='["anomaly-detection", "sales-analysis"]')

    monkeypatch.setattr("src.skills.selector.get_llm", lambda: Fake())
    slugs = select_skills("分析销售额下降原因", "理解", get_all_skills())
    assert slugs == ["anomaly-detection", "sales-analysis"]


def test_select_skills_fallback_to_keywords(monkeypatch) -> None:
    def boom():
        raise RuntimeError("no api key")

    monkeypatch.setattr("src.skills.selector.get_llm", boom)
    slugs = select_skills("检测异常值", "", get_all_skills())
    assert "anomaly-detection" in slugs


def test_skill_selection_node_injects_context(monkeypatch) -> None:
    class Fake:
        def invoke(self, messages):
            return AIMessage(content='["anomaly-detection"]')

    monkeypatch.setattr("src.skills.selector.get_llm", lambda: Fake())
    out = skill_selection_node({"user_request": "检测异常值", "understanding": "检测异常"})
    assert out["selected_skills"] == ["anomaly-detection"]
    assert "异常检测" in out["skills_context"]
    assert "异常检测" in out["messages"][0].content
