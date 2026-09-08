"""Skill Loader：自动发现并解析 SKILL.md。

SKILL.md 格式：
    ---
    name: 异常检测
    description: 检测数据中的异常值
    triggers: 异常, 异常值, 离群点
    tools: detect_outliers, execute_python
    ---

    # 异常检测
    ## Purpose
    ...
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.skills.models import SkillMetadata

# src/skills/ 包目录（skill 子目录与 SKILL.md 所在）
SKILLS_DIR = Path(__file__).resolve().parent


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 YAML 风格 frontmatter（key: value），返回 (元数据, 正文)。"""
    meta: dict = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm, body = parts[1], parts[2]
            for line in fm.strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ("triggers", "tools"):
                        meta[key] = [
                            x.strip() for x in value.replace("，", ",").split(",") if x.strip()
                        ]
                    else:
                        meta[key] = value
    return meta, body


def load_skill(path: str | Path) -> SkillMetadata:
    """加载单个 SKILL.md。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    slug = p.parent.name
    return SkillMetadata(
        slug=slug,
        name=meta.get("name", slug),
        description=meta.get("description", ""),
        triggers=meta.get("triggers", []),
        tools=meta.get("tools", []),
        content=body.strip(),
        path=str(p),
    )


def discover_skills(skills_dir: str | Path | None = None) -> list[SkillMetadata]:
    """自动发现目录下所有 */SKILL.md。"""
    base = Path(skills_dir) if skills_dir else SKILLS_DIR
    skills: list[SkillMetadata] = []
    for path in sorted(base.glob("*/SKILL.md")):
        skills.append(load_skill(path))
    return skills


@lru_cache
def get_all_skills() -> tuple[SkillMetadata, ...]:
    """返回所有已发现的 Skill（缓存）。"""
    return tuple(discover_skills())


def get_skill(slug: str) -> SkillMetadata | None:
    """按 slug 查找 Skill。"""
    for s in get_all_skills():
        if s.slug == slug:
            return s
    return None


def format_skills(skills: list[SkillMetadata] | tuple[SkillMetadata, ...]) -> str:
    """把选中的 Skill 正文编译为注入上下文的文本。"""
    if not skills:
        return "（未选择任何 Skill）"
    parts = [f"### Skill: {s.name}（{s.slug}）\n{s.content}" for s in skills]
    return "\n\n".join(parts)
