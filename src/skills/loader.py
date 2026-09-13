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

# 加载链路：发现子目录 SKILL.md → 手工解析 frontmatter（不依赖 PyYAML）
# → 构造 SkillMetadata；get_all_skills 带进程内缓存，format_skills 负责注入文本。

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.skills.models import SkillMetadata

# src/skills/ 包目录（skill 子目录与 SKILL.md 所在）
SKILLS_DIR = Path(__file__).resolve().parent


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 YAML 风格 frontmatter（key: value），返回 (元数据, 正文)。

    这里不引入 PyYAML：frontmatter 结构简单（扁平 key: value），
    手工解析即可；无 frontmatter 时元数据为空、正文为全文。

    Args:
        text: SKILL.md 的完整文本。

    Returns:
        (meta, body) 二元组：meta 为字段字典（triggers/tools 已拆成列表），
        body 为 frontmatter 之后的 Markdown 正文。
    """
    meta: dict = {}
    body = text
    # frontmatter 必须以 --- 开头
    if text.startswith("---"):
        # 最多切 2 次：[空串, frontmatter, 正文]
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm, body = parts[1], parts[2]
            for line in fm.strip().splitlines():
                if ":" in line:
                    # 按第一个冒号切分，值内部允许再出现冒号
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ("triggers", "tools"):
                        # 列表字段：兼容中文逗号“，”，按英文逗号拆分并去空白/空项
                        meta[key] = [
                            x.strip() for x in value.replace("，", ",").split(",") if x.strip()
                        ]
                    else:
                        # 普通标量字段（name/description 等）原样保留字符串
                        meta[key] = value
    return meta, body


def load_skill(path: str | Path) -> SkillMetadata:
    """加载单个 SKILL.md 并构造 SkillMetadata。

    Args:
        path: SKILL.md 文件路径。

    Returns:
        解析后的技能元数据；slug 取 SKILL.md 所在子目录名，
        frontmatter 缺失的 name/description/triggers/tools 使用默认值。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    # slug 以目录名为准（而非 frontmatter），保证与文件系统一一对应
    slug = p.parent.name
    return SkillMetadata(
        slug=slug,
        name=meta.get("name", slug),  # 缺 name 时回退为目录名
        description=meta.get("description", ""),
        triggers=meta.get("triggers", []),
        tools=meta.get("tools", []),
        content=body.strip(),
        path=str(p),
    )


def discover_skills(skills_dir: str | Path | None = None) -> list[SkillMetadata]:
    """自动发现目录下所有 ``*/SKILL.md`` 并逐个加载。

    Args:
        skills_dir: 技能根目录；为 None 时使用本包内置的 SKILLS_DIR。

    Returns:
        SkillMetadata 列表（按路径排序，顺序稳定）。
    """
    base = Path(skills_dir) if skills_dir else SKILLS_DIR
    skills: list[SkillMetadata] = []
    # 仅匹配一级子目录中的 SKILL.md，sorted 保证技能顺序可复现
    for path in sorted(base.glob("*/SKILL.md")):
        skills.append(load_skill(path))
    return skills


@lru_cache
def get_all_skills() -> tuple[SkillMetadata, ...]:
    """返回所有已发现的 Skill（进程内缓存，只读一次磁盘）。"""
    return tuple(discover_skills())


def get_skill(slug: str) -> SkillMetadata | None:
    """按 slug（子目录名）查找单个 Skill。

    Args:
        slug: 技能目录名标识。

    Returns:
        匹配的 SkillMetadata；不存在时返回 None。
    """
    for s in get_all_skills():
        if s.slug == slug:
            return s
    return None


def format_skills(skills: list[SkillMetadata] | tuple[SkillMetadata, ...]) -> str:
    """把选中的 Skill 正文编译为注入 LLM 上下文的文本。

    Args:
        skills: 选中的技能列表（通常 1-3 个）。

    Returns:
        拼接后的 Markdown 文本；空列表返回占位提示，避免提示词出现空白段落。
    """
    if not skills:
        return "（未选择任何 Skill）"
    # 每个技能带三级标题（中文名 + slug），后接其方法论正文
    parts = [f"### Skill: {s.name}（{s.slug}）\n{s.content}" for s in skills]
    return "\n\n".join(parts)
