"""LangGraph Agent 包：AI 数据分析师智能体的核心装配层。

本包负责把「任务理解 → Skill 选择 → 数据画像 → 规划 → 工具调用循环 →
洞察提炼 → 报告生成」整条工作流组装成一张可编译执行的 LangGraph 图。

对外暴露两个入口：
- ``build_graph``：构建并编译 Agent 工作流图；
- ``make_initial_state``：根据用户数据文件路径与分析需求构造图的初始状态。

**为什么这里用惰性导入（PEP 562）而不是直接在模块顶层 import**：
图会连带导入 ``src.agent.nodes`` → ``src.skills``，而 ``src.skills.selector``
又需要 ``src.agent.observability``。如果本文件在加载时就 eager 导入图，
那么「先 import src.skills」这条路径会触发循环导入而直接报 ImportError——
即只有从 ``main.py`` 进入才正常，从 skills 或 observability 单独进入就崩。
改为按需导入后，两种入口都能正常工作，代价只是首次访问时多一次 import。

用法保持不变::

    from src.agent import build_graph, make_initial_state
"""

from __future__ import annotations

from typing import Any

# 公开符号 → 所在模块的映射（按需导入，避免加载期循环依赖）
_LAZY_EXPORTS = {
    "build_graph": "src.agent.graph",
    "make_initial_state": "src.agent.graph",
}

__all__ = ["build_graph", "make_initial_state"]


def __getattr__(name: str) -> Any:
    """按需导入本包的公开符号（PEP 562 模块级 __getattr__）。

    :param name: 被访问的属性名
    :return: 对应模块中的对象
    :raises AttributeError: 属性不在公开清单中
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # 延迟到真正使用时才导入，从而打断 agent ↔ skills 的循环
    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    """让 dir() 与 IDE 补全能列出惰性导出的符号。"""
    return sorted(__all__)
