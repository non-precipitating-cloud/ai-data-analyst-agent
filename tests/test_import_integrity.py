"""包导入完整性测试（防止循环导入回归）。

背景：``src.agent.__init__`` 曾用模块级 eager import 加载整张图，而图会连带
导入 ``src.agent.nodes`` → ``src.skills`` → ``src.skills.selector`` →
``src.agent.observability``。结果是「先 import src.skills」会触发循环导入
并抛 ImportError——从 ``main.py`` 进入一切正常，从 skills / observability
单独进入就崩。这类缺陷平时跑主流程发现不了，因此单独加回归。

测试在每个子进程里执行导入，保证不受本进程已加载模块的影响。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# 每个入口都必须能作为「第一个被导入的模块」正常工作
IMPORT_ENTRY_POINTS = [
    "import src.skills",
    "import src.agent.observability",
    "import src.agent.memory",
    "import src.agent.report_builder",
    "import src.agent.nodes.tool_calling",
    "import src.skills.selector",
    "from src.agent import build_graph, make_initial_state",
    "import src.tools",
    "import src.rag",
    "import src.db.service",
    "import src.mcp.server",
]


@pytest.mark.parametrize("statement", IMPORT_ENTRY_POINTS)
def test_module_is_importable_as_entry_point(statement: str) -> None:
    """任意模块都能作为程序入口被首先导入，不应出现循环导入。"""
    proc = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"`{statement}` 导入失败：\n{proc.stderr[-1500:]}"
    )


def test_lazy_exports_are_actually_available() -> None:
    """惰性导出不能变成「导不到」——属性必须真实可访问。"""
    import src.agent as agent

    assert callable(agent.build_graph)
    assert callable(agent.make_initial_state)
    assert "build_graph" in dir(agent)


def test_unknown_attribute_still_raises() -> None:
    """惰性 __getattr__ 不应把拼错的属性名静默吞掉。"""
    import src.agent as agent

    with pytest.raises(AttributeError):
        _ = agent.definitely_not_a_real_symbol


def test_cli_symbols_survive_non_utf8_console() -> None:
    """CLI 打印 ✓/✗/⚠ 时，在非 UTF-8 终端编码下也不能崩。

    Windows 默认控制台编码是 cp936(GBK)，其中没有 ✓/✗/⚠ 这些码位。
    不做兜底时 ``print`` 会抛 UnicodeEncodeError 直接把程序打挂——
    这是真实用户（Windows + 默认终端）会遇到的崩溃，因此单独回归。
    """
    code = (
        "import src.logging_config as lc\n"
        "lc.configure_stdio()\n"
        "print('✓ ✗ ⚠ 中文输出')\n"
    )
    # 在完整环境变量基础上只覆盖编码相关项：清空环境会让 Windows 上的
    # asyncio/winsock 初始化失败，那测的就不是编码问题了
    env = {**os.environ, "PYTHONIOENCODING": "gbk"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"非 UTF-8 终端下崩溃：\n{proc.stderr[-1200:]}"
