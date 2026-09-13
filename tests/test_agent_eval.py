"""把离线 Agent 评测纳入常规回归（tests 与 eval 共用同一套场景）。

评测场景定义在 ``eval/agent_eval.py``，这里只负责在 pytest 里执行它们。
这样做的好处是：评测与单测跑的是同一份真实生产代码路径，不会出现
「评测里过、实际跑起来不过」的两套实现。

关于评测能证明什么、不能证明什么，见 eval/agent_eval.py 的模块文档：
它验证的是**框架层**行为（工具选择接线、错误自纠、图表真实性、数字接地、
循环终止），用的是脚本化 LLM，**不代表真实模型的推理质量**。
"""

from __future__ import annotations

import pytest

from eval.agent_eval import build_scenarios, run_scenario


@pytest.mark.parametrize("scenario", build_scenarios(), ids=lambda s: s.name)
def test_agent_scenario(scenario) -> None:
    """每个评测场景都必须在真实工具执行下全部断言通过。"""
    result = run_scenario(scenario)
    failures = [d["message"] for d in result["details"] if not d["ok"]]
    assert result["passed"], f"场景 {result['name']} 未通过：{failures}"


def test_scenarios_cover_required_capabilities() -> None:
    """评测必须覆盖需求中列出的能力维度，避免场景被悄悄删减。

    这是一个「防退化」断言：如果有人为了让它变绿而删掉场景，
    这条测试会先失败。
    """
    names = {s.name for s in build_scenarios()}
    required = {
        "tool_selection_and_numeric_grounding",   # Tool selection + Numeric answer
        "self_correction_on_bad_column",          # Self-correction
        "chart_reality_and_reference",            # Chart correctness
        "sql_correctness_and_self_correction",    # SQL correctness
        "report_structure",                       # Report correctness
        "duplicate_calls_terminate_loop",         # Agent termination
        "sandbox_rejection_then_recovery",        # Security + recovery
    }
    assert required <= names, f"评测场景缺失: {required - names}"
