"""报告图表清单测试：验证报告只会引用实际生成的图表。"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.nodes.report import _format_charts, report_node
from src.agent.nodes.tool_calling import _extract_chart_path, tools_node
from src.config.settings import get_settings
from tests.conftest import SALES_CSV


class _CapturingLLM:
    """捕获传给报告节点的 prompt，返回固定报告内容。"""

    def __init__(self) -> None:
        self.prompt = ""

    def invoke(self, messages) -> AIMessage:
        self.prompt = messages[-1].content
        return AIMessage(content="# 报告\n\n## 图表说明\n见清单")


def test_extract_chart_path() -> None:
    assert _extract_chart_path("图表已生成: C:/a/b.png") == "C:/a/b.png"
    assert _extract_chart_path("其他输出") is None


def test_tools_node_records_generated_charts() -> None:
    """tools_node 执行 generate_chart 后，会把真实存在的文件路径写入 generated_charts。"""
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "generate_chart",
                "args": {"path": SALES_CSV, "chart_type": "bar", "x": "region", "y": "sales"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    update = tools_node({"messages": [msg]})

    assert len(update["generated_charts"]) == 1
    chart_path = Path(update["generated_charts"][0])
    assert chart_path.exists()  # 路径必须真实存在
    chart_path.unlink()  # 清理


def test_report_prompt_only_lists_real_charts(monkeypatch) -> None:
    """报告 prompt 的图表清单只能包含 generated_charts 中的真实文件，不出现不存在的文件。"""
    charts_dir = get_settings().charts_dir
    real = [charts_dir / "real_a.png", charts_dir / "real_b.png"]
    for p in real:
        p.write_bytes(b"")

    fake = _CapturingLLM()
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)

    state = {
        "user_request": "分析销售额",
        "dataset_metadata": {"num_rows": 10, "num_columns": 3, "columns": ["a", "b", "c"]},
        "insights": ["洞察1"],
        "observations": ["观察1"],
        "tool_results": [],
        "generated_charts": [str(real[0]), str(real[1])],
    }
    result = report_node(state)

    # 清单里只出现这两个真实图表，且不包含任何其他 .png
    pngs = set(re.findall(r"[\w\-]+\.png", fake.prompt))
    assert pngs == {"real_a.png", "real_b.png"}

    # 报告正文不应出现清单之外的文件
    assert "real_c.png" not in result["final_report"]

    # 清理
    for p in real:
        p.unlink(missing_ok=True)
    Path(result["report_path"]).unlink(missing_ok=True)


def test_format_charts_empty() -> None:
    assert "未生成" in _format_charts([])
