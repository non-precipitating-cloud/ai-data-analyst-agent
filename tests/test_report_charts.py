"""报告图表清单测试：保证报告只会引用实际生成的图表。

覆盖：
- 从工具输出文本中提取图表路径（_extract_chart_path）；
- tools_node 执行 generate_chart 后把真实存在的文件记入 generated_charts；
- report_node 组装的 prompt 图表清单严格限定为真实文件，不泄漏虚构路径；
- _format_charts 的空态提示。
"""

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
        # 保存最后一条（即拼装好的报告 prompt）
        self.prompt = messages[-1].content
        return AIMessage(content="# 报告\n\n## 图表说明\n见清单")


def test_extract_chart_path() -> None:
    """应从工具输出文本中提取 Windows 风格 .png 路径；无路径时返回 None。"""
    assert _extract_chart_path("图表已生成: C:/a/b.png") == "C:/a/b.png"
    assert _extract_chart_path("其他输出") is None


def test_tools_node_records_generated_charts() -> None:
    """tools_node 执行 generate_chart 后，会把真实存在的文件路径写入 generated_charts。"""
    # 构造一条仅含 generate_chart 调用的 AI 消息
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

    # 恰好记录一张图表
    assert len(update["generated_charts"]) == 1
    chart_path = Path(update["generated_charts"][0])
    assert chart_path.exists()  # 路径必须真实存在
    chart_path.unlink()  # 清理


def test_report_prompt_only_lists_real_charts(monkeypatch) -> None:
    """报告 prompt 的图表清单只能包含 generated_charts 中的真实文件，不出现不存在的文件。"""
    charts_dir = get_settings().charts_dir
    # 准备两个真实存在（空文件）的图表占位
    real = [charts_dir / "real_a.png", charts_dir / "real_b.png"]
    for p in real:
        p.write_bytes(b"")

    fake = _CapturingLLM()
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)

    # 状态里只登记这两张真实图表
    state = {
        "user_request": "分析销售额",
        "dataset_metadata": {"num_rows": 10, "num_columns": 3, "columns": ["a", "b", "c"]},
        "insights": ["洞察1"],
        "observations": ["观察1"],
        "tool_results": [],
        "generated_charts": [str(real[0]), str(real[1])],
    }
    result = report_node(state)

    # 用正则从 prompt 中抽出所有 .png 文件名，应恰好等于两张真实图表
    pngs = set(re.findall(r"[\w\-]+\.png", fake.prompt))
    assert pngs == {"real_a.png", "real_b.png"}

    # 报告正文不应出现清单之外的文件
    assert "real_c.png" not in result["final_report"]

    # 清理
    for p in real:
        p.unlink(missing_ok=True)
    Path(result["report_path"]).unlink(missing_ok=True)


def test_format_charts_empty() -> None:
    """空图表清单格式化后应包含“未生成”提示。"""
    assert "未生成" in _format_charts([])
