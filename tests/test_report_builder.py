"""Phase 11：report_builder 与报告节点测试（src.agent.report_builder / nodes.report）。

覆盖四大块：
- 图表路径校验：只承认真实存在且扩展名正确的图片；
- 工具调用摘要：本地 / MCP 工具的类型标注与空态提示；
- 报告上下文构建：过滤虚构图表、截断超长工具结果、空状态优雅处理；
- 报告质量检查：虚构图表 / 占位符 / 空内容判定与清洗，以及 report_node 的集成行为。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.nodes.report import report_node
from src.agent.report_builder import (
    build_report_context,
    build_tool_call_summary,
    chart_title_from_path,
    is_valid_chart_path,
    sanitize_report,
    validate_chart_paths,
    validate_report,
)
from src.config.settings import get_settings
from tests.conftest import SALES_CSV


# ---- 图表路径校验 ----
def test_is_valid_chart_path(tmp_path) -> None:
    """存在的 .png 为合法路径；不存在文件或错误扩展名均判为非法。"""
    p = tmp_path / "a.png"
    p.write_bytes(b"")
    assert is_valid_chart_path(p) is True
    # 文件不存在
    assert is_valid_chart_path(tmp_path / "missing.png") is False
    # 扩展名不是图片
    assert is_valid_chart_path(tmp_path / "a.txt") is False


def test_validate_chart_paths_filters(tmp_path) -> None:
    """批量校验应把路径拆成“有效”和“无效”两个列表，互不混淆。"""
    good = tmp_path / "good.png"
    good.write_bytes(b"")
    bad = tmp_path / "bad.png"  # 不存在
    valid, invalid = validate_chart_paths([str(good), str(bad)])
    assert valid == [str(good)]
    assert invalid == [str(bad)]


def test_chart_title_from_path() -> None:
    """图表文件名应被反解为人类可读的中文标题（类型+维度+指标）。"""
    assert chart_title_from_path("charts/line_month_sales_20260908_171345.png") == "月度销售额趋势"
    assert chart_title_from_path("bar_region_sales_20260908_171346.png") == "区域销售额对比"
    assert chart_title_from_path("box_region_sales_20260908_171352.png") == "区域销售额分布"
    assert chart_title_from_path("hist_sales_all_20260908_171352.png") == "销售额分布"


# ---- 工具摘要 / 类型 ----
def test_build_tool_call_summary_types() -> None:
    """摘要中本地工具标注 local，MCP 工具标注 MCP，并附带状态。"""
    calls = [
        {"name": "read_dataset", "status": "success"},
        {"name": "mcp__detect_outliers", "status": "success"},
    ]
    s = build_tool_call_summary(calls)
    assert "read_dataset | local | success" in s
    assert "mcp__detect_outliers | MCP | success" in s


def test_build_tool_call_summary_empty() -> None:
    """没有任何工具调用时，摘要应给出“未调用”之类的空态提示。"""
    assert "未调用" in build_tool_call_summary([])


# ---- report context ----
def test_build_report_context_filters_fake_charts(tmp_path) -> None:
    """上下文构建时应剔除不存在的图表，同时保留真实元数据、技能与工具结果。"""
    charts_dir = get_settings().charts_dir
    # 在真实图表目录放一个占位空文件，模拟已生成图表
    real = charts_dir / "real_a.png"
    real.write_bytes(b"")
    fake = charts_dir / "fake.png"  # 不存在

    # 构造一份同时含真实/虚构图表的完整状态
    state = {
        "user_request": "分析销售额",
        "dataset_metadata": {
            "num_rows": 2160, "num_columns": 12,
            "columns": ["sales"], "dtypes": {}, "numeric_columns": ["sales"],
            "missing_values": {},
        },
        "tool_calls": [{"name": "read_dataset", "status": "success"}],
        "tool_results": [{"name": "read_dataset", "result": '{"num_rows":2160}', "status": "success"}],
        "generated_charts": [str(real), str(fake)],
        "insights": ["洞察1"],
        "observations": ["观察1"],
        "selected_skills": ["sales-analysis"],
        "skills_context": "销售分析方法论",
        "errors": [],
    }
    ctx = build_report_context(state)
    # 虚构图表被过滤，只剩真实图表
    assert len(ctx["generated_charts"]) == 1
    assert ctx["generated_charts"][0]["name"] == "real_a.png"
    assert ctx["valid_chart_names"] == {"real_a.png"}
    # 技能上下文原样保留
    assert ctx["selected_skills"] == ["sales-analysis"]
    assert "2160" in ctx["tool_results"][0]["result"]  # 真实数字保留
    real.unlink()


def test_report_context_truncates_long_results() -> None:
    """超长工具结果应被截断并加上标记，避免把上万字符塞进报告 prompt。"""
    state = {"tool_results": [{"name": "execute_python", "result": "x" * 10000, "status": "success"}]}
    ctx = build_report_context(state)
    result = ctx["tool_results"][0]["result"]
    assert "截断" in result                      # 标记已截断
    assert len(result) < 3000                     # 远小于原始 10000


def test_report_context_empty_graceful() -> None:
    """空状态也应能安全构建上下文，图表/工具结果为空集合，不报错。"""
    ctx = build_report_context({})
    assert ctx["generated_charts"] == []
    assert ctx["valid_chart_names"] == set()
    assert ctx["tool_results"] == []


# ---- 质量检查 ----
def test_validate_report_ok() -> None:
    """包含核心发现且只引用“未生成图表”占位说明的报告应通过校验。"""
    content = "# 报告\n\n## 核心发现\n销售额下降 18.58%。\n\n## 图表\n本次分析未生成图表。"
    ok, issues = validate_report(content, set())
    assert ok, issues


def test_validate_report_fake_chart() -> None:
    """引用了有效清单之外的图片时，校验应失败并指出存在不存在的图表。"""
    ok, issues = validate_report("# 报告\n\n![x](charts/fake.png)", set())
    assert not ok
    assert any("不存在的图表" in i for i in issues)


def test_validate_report_placeholder() -> None:
    """报告中残留 <chart_path> 占位符时应判定不合格。"""
    ok, _ = validate_report("# 报告\n\n<chart_path>", set())
    assert not ok


def test_validate_report_empty() -> None:
    """空报告应直接判定不合格。"""
    ok, _ = validate_report("", set())
    assert not ok


def test_sanitize_removes_fake_chart() -> None:
    """清洗时应保留有效图片引用、移除虚构图片引用。"""
    content = "# 报告\n\n![a](charts/real.png)\n\n![b](charts/fake.png)"
    out = sanitize_report(content, {"real.png"})
    assert "real.png" in out
    assert "fake.png" not in out


# ---- report_node 集成 ----
class _CaptureLLM:
    """捕获传给报告节点的最后一条 prompt，同时返回固定的报告文本。"""

    def __init__(self) -> None:
        self.prompt = ""

    def invoke(self, messages) -> AIMessage:
        # 记录实际发给 LLM 的 prompt 内容，供断言检查
        self.prompt = messages[-1].content
        return AIMessage(content="# AI 数据分析报告\n\n## 图表\n本次分析未生成图表。")


def _base_state() -> dict:
    """构造报告节点所需的最小合法状态（无图表、无工具结果）。"""
    return {
        "user_request": "分析销售额",
        "dataset_metadata": {
            "num_rows": 2160, "num_columns": 12,
            "columns": ["sales", "region"], "dtypes": {}, "numeric_columns": ["sales"],
            "missing_values": {},
        },
        "tool_calls": [],
        "tool_results": [],
        "generated_charts": [],
        "insights": [],
        "observations": [],
        "selected_skills": [],
        "skills_context": "",
        "errors": [],
    }


def test_report_node_no_charts(monkeypatch) -> None:
    """无图表时报告节点应正常结束（done），正文含“未生成图表”说明。"""
    fake = _CaptureLLM()
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)
    result = report_node(_base_state())
    assert result["status"] == "done"
    assert "未生成图表" in result["final_report"]
    Path(result["report_path"]).unlink(missing_ok=True)


def test_report_node_prompt_contains_structured_data(monkeypatch) -> None:
    """报告 prompt 应包含图表标题/路径、skills、工具摘要(local/mcp)、RAG。"""
    from src.tools.chart_tool import generate_chart_file

    # 真实生成一张柱状图，供报告上下文引用
    chart_path = generate_chart_file(SALES_CSV, "bar", "region", "sales")

    fake = _CaptureLLM()
    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: fake)

    # 在基础状态上补充图表、技能、工具调用与 RAG 结果
    state = _base_state()
    state["generated_charts"] = [chart_path]
    state["selected_skills"] = ["sales-analysis", "anomaly-detection"]
    state["tool_calls"] = [
        {"name": "read_dataset", "status": "success"},
        {"name": "mcp__detect_outliers", "status": "success"},
    ]
    state["tool_results"] = [
        {"name": "retrieve_knowledge", "result": "[异常检测] z-score 阈值3", "status": "success"},
    ]

    report_node(state)

    # 校验实际 prompt 中各结构化部分都已注入
    prompt = fake.prompt
    assert "sales-analysis" in prompt
    assert "read_dataset | local" in prompt
    assert "mcp__detect_outliers | MCP" in prompt
    assert "知识库检索结果" in prompt  # RAG 部分
    assert "charts/" in prompt          # 图表相对路径
    assert chart_title_from_path(chart_path) in prompt  # 图表标题

    Path(chart_path).unlink(missing_ok=True)
