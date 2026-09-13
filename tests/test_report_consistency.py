"""报告与图表一致性回归测试。

对应「图表系统」与「数据分析正确性」两节要求：
- 报告引用的图表必须真实存在，路径必须能被 Markdown 正确加载；
- 图表标题要能正确还原含下划线的字段名（如 unit_price）；
- 质量检查不能把正常报告误判为有残留（含 JSON 的报告很常见）。
"""

from __future__ import annotations

from pathlib import Path

from src.agent.report_builder import (
    build_report_context,
    chart_title_from_path,
    sanitize_report,
    validate_report,
)
from src.config.settings import get_settings
from tests.conftest import SALES_CSV


# ==========================================================================
# 图表链接规范化
# ==========================================================================
def test_sanitize_normalizes_bare_filename_to_relative_path() -> None:
    """裸文件名应被改写成 charts/<文件名>，否则报告里的链接是坏的。"""
    out = sanitize_report("![图](sales.png)", {"sales.png"})
    assert "!(charts/sales.png)" in out or "](charts/sales.png)" in out


def test_sanitize_normalizes_absolute_path() -> None:
    """绝对路径同样应被改写为相对路径，保证报告可移植。"""
    out = sanitize_report("![图](D:/agent/reports/charts/sales.png)", {"sales.png"})
    assert "](charts/sales.png)" in out
    assert "D:/agent" not in out


def test_sanitize_drops_missing_chart_entirely() -> None:
    """引用不存在图表的图片标签必须整条删除，而不是留下打不开的链接。"""
    out = sanitize_report("前文\n\n![图](charts/ghost.png)\n\n后文", {"real.png"})
    assert "ghost.png" not in out
    assert "前文" in out and "后文" in out


def test_sanitize_keeps_valid_link_unchanged() -> None:
    """已经是规范写法的引用不应被无谓改写。"""
    content = "![区域销售额对比](charts/bar_region_sales.png)"
    assert sanitize_report(content, {"bar_region_sales.png"}) == content


# ==========================================================================
# 图表标题解析
# ==========================================================================
def test_chart_title_handles_underscored_column_names() -> None:
    """含下划线的字段名（unit_price）必须靠真实列名正确还原。

    按位置切分会得到 "unit" 与 "price" 两个不存在的字段，标题就错了。
    """
    columns = ["month", "unit_price", "sales"]
    title = chart_title_from_path("line_month_unit_price_20260913_120000.png", columns)
    assert "单价" in title
    assert "unit" not in title and "price" not in title


def test_chart_title_falls_back_without_columns() -> None:
    """不提供列名时退回按位置切分，兼容历史文件名。"""
    assert chart_title_from_path("bar_region_sales_20260908_171346.png") == "区域销售额对比"


def test_chart_title_unknown_column_keeps_original_name() -> None:
    """列名不在中英映射表里时保留原名，不编造中文。"""
    assert "revenue" in chart_title_from_path("bar_region_revenue_20260913_120000.png")


# ==========================================================================
# 报告上下文的图表白名单
# ==========================================================================
def test_context_only_lists_existing_charts() -> None:
    """上下文的图表白名单必须过滤掉磁盘上不存在的文件。"""
    charts_dir = get_settings().charts_dir
    real = charts_dir / "consistency_real.png"
    real.write_bytes(b"")
    try:
        ctx = build_report_context({
            "dataset_metadata": {"columns": ["region", "sales"]},
            "generated_charts": [str(real), str(charts_dir / "consistency_ghost.png")],
        })
        assert ctx["valid_chart_names"] == {"consistency_real.png"}
        assert len(ctx["generated_charts"]) == 1
        # 相对路径必须是 charts/ 前缀，报告放在 reports/ 下才能正确加载
        assert ctx["generated_charts"][0]["relative"] == "charts/consistency_real.png"
    finally:
        real.unlink(missing_ok=True)


def test_context_includes_degradation_reasons() -> None:
    """降级原因必须进入报告上下文，才能被写进「分析局限性」。"""
    ctx = build_report_context({"degraded": ["步数耗尽"]})
    assert ctx["degraded"] == ["步数耗尽"]


# ==========================================================================
# 质量检查的误报控制
# ==========================================================================
def test_validate_accepts_report_containing_json() -> None:
    """报告里包含 JSON 数据（连续右花括号）不应被误判为模板残留。"""
    content = '# 报告\n\n## 核心发现\n```json\n{"a": {"b": 1}}\n```\n'
    ok, issues = validate_report(content, set())
    assert ok, issues


def test_validate_still_flags_jinja_placeholder() -> None:
    """真正的 Jinja 占位符仍应被判为未填充。"""
    ok, issues = validate_report("# 报告\n\n## 核心发现\n{{ core_findings }}", set())
    assert not ok
    assert any("占位符" in i for i in issues)


def test_validate_flags_chart_not_in_whitelist() -> None:
    """引用了白名单之外的图表必须报错（防止「声称生成了不存在的图表」）。"""
    ok, issues = validate_report("# 报告\n\n![x](charts/ghost.png)", {"real.png"})
    assert not ok
    assert any("不存在的图表" in i for i in issues)


def test_full_pipeline_report_references_only_real_charts() -> None:
    """端到端：真实生成图表 → 上下文白名单 → 清洗后报告只引用真实文件。"""
    from src.tools.chart_tool import generate_chart_file

    chart_path = generate_chart_file(SALES_CSV, "bar", "region", "sales")
    try:
        ctx = build_report_context({
            "dataset_metadata": {"columns": ["region", "sales"]},
            "generated_charts": [chart_path, "/nonexistent/fake.png"],
            "tool_calls": [],
            "tool_results": [],
        })
        # 模型「顺手」引用了两张图，其中一张是幻觉出来的
        raw = (
            "# 报告\n\n## 图表\n"
            f"![区域销售额对比]({ctx['generated_charts'][0]['relative']})\n"
            "![幻觉图](charts/fake.png)\n"
        )
        cleaned = sanitize_report(raw, ctx["valid_chart_names"])
        assert "fake.png" not in cleaned
        # 真实图表链接必须能在磁盘上解析到
        referenced = [l for l in cleaned.splitlines() if "](" in l]
        for line in referenced:
            url = line.split("](", 1)[1].rstrip(")")
            assert (get_settings().reports_dir / url).is_file(), url
    finally:
        Path(chart_path).unlink(missing_ok=True)
