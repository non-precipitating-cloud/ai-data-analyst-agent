"""Agent 循环测试：终止条件、去重、上下文裁剪、LLM 故障降级。

对应「Agent 核心逻辑」一节要求：
- 循环必须有明确的终止条件，不能只靠步数上限兜底；
- 重复调用不应重复执行，也不能让循环空转；
- 上下文不能随步数线性膨胀（否则 token 与稳定性都成问题）；
- 单次 LLM 调用失败不能让整张图崩溃，要有可解释的降级路径。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.graph import route_after_agent
from src.agent.nodes.insight import insight_node
from src.agent.nodes.report import report_node
from src.agent.nodes.tool_calling import (
    agent_node,
    call_signature,
    executed_signatures,
    is_chart_tool,
    tools_node,
)
from src.agent.observability import AgentLLMError
from src.agent.utils import trim_messages_for_llm
from src.config.settings import get_settings
from tests.conftest import SALES_CSV


def _ai(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    """构造携带单个工具调用的 AI 消息。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ==========================================================================
# 路由终止条件
# ==========================================================================
def test_route_without_tool_calls_goes_to_insight() -> None:
    """没有工具调用时直接进入洞察，属于正常收敛。"""
    state = {"messages": [AIMessage(content="结论")], "step_count": 1, "max_steps": 15}
    assert route_after_agent(state) == "insight"


def test_route_stops_at_max_steps() -> None:
    """达到步数上限时强制结束循环，防止死循环。"""
    state = {"messages": [_ai("read_dataset", {"path": SALES_CSV})], "step_count": 15, "max_steps": 15}
    assert route_after_agent(state) == "insight"


def test_route_stops_on_no_progress() -> None:
    """本轮请求的调用全都成功执行过时，判定「无进展」并结束循环。

    这是比步数上限更早、更省资源的终止条件：再跑一次也拿不到任何新信息。
    """
    args = {"path": SALES_CSV}
    state = {
        "messages": [_ai("read_dataset", args)],
        "step_count": 2,
        "max_steps": 15,
        "tool_calls": [{"name": "read_dataset", "args": args, "status": "success"}],
    }
    assert route_after_agent(state) == "insight"


def test_route_allows_retry_after_failure() -> None:
    """失败过的调用不算「已执行」，模型必须还有机会改参数重试。"""
    args = {"path": SALES_CSV}
    state = {
        "messages": [_ai("read_dataset", args)],
        "step_count": 2,
        "max_steps": 15,
        "tool_calls": [{"name": "read_dataset", "args": args, "status": "error"}],
        "tool_results": [{"name": "read_dataset", "result": "错误[x]", "status": "error"}],
    }
    assert route_after_agent(state) == "tools"


def test_route_continues_when_only_some_calls_are_duplicates() -> None:
    """只要有一个调用是新的，就应继续执行工具，而不是整体终止。"""
    done_args = {"path": SALES_CSV}
    new_args = {"path": SALES_CSV, "column": "sales"}
    state = {
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "read_dataset", "args": done_args, "id": "a"},
            {"name": "calculate_statistics", "args": new_args, "id": "b"},
        ])],
        "step_count": 2,
        "max_steps": 15,
        "tool_calls": [{"name": "read_dataset", "args": done_args, "status": "success"}],
    }
    assert route_after_agent(state) == "tools"


def test_call_signature_is_key_order_insensitive() -> None:
    """参数键顺序不同应视为同一个调用（否则去重会失效）。"""
    assert call_signature("t", {"a": 1, "b": 2}) == call_signature("t", {"b": 2, "a": 1})


def test_executed_signatures_only_counts_success() -> None:
    """只有成功过的调用才算「已执行」，失败的不计入。"""
    state = {
        "tool_calls": [
            {"name": "a", "args": {}, "status": "success"},
            {"name": "b", "args": {}, "status": "error"},
        ]
    }
    sigs = executed_signatures(state)
    assert call_signature("a", {}) in sigs
    assert call_signature("b", {}) not in sigs


# ==========================================================================
# 同一步内的去重
# ==========================================================================
def test_tools_node_reuses_result_for_same_step_duplicate() -> None:
    """同一步里重复请求同一调用时，第二条复用首次结果而不重复执行。"""
    calls = [
        {"name": "calculate_statistics", "args": {"path": SALES_CSV, "column": "sales"}, "id": "a"},
        {"name": "calculate_statistics", "args": {"path": SALES_CSV, "column": "sales"}, "id": "b"},
    ]
    update = tools_node({"messages": [AIMessage(content="", tool_calls=calls)]})
    assert len(update["tool_results"]) == 2
    assert "已直接返回首次结果" in update["tool_results"][1]["result"]
    # 第一次是真实执行的结果，两次内容都应是合法 JSON 统计结果
    assert "mean" in update["tool_results"][0]["result"]


def test_chart_tool_detection_covers_mcp_variant() -> None:
    """MCP 版图表工具也必须被识别，否则报告会误报「未生成图表」。"""
    assert is_chart_tool("generate_chart")
    assert is_chart_tool("mcp__generate_chart")
    assert not is_chart_tool("read_dataset")


# ==========================================================================
# 上下文裁剪
# ==========================================================================
def test_trim_folds_old_tool_results_but_keeps_recent() -> None:
    """较早的工具结果应被折叠为占位，最近的若干条保留原文。"""
    messages = [SystemMessage(content="sys"), HumanMessage(content="req")]
    for i in range(10):
        messages.append(ToolMessage(content=f"结果{i}" * 50, tool_call_id=f"t{i}", name="tool"))
    trimmed = trim_messages_for_llm(messages, keep_recent_tool_results=3)

    assert len(trimmed) == len(messages)
    # 系统与用户消息原样保留
    assert trimmed[0].content == "sys"
    assert trimmed[1].content == "req"
    # 最后 3 条工具结果保留原文
    for m in trimmed[-3:]:
        assert "已折叠" not in m.content
    # 更早的被折叠，且体积显著下降
    folded = [m for m in trimmed if "已折叠" in m.content]
    assert len(folded) == 7
    assert sum(len(m.content) for m in folded) < 100 * len(folded)


def test_trim_is_noop_when_few_results() -> None:
    """工具结果条数未超阈值时不改动消息，避免无谓的重建。"""
    messages = [SystemMessage(content="s"), ToolMessage(content="x", tool_call_id="1")]
    assert trim_messages_for_llm(messages, keep_recent_tool_results=6) == messages


def test_trim_preserves_ai_decisions() -> None:
    """AI 决策消息必须完整保留，否则模型会丢失自己的推理链。"""
    messages = [
        SystemMessage(content="s"),
        ToolMessage(content="old" * 100, tool_call_id="0", name="t"),
        AIMessage(content="我决定先看地区分布"),
        ToolMessage(content="new", tool_call_id="1", name="t"),
    ]
    trimmed = trim_messages_for_llm(messages, keep_recent_tool_results=1)
    assert any(
        isinstance(m, AIMessage) and m.content == "我决定先看地区分布" for m in trimmed
    )


# ==========================================================================
# LLM 故障降级
# ==========================================================================
class _FailingLLM:
    """所有调用都抛错的假模型。"""

    model_name = "always-failing"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        return self

    def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN201
        raise RuntimeError("模拟模型服务不可用")


class _WorkingLLM:
    """返回固定消息的假模型（用于验证降级路径之外的正常路径）。"""

    model_name = "working"

    def __init__(self, content: str = "好的") -> None:
        self.content = content

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        return self

    def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN201
        return AIMessage(content=self.content)


def test_agent_node_degrades_instead_of_crashing(monkeypatch) -> None:
    """LLM 不可用时 agent_node 不应抛出中断整张图，而要留下来降级标记。"""
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: _FailingLLM())
    monkeypatch.setattr(get_settings(), "agent_llm_retries", 0)

    out = agent_node({"messages": [HumanMessage(content="分析")], "step_count": 0})
    # 不带 tool_calls → 路由会进入 insight，而不是继续循环
    assert not getattr(out["messages"][0], "tool_calls", None)
    assert out["degraded"]
    assert out["errors"]
    assert out["llm_calls"][0]["success"] is False


def test_agent_node_records_llm_usage(monkeypatch) -> None:
    """正常调用时也要产出可观测记录（节点名/模型/耗时）。"""
    monkeypatch.setattr("src.agent.nodes.tool_calling.get_llm", lambda: _WorkingLLM())
    out = agent_node({"messages": [HumanMessage(content="分析")], "step_count": 0})
    record = out["llm_calls"][0]
    assert record["success"] is True
    assert record["node"] == "agent"
    assert record["model"] == "working"
    assert record["duration_ms"] >= 0


def test_insight_node_falls_back_when_llm_unavailable(monkeypatch) -> None:
    """洞察阶段 LLM 不可用时，应给出「未生成洞察」的确定性兜底，而不是留空。"""
    monkeypatch.setattr("src.agent.nodes.insight.get_llm", lambda: _FailingLLM())
    monkeypatch.setattr(get_settings(), "agent_llm_retries", 0)

    out = insight_node({
        "user_request": "分析",
        "messages": [HumanMessage(content="x")],
        "tool_results": [{"name": "read_dataset", "result": "{}", "status": "success"}],
        "tool_calls": [{"name": "read_dataset", "args": {}, "status": "success"}],
    })
    text = out["insights"][0]
    assert "未能生成" in text
    assert "read_dataset" in text  # 如实列出成功的工具
    assert out["degraded"]


def test_report_node_produces_deterministic_report_when_llm_unavailable(monkeypatch) -> None:
    """报告阶段 LLM 不可用时，仍应落盘一份「数据可追溯」的降级报告。"""
    from pathlib import Path

    monkeypatch.setattr("src.agent.nodes.report.get_llm", lambda: _FailingLLM())
    monkeypatch.setattr(get_settings(), "agent_llm_retries", 0)

    out = report_node({
        "user_request": "分析销售额",
        "dataset_metadata": {"num_rows": 2160, "num_columns": 12, "columns": ["sales"]},
        "tool_calls": [{"name": "read_dataset", "args": {}, "status": "success"}],
        "tool_results": [{"name": "read_dataset", "result": '{"num_rows":2160}', "status": "success"}],
        "generated_charts": [],
        "insights": [],
        "observations": [],
        "selected_skills": [],
    })
    assert out["status"] == "done"
    assert "降级输出" in out["final_report"]
    # 真实数据必须原样保留在报告里
    assert "2160" in out["final_report"]
    assert out["degraded"]
    Path(out["report_path"]).unlink(missing_ok=True)


def test_agent_llm_error_is_raised_after_retries(monkeypatch) -> None:
    """invoke_llm 在重试耗尽后必须抛出 AgentLLMError，供上层区分处理。"""
    from src.agent.observability import invoke_llm
    import pytest

    monkeypatch.setattr(get_settings(), "agent_llm_retries", 1)
    with pytest.raises(AgentLLMError):
        invoke_llm([HumanMessage(content="x")], llm=_FailingLLM(), node="test")
