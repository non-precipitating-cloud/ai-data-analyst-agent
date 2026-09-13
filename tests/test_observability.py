"""可观测性测试：LLM 用量、工具耗时、运行级汇总。

对应「Observability」与「Token 使用」两节要求：
- 一次 Agent 运行要能被完整追踪：run_id / session_id / 模型 / 节点 /
  token / 耗时 / 成败 / 错误；
- token 用量只记录接口**真实返回**的值，拿不到就留空，
  **绝不估算、绝不换算金额**（凭空乘单价会产出看似精确实则不可信的成本）。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.observability import extract_usage, invoke_llm
from src.config.settings import get_settings
from src.db.service import PersistenceService
from tests.conftest import SALES_CSV

_DATASET_INFO = {"row_count": 2160, "column_count": 12, "columns": ["sales", "region"]}


# ==========================================================================
# 用量提取
# ==========================================================================
def test_extract_usage_from_usage_metadata() -> None:
    """LangChain 标准化的 usage_metadata 应被完整取出。"""
    msg = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    usage = extract_usage(msg)
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_extract_usage_from_openai_response_metadata() -> None:
    """部分 OpenAI 兼容服务只在 response_metadata 里给用量，同样要能取到。"""
    msg = AIMessage(
        content="x",
        response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    )
    usage = extract_usage(msg)
    assert usage["total_tokens"] == 10


def test_extract_usage_returns_empty_when_unavailable() -> None:
    """接口没返回用量时返回空字典，**不补零**——零会被误读为「没消耗」。"""
    assert extract_usage(AIMessage(content="x")) == {}


def test_invoke_llm_records_usage_and_duration() -> None:
    """一次成功调用应产出含节点名、模型、耗时与用量的记录。"""

    class _LLM:
        model_name = "test-model"

        def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN201
            return AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )

    _, record = invoke_llm([HumanMessage(content="hi")], llm=_LLM(), node="planner")
    assert record["node"] == "planner"
    assert record["model"] == "test-model"
    assert record["total_tokens"] == 5
    assert record["success"] is True
    assert record["duration_ms"] >= 0


def test_invoke_llm_retries_transient_failure(monkeypatch) -> None:
    """瞬时故障（如限流）应被重试，最终成功。"""
    monkeypatch.setattr(get_settings(), "agent_llm_retries", 2)
    calls = {"n": 0}

    class _FlakyLLM:
        model_name = "flaky"

        def invoke(self, messages, **kwargs):  # noqa: ANN001, ANN201
            calls["n"] += 1
            if calls["n"] == 1:
                # 用类名匹配瞬时故障（与 observability._TRANSIENT_EXCEPTION_NAMES 对应）
                raise type("RateLimitError", (Exception,), {})("429 too many requests")
            return AIMessage(content="ok")

    response, record = invoke_llm([HumanMessage(content="x")], llm=_FlakyLLM(), node="agent")
    assert response.content == "ok"
    assert calls["n"] == 2
    assert record["attempts"] == 2


# ==========================================================================
# 落库
# ==========================================================================
def test_llm_calls_are_persisted(sqlite_db, fake_redis) -> None:
    """LLM 调用应逐条落库，并在运行记录上汇总。"""
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)

    summary = svc.record_llm_calls([
        {"node": "planner", "model": "m", "total_tokens": 100, "duration_ms": 50,
         "success": True, "attempts": 1},
        {"node": "agent", "model": "m", "total_tokens": 250, "duration_ms": 80,
         "success": True, "attempts": 1},
        {"node": "insight", "model": "m", "success": False, "error": "boom"},
    ])

    assert summary["calls"] == 3
    assert summary["failed"] == 1
    assert summary["total_tokens"] == 350

    rows = svc.repo.list_llm_calls(ids["run_id"])
    assert len(rows) == 3
    assert rows[0].node == "planner"
    assert rows[0].total_tokens == 100
    assert rows[2].status == "failed"
    assert rows[2].error_message == "boom"
    # 未返回用量的记录保持 NULL，而不是 0
    assert rows[2].total_tokens is None


def test_run_summary_written_on_complete(sqlite_db, fake_redis) -> None:
    """complete 时把调用次数/失败次数/token 合计写入 agent_runs。"""
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)
    summary = svc.record_llm_calls([
        {"node": "agent", "model": "m", "total_tokens": 42, "duration_ms": 10, "success": True},
    ])
    svc.complete("completed", step_count=3, llm_calls=summary)

    run = svc.repo.get_run(ids["run_id"])
    assert run.llm_call_count == 1
    assert run.llm_failed_count == 0
    assert run.total_tokens == 42
    assert run.duration_ms >= 0


def test_run_summary_keeps_tokens_null_when_unavailable(sqlite_db, fake_redis) -> None:
    """接口始终没给用量时，运行级 total_tokens 保持 NULL，而不是被写成 0。"""
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)
    summary = svc.record_llm_calls([{"node": "agent", "model": "m", "success": True}])
    svc.complete("completed", llm_calls=summary)

    assert svc.repo.get_run(ids["run_id"]).total_tokens is None


def test_tool_call_duration_persisted(sqlite_db, fake_redis) -> None:
    """工具调用耗时应落库，便于事后定位慢工具。"""
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)
    svc.record_tool_call(
        "execute_python", "local", {"code": "..."}, "out", "success", duration_ms=1234
    )
    rows = svc.repo.list_tool_calls(ids["run_id"])
    assert rows[0].duration_ms == 1234


def test_failed_tool_call_records_error_message(sqlite_db, fake_redis) -> None:
    """失败的调用要把错误摘要放进 error_message，便于按错误检索。"""
    svc = PersistenceService()
    ids = svc.start_run(SALES_CSV, "分析", dataset_info=_DATASET_INFO)
    svc.record_tool_call(
        "execute_sql", "local", {"sql": "bad"},
        "错误[invalid_argument|retryable]：no such column",
        "error", error="错误[invalid_argument|retryable]：no such column",
    )
    rows = svc.repo.list_tool_calls(ids["run_id"])
    assert rows[0].status == "error"
    assert "invalid_argument" in rows[0].error_message
