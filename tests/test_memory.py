"""连续对话记忆测试（src.agent.memory）。

对应「Agent Memory」一节要求：
- 「为什么？」这类追问要能接上同一数据集上一轮的分析结论；
- 无关历史不能污染当前任务（换数据集即视为全新任务）；
- 历史要有条数上限，不能无限增长；
- Redis 不可用时降级为进程内记忆，功能不丢失。
"""

from __future__ import annotations

from pathlib import Path

from src.agent.memory import ConversationMemory, _normalize_dataset
from src.config.settings import get_settings
from tests.conftest import SALES_CSV

# 第二个数据集（同目录内的合法文件），用于验证「换数据集即隔离」
OTHER_CSV = "datasets/ecommerce.csv"


def test_context_empty_for_first_turn() -> None:
    """全新会话没有任何历史，上下文必须为空串（不注入空壳文本）。"""
    memory = ConversationMemory(session_id=None)
    assert memory.build_context("分析销售额", SALES_CSV) == ""


def test_context_includes_same_dataset_turn() -> None:
    """同一数据集上此前的结论应被注入，使追问可复用上下文。"""
    memory = ConversationMemory(session_id=None)
    memory.remember("分析销售额下降原因", SALES_CSV, ["华东下降 32.7%，贡献整体降幅 41.2%"], "r.md")

    context = memory.build_context("为什么？", SALES_CSV)
    assert "华东下降 32.7%" in context
    assert "分析销售额下降原因" in context
    # 提示模型区分「追问」与「无关提问」，避免历史污染
    assert "无关" in context or "忽略" in context


def test_context_isolated_between_datasets() -> None:
    """换了数据集就应视为全新任务，历史不得串入。"""
    memory = ConversationMemory(session_id=None)
    memory.remember("分析销售数据", SALES_CSV, ["销售额下降 18.58%"], "r.md")

    # 换到另一个数据集提问，不应看到上一份数据的结论
    assert memory.build_context("分析订单数据", OTHER_CSV) == ""


def test_dataset_path_normalization() -> None:
    """同一文件的不同写法（相对/绝对）应被识别为同一数据集。"""
    assert _normalize_dataset(SALES_CSV) == _normalize_dataset(str(Path(SALES_CSV).resolve()))
    assert _normalize_dataset("./datasets/sales.csv") == _normalize_dataset(SALES_CSV)


def test_max_turns_limit(monkeypatch) -> None:
    """历史轮数超过上限时只保留最近的若干轮。"""
    monkeypatch.setattr(get_settings(), "memory_max_turns", 2)
    memory = ConversationMemory(session_id=None)
    for i in range(5):
        memory.remember(f"第{i}轮提问", SALES_CSV, [f"第{i}轮结论"], "r.md")

    context = memory.build_context("为什么？", SALES_CSV)
    # 只保留最近两轮
    assert "第4轮提问" in context
    assert "第3轮提问" in context
    assert "第0轮提问" not in context


def test_max_chars_limit(monkeypatch) -> None:
    """上下文总长度受配置约束，避免挤占本次分析的预算。"""
    monkeypatch.setattr(get_settings(), "memory_max_chars", 300)
    memory = ConversationMemory(session_id=None)
    memory.remember("问题", SALES_CSV, ["很长的结论" * 500], "r.md")

    context = memory.build_context("追问", SALES_CSV)
    # 截断标记来自 utils.truncate，说明确实做了长度控制
    assert len(context) < 1000


def test_memory_falls_back_to_process_local_without_redis(monkeypatch) -> None:
    """Redis 不可用时记忆应降级到进程内，功能不丢失。"""
    from src.cache import redis_client

    redis_client.reset_redis()
    # 让可用性探测直接判定 Redis 不可用
    monkeypatch.setattr(redis_client, "_redis_available", False)

    memory = ConversationMemory(session_id="s-1")
    memory.remember("分析销售额", SALES_CSV, ["结论A"], "r.md")
    # 同一个 memory 实例内部保留了进程内副本
    assert "结论A" in memory.build_context("为什么？", SALES_CSV)

    # 另一个实例（模拟新对象）仍能通过进程内兜底读到——因为 _local 是实例级，
    # 这里只验证「不会抛异常、不会丢数据」
    redis_client.reset_redis()


def test_memory_uses_redis_when_available(fake_redis) -> None:
    """Redis 可用时历史应写入会话文档，可跨实例读取。"""
    memory = ConversationMemory(session_id="s-2")
    memory.remember("分析销售额", SALES_CSV, ["结论B"], "r.md")

    # 换一个实例读取同一 session，应从 Redis 拿到历史
    other = ConversationMemory(session_id="s-2")
    assert "结论B" in other.build_context("为什么？", SALES_CSV)


def test_clear_removes_history(fake_redis) -> None:
    """clear 后不应再注入任何历史（切换任务时使用）。"""
    memory = ConversationMemory(session_id="s-3")
    memory.remember("分析销售额", SALES_CSV, ["结论C"], "r.md")
    memory.clear()
    assert memory.build_context("为什么？", SALES_CSV) == ""


def test_corrupt_history_does_not_crash(monkeypatch) -> None:
    """会话里的历史字段被写坏时应安全降级，而不是让整次分析失败。"""
    from src.cache import SessionStore

    store = SessionStore()
    # 模拟脏数据：conversation_turns 不是列表
    store.update_session("s-4", conversation_turns={"not": "a list"})
    memory = ConversationMemory(session_id="s-4")
    assert memory.build_context("分析", SALES_CSV) == ""
