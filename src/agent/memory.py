"""Agent 连续对话记忆：让「为什么？」这类追问能接上上一轮的分析。

要解决的两个问题（方向相反，必须同时处理）：

1. **要记得住**：用户先问「分析销售额下降原因」，Agent 答「华东下降最多」，
   接着只问一句「为什么？」。如果新一轮不带任何历史，Agent 根本不知道
   「为什么」指代什么，只能重新从头分析，既慢又答非所问。
2. **不能记太多**：把全部历史一股脑塞进上下文，会让无关任务的旧结论污染
   当前分析（例如刚分析完财务数据，又切回来问销售，模型可能混用两组数字），
   并且迅速耗尽上下文预算。

因此这里的策略是「**按数据集隔离 + 只保留最近若干轮 + 只注入结论摘要**」：
- 只取与当前数据集**相同**的历史轮次（`dataset_path` 归一化后比较），
  换了数据集就自动视为全新任务，历史不参与；
- 只保留最近 `conversation_max_turns` 轮（配置项，默认 3）；
- 每轮只存「用户问题 + 结论摘要 + 报告路径」，不存完整报告与工具原始输出，
  总长度再按 `conversation_max_chars` 截断。

存储上复用现有基础设施：优先写入 Redis 会话文档（随 session 持久化，
进程重启后仍可延续对话），Redis 不可用时自动退化为进程内字典，
不额外引入依赖。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.utils import truncate
from src.config.settings import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

# Redis 会话文档中存放历史轮次的字段名
_TURNS_FIELD = "conversation_turns"


def _normalize_dataset(dataset_path: str) -> str:
    """把数据集路径归一化，供历史轮次做「同一数据集」判断。

    用户可能一次输入 ``datasets/sales.csv``，下一次输入绝对路径或
    ``./datasets/sales.csv``——它们指向同一个文件，应被视为同一数据集。
    归一化失败（路径不存在等）时退回字符串本身，保证比较不会抛异常。

    :param dataset_path: 用户提供的数据集路径
    :return: 归一化后的路径字符串
    """
    try:
        p = Path(dataset_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return str(p.resolve())
    except (OSError, ValueError):
        return str(dataset_path)


class ConversationMemory:
    """会话级连续对话记忆（Redis 优先，进程内字典兜底）。

    用法::

        memory = ConversationMemory(session_id)
        context = memory.build_context(request, dataset_path)   # 取历史
        ... 运行 Agent ...
        memory.remember(request, dataset_path, insights, report_path)  # 存本轮
    """

    def __init__(self, session_id: str | None = None) -> None:
        """
        Args:
            session_id: 会话标识；Redis 可用时作为存储 key。
                为 None 时只使用进程内存储（不跨进程持久化）。
        """
        self.session_id = session_id
        # 进程内兜底存储：Redis 不可用（或未给 session_id）时的唯一存储
        self._local: list[dict[str, Any]] = []

    # ---- 读 ----
    def _load(self) -> list[dict[str, Any]]:
        """读取全部历史轮次（Redis 优先，取不到则用进程内列表）。"""
        if not self.session_id:
            return list(self._local)
        try:
            # 延迟导入：memory 模块不应在导入期就依赖 Redis 连接
            from src.cache import SessionStore

            session = SessionStore().get_session(self.session_id) or {}
            turns = session.get(_TURNS_FIELD)
            if isinstance(turns, list):
                return turns
        except Exception as e:  # noqa: BLE001 —— 读取失败即降级，不影响分析
            logger.warning("读取对话历史失败，改用进程内记录：%s", e)
        return list(self._local)

    def _save(self, turns: list[dict[str, Any]]) -> None:
        """写回全部历史轮次（Redis 优先，失败则只写进程内列表）。"""
        # 无论 Redis 是否可用都更新进程内副本，保证同一进程内行为一致
        self._local = turns
        if not self.session_id:
            return
        try:
            from src.cache import SessionStore

            SessionStore().update_session(self.session_id, **{_TURNS_FIELD: turns})
        except Exception as e:  # noqa: BLE001
            logger.warning("写入对话历史失败，仅保留在进程内：%s", e)

    # ---- 写 ----
    def remember(
        self,
        request: str,
        dataset_path: str,
        insights: list[str] | str | None = None,
        report_path: str = "",
    ) -> None:
        """记录本轮问答，供后续追问复用。

        只保存「问题 + 结论摘要 + 报告路径」：完整报告与工具原始输出体积过大，
        且对理解上下文帮助有限，没有必要长期携带。

        :param request: 用户本轮的分析需求
        :param dataset_path: 本轮分析的数据集路径
        :param insights: 本轮结论（洞察文本列表或单个字符串）
        :param report_path: 本轮报告落盘路径
        """
        settings = get_settings()
        if isinstance(insights, str):
            summary_text = insights
        else:
            summary_text = "\n".join(insights or [])
        turn = {
            "request": request,
            "dataset": _normalize_dataset(dataset_path),
            "dataset_display": dataset_path,
            # 摘要再截断一次，防止单轮结论过长挤占后续所有轮次的预算
            "summary": truncate(summary_text, max(settings.memory_max_chars // 2, 500)),
            "report_path": report_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        turns = self._load()
        turns.append(turn)
        # 只保留最近 N 轮，避免历史无限增长
        self._save(turns[-settings.memory_max_turns:])

    # ---- 组装上下文 ----
    def build_context(self, request: str, dataset_path: str) -> str:
        """为当前问题组装历史上下文文本。

        返回空串表示「这是一次全新的任务」，此时 Agent 不会看到任何历史，
        与改造前的行为完全一致。

        :param request: 当前用户需求（当前实现不参与过滤，保留参数以便未来
            做「仅当问题指代不明时才注入历史」这类优化）
        :param dataset_path: 当前数据集路径；只有同一数据集的历史才会被注入
        :return: 供注入首轮消息的历史上下文文本；无可用历史时为空串
        """
        settings = get_settings()
        current = _normalize_dataset(dataset_path)
        turns = [t for t in self._load() if t.get("dataset") == current]
        if not turns:
            return ""
        # 取最近 N 轮
        turns = turns[-settings.memory_max_turns:]

        lines = [
            "【历史对话上下文】以下是同一数据集上此前的分析，"
            "可用于理解「为什么」「那它呢」这类省略式追问："
        ]
        for i, t in enumerate(turns, start=1):
            lines.append(
                f"{i}. 用户此前提问：{t.get('request', '')}\n"
                f"   当时得出的结论摘要：{t.get('summary', '（无）')}"
            )
        lines.append(
            "【使用要求】若当前问题是对上述结论的追问，请在此基础上继续深入；"
            "若当前问题与历史无关，请忽略上述内容，独立完成本次分析。"
            "历史中的数字仅供参考，最终结论必须来自本次工具调用的真实结果。"
        )
        return truncate("\n".join(lines), settings.memory_max_chars)

    def clear(self) -> None:
        """清空历史（用户显式切换任务/数据集时调用）。"""
        self._save([])
