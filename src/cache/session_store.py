"""Redis Session：保存 Agent 会话状态（JSON 文本，整存整取）。

会话状态记录一次分析任务的执行进度（当前步骤、状态、用户原始需求等），
以 ``agent:session:{session_id}`` 为 key 存入 Redis，使任务可跨请求恢复、查询。
与 Cache 的区别：会话不设 TTL（长期保留直到删除），且提供读-改-写的
update 语义；Redis 不可用时同样静默降级，不中断 Agent 流程。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from src.cache.redis_client import get_redis, is_redis_available

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串，作为会话的 updated_at 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Agent 会话状态存储。key: agent:session:{session_id}。

    会话以一个 JSON 文档整体读写，字段可随业务自由扩展。
    """

    def __init__(self, prefix: str = "agent:session") -> None:
        """初始化会话存储。

        Args:
            prefix: Redis key 前缀，最终 key 形如 ``{prefix}:{session_id}``。
        """
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        """根据会话 ID 拼出完整 Redis key。"""
        return f"{self.prefix}:{session_id}"

    def create_session(
        self, session_id: str, task_id=None, user_request: str = ""
    ) -> dict:
        """创建新会话并写入初始状态。

        Args:
            session_id: 会话唯一标识。
            task_id: 关联的任务 ID（可为 None）。
            user_request: 用户的原始分析需求。

        Returns:
            写入 Redis 的完整初始会话字典。
        """
        # 初始状态：created、步骤号 0，并记录创建/更新时间
        data = {
            "session_id": session_id,
            "task_id": task_id,
            "current_status": "created",
            "current_step": 0,
            "user_request": user_request,
            "updated_at": _now_iso(),
        }
        self._set(session_id, data)
        return data

    def get_session(self, session_id: str) -> dict | None:
        """读取会话并反序列化。

        Args:
            session_id: 会话唯一标识。

        Returns:
            会话字典；会话不存在、Redis 不可用或 JSON 损坏时返回 None。
        """
        raw = self._get(session_id)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # 存储内容不是合法 JSON（脏数据）时按“无会话”处理
            return None

    def update_session(self, session_id: str, **fields) -> dict | None:
        """读-改-写更新会话字段，并刷新 updated_at。

        Args:
            session_id: 会话唯一标识。
            **fields: 需要合并更新的任意字段（如 current_status、current_step）。

        Returns:
            更新后的完整会话字典。
        """
        # 会话不存在时以最小骨架新建，保证更新操作幂等
        existing = self.get_session(session_id) or {"session_id": session_id}
        existing.update(fields)
        # 每次更新都刷新时间戳，便于追踪会话最近活动时间
        existing["updated_at"] = _now_iso()
        self._set(session_id, existing)
        return existing

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话。

        Args:
            session_id: 会话唯一标识。

        Returns:
            True 表示删除命令执行成功；Redis 不可用或异常时返回 False。
        """
        if not is_redis_available():
            return False
        try:
            get_redis().delete(self._key(session_id))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 删除 session 失败：%s", e)
            return False

    def _set(self, session_id: str, data: dict) -> bool:
        """把会话字典序列化为 JSON 写入 Redis（内部方法）。

        Returns:
            True 表示写入成功；Redis 不可用或异常时返回 False。
        """
        if not is_redis_available():
            return False
        try:
            # 会话不设置 TTL：长期保留，直到显式删除
            get_redis().set(self._key(session_id), json.dumps(data, ensure_ascii=False))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 写入 session 失败：%s", e)
            return False

    def _get(self, session_id: str) -> str | None:
        """读取会话原始 JSON 文本（内部方法，反序列化由调用方完成）。

        Returns:
            Redis 中存储的 JSON 字符串；不存在、不可用或异常时返回 None。
        """
        if not is_redis_available():
            return None
        try:
            return get_redis().get(self._key(session_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 读取 session 失败：%s", e)
            return None
