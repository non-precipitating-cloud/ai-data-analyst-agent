"""Redis Session：保存 Agent 会话状态（JSON）。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from src.cache.redis_client import get_redis, is_redis_available

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Agent 会话状态存储。key: agent:session:{session_id}。"""

    def __init__(self, prefix: str = "agent:session") -> None:
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    def create_session(
        self, session_id: str, task_id=None, user_request: str = ""
    ) -> dict:
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
        raw = self._get(session_id)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def update_session(self, session_id: str, **fields) -> dict | None:
        existing = self.get_session(session_id) or {"session_id": session_id}
        existing.update(fields)
        existing["updated_at"] = _now_iso()
        self._set(session_id, existing)
        return existing

    def delete_session(self, session_id: str) -> bool:
        if not is_redis_available():
            return False
        try:
            get_redis().delete(self._key(session_id))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 删除 session 失败：%s", e)
            return False

    def _set(self, session_id: str, data: dict) -> bool:
        if not is_redis_available():
            return False
        try:
            get_redis().set(self._key(session_id), json.dumps(data, ensure_ascii=False))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 写入 session 失败：%s", e)
            return False

    def _get(self, session_id: str) -> str | None:
        if not is_redis_available():
            return None
        try:
            return get_redis().get(self._key(session_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis 读取 session 失败：%s", e)
            return None
