"""PersistenceService：统一编排业务持久化 + 会话/缓存，全程优雅降级。

数据库/Redis 不可用时，仅记录日志并跳过持久化，绝不中断 Agent 核心分析。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from src.cache import Cache, SessionStore
from src.cache.redis_client import is_redis_available
from src.config.settings import get_settings
from src.db.database import is_db_available
from src.db.repository import Repository

logger = logging.getLogger(__name__)

# 工具名 → 结果类型
_RESULT_TYPE_MAP = {
    "profile_dataset": "profile",
    "read_dataset": "profile",
    "inspect_schema": "profile",
    "get_schema": "profile",
    "calculate_statistics": "statistics",
    "calculate_correlation": "correlation",
    "detect_outliers": "outlier",
    "generate_chart": "chart",
}
_MEANINGFUL_RESULT_TYPES = {"profile", "statistics", "correlation", "outlier", "chart"}


def tool_type_of(name: str) -> str:
    """工具类型：mcp__ 前缀为 MCP，其余为本地。"""
    return "mcp" if name.startswith("mcp__") else "local"


def result_type_of(name: str) -> str:
    n = name.removeprefix("mcp__")
    return _RESULT_TYPE_MAP.get(n, "other")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PersistenceService:
    """业务持久化 + Redis 会话/缓存的统一入口。"""

    def __init__(self) -> None:
        self._repo = Repository()
        self._sessions = SessionStore()
        self._cache = Cache()
        self.session_id: str | None = None
        self.task_id: int | None = None
        self.run_id: int | None = None
        self.dataset_id: int | None = None
        self._db_on: bool | None = None
        self._redis_on: bool | None = None

    # ---- 可用性 ----
    def _db_ok(self) -> bool:
        if self._db_on is None:
            self._db_on = get_settings().db_enabled and is_db_available()
        return self._db_on

    def _redis_ok(self) -> bool:
        if self._redis_on is None:
            self._redis_on = get_settings().redis_enabled and is_redis_available()
        return self._redis_on

    # ---- 启动 ----
    def start_run(
        self,
        dataset_path: str,
        user_request: str,
        session_id: str | None = None,
        dataset_info: dict | None = None,
    ) -> dict:
        self.session_id = session_id or str(uuid.uuid4())
        self.task_id = None
        self.run_id = None
        self.dataset_id = None

        if self._db_ok():
            try:
                self.dataset_id = self._create_dataset(dataset_path, dataset_info)
                self.task_id = self._repo.create_task(
                    self.dataset_id, user_request, status="running"
                )
                self._repo.update_task(self.task_id, started_at=_now())
                self.run_id = self._repo.create_run(self.task_id, self.session_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("创建 task/run 失败：%s", e)

        if self._redis_ok():
            self._sessions.create_session(self.session_id, self.task_id, user_request)

        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
        }

    def _create_dataset(self, path: str, info: dict | None) -> int:
        from src.tools.file_tools import _profile_dataset

        meta = info or {}
        if not meta.get("num_rows") and not meta.get("columns"):
            try:
                meta = _profile_dataset(path)
            except Exception:  # noqa: BLE001
                meta = {}
        resolved = get_settings().resolve_dataset_path(path)
        return self._repo.create_dataset(
            filename=resolved.name,
            file_path=str(resolved),
            file_type=resolved.suffix.lstrip("."),
            row_count=meta.get("num_rows", 0),
            column_count=meta.get("num_columns", 0),
            schema_json={"columns": meta.get("columns", []), "dtypes": meta.get("dtypes", {})},
            metadata_json=meta or None,
        )

    # ---- 运行期 ----
    def update_status(self, status: str, step: int | None = None) -> None:
        if not self._redis_ok() or not self.session_id:
            return
        try:
            fields = {"current_status": status}
            if step is not None:
                fields["current_step"] = step
            self._sessions.update_session(self.session_id, **fields)
        except Exception as e:  # noqa: BLE001
            logger.warning("更新 session 状态失败：%s", e)

    def record_plan(self, plan: list | dict | None, skills: list[str] | None = None) -> None:
        if not self._db_ok() or self.task_id is None:
            return
        try:
            fields: dict = {"analysis_plan_json": plan}
            if skills is not None:
                fields["metadata_json"] = {"selected_skills": skills}
            self._repo.update_task(self.task_id, **fields)
        except Exception as e:  # noqa: BLE001
            logger.warning("记录分析计划失败：%s", e)

    def record_tool_call(
        self,
        name: str,
        tool_type: str,
        arguments: dict | None,
        result: str | None,
        status: str,
        error: str | None = None,
    ) -> None:
        if not self._db_ok():
            return
        try:
            truncated = (result or "")[: get_settings().output_truncate_chars]
            self._repo.add_tool_call(
                self.run_id, name, tool_type, arguments, truncated, status, error
            )
            rt = result_type_of(name)
            if rt in _MEANINGFUL_RESULT_TYPES:
                self._repo.add_result(
                    self.task_id, rt, {"tool": name, "summary": truncated[:1000]}
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("记录工具调用失败：%s", e)

    def record_result(self, result_type: str, content: dict | None) -> None:
        if not self._db_ok() or self.task_id is None:
            return
        try:
            self._repo.add_result(self.task_id, result_type, content)
        except Exception as e:  # noqa: BLE001
            logger.warning("记录分析结果失败：%s", e)

    def record_report(self, report_path: str | None, content: str | None) -> None:
        if not self._db_ok() or self.task_id is None:
            return
        try:
            self._repo.add_report(self.task_id, report_path, content)
        except Exception as e:  # noqa: BLE001
            logger.warning("记录报告失败：%s", e)

    # ---- 完成 ----
    def complete(self, status: str = "completed", error: str | None = None, step_count: int | None = None) -> None:
        if self._db_ok():
            try:
                if self.task_id is not None:
                    self._repo.update_task(self.task_id, status=status, error_message=error)
                if self.run_id is not None:
                    fields: dict = {"status": status, "error_message": error}
                    if step_count is not None:
                        fields["step_count"] = step_count
                    self._repo.update_run(self.run_id, **fields)
            except Exception as e:  # noqa: BLE001
                logger.warning("更新 task/run 完成状态失败：%s", e)
        if self._redis_ok() and self.session_id:
            try:
                self._sessions.update_session(self.session_id, current_status=status)
            except Exception:  # noqa: BLE001
                pass

    # ---- 查询 ----
    @property
    def repo(self) -> Repository:
        return self._repo
