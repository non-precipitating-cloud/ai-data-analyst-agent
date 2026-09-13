"""PersistenceService：统一编排业务持久化 + 会话/缓存，全程优雅降级。

数据库/Redis 不可用时，仅记录日志并跳过持久化，绝不中断 Agent 核心分析。
Agent 各环节（启动、计划、工具调用、结果、报告、完成）只与本服务打交道，
由它决定写入 PostgreSQL、Redis 中的哪些位置。
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

# 工具名 → 分析结果类型的映射表：工具调用落库时据此归类写入 analysis_results
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
# 有实际分析价值、需要额外写入 analysis_results 的结果类型集合
_MEANINGFUL_RESULT_TYPES = {"profile", "statistics", "correlation", "outlier", "chart"}


def tool_type_of(name: str) -> str:
    """工具类型：mcp__ 前缀为 MCP，其余为本地。"""
    return "mcp" if name.startswith("mcp__") else "local"


def result_type_of(name: str) -> str:
    """把工具名映射为分析结果类型；未登记的工具统一归类为 other。

    参数:
        name: 工具名，可能带 mcp__ 前缀。
    """
    # 先剥离 MCP 前缀再查映射表，避免为同一工具登记两个名字
    n = name.removeprefix("mcp__")
    return _RESULT_TYPE_MAP.get(n, "other")


def _now() -> datetime:
    """返回带时区的当前 UTC 时间。"""
    return datetime.now(timezone.utc)


class PersistenceService:
    """业务持久化 + Redis 会话/缓存的统一入口。

    内部持有 Repository（数据库）、SessionStore/Cache（Redis）三类组件，
    并缓存数据库与 Redis 的可用性探测结果，实现「不可用即跳过」的降级策略。
    """

    def __init__(self) -> None:
        """初始化各存储组件与本次运行的关联 ID（初始均为空）。"""
        self._repo = Repository()        # 数据库读写
        self._sessions = SessionStore()  # Redis 会话状态
        self._cache = Cache()            # Redis 通用缓存
        self.session_id: str | None = None  # 当前会话 ID
        self.task_id: int | None = None     # 当前分析任务 ID
        self.run_id: int | None = None      # 当前 Agent 运行 ID
        self.dataset_id: int | None = None  # 当前数据集 ID
        self._db_on: bool | None = None     # 数据库可用性缓存（None=未探测）
        self._redis_on: bool | None = None  # Redis 可用性缓存（None=未探测）

    # ---- 可用性 ----
    def _db_ok(self) -> bool:
        """数据库是否可用：同时满足配置开关开启且连通性探测通过（结果缓存）。"""
        if self._db_on is None:
            self._db_on = get_settings().db_enabled and is_db_available()
        return self._db_on

    def _redis_ok(self) -> bool:
        """Redis 是否可用：同时满足配置开关开启且连通性探测通过（结果缓存）。"""
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
        """一次分析运行的起点：生成会话、登记数据集/任务/运行、创建 Redis 会话。

        参数:
            dataset_path: 待分析数据集路径。
            user_request: 用户的自然语言需求。
            session_id: 外部指定的会话 ID；不传则自动生成 UUID。
            dataset_info: 调用方已掌握的数据集画像；没有则内部尝试探测。

        返回:
            包含 session_id / task_id / run_id 的字典（数据库不可用时后两者为 None）。
        """
        # 会话 ID 缺省时随机生成；同时清空上一轮运行的关联 ID
        self.session_id = session_id or str(uuid.uuid4())
        self.task_id = None
        self.run_id = None
        self.dataset_id = None

        if self._db_ok():
            try:
                # 依次写入：数据集 -> 分析任务（running）-> Agent 运行
                self.dataset_id = self._create_dataset(dataset_path, dataset_info)
                self.task_id = self._repo.create_task(
                    self.dataset_id, user_request, status="running"
                )
                self._repo.update_task(self.task_id, started_at=_now())
                self.run_id = self._repo.create_run(self.task_id, self.session_id)
            except Exception as e:  # noqa: BLE001
                # 持久化失败不影响分析本身：仅告警，ID 保持 None
                logger.warning("创建 task/run 失败：%s", e)

        if self._redis_ok():
            self._sessions.create_session(self.session_id, self.task_id, user_request)

        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
        }

    def _create_dataset(self, path: str, info: dict | None) -> int:
        """登记数据集并返回其 ID；缺少数画像时现场尝试探测。"""
        # 函数内延迟导入，规避与 src.tools 之间的循环导入
        from src.tools.file_tools import _profile_dataset

        meta = info or {}
        if not meta.get("num_rows") and not meta.get("columns"):
            # 调用方没给画像信息：尝试读文件自行探测，失败则用空信息兜底
            try:
                meta = _profile_dataset(path)
            except Exception:  # noqa: BLE001
                meta = {}
        # 统一解析为绝对路径后再入库；若路径不在允许的数据目录（越界拦截），
        # 持久化作为旁路能力不应中断主分析流程，退回使用原始路径信息登记
        from pathlib import Path

        try:
            resolved = get_settings().resolve_dataset_path(path)
            file_path, filename, file_type = str(resolved), resolved.name, resolved.suffix.lstrip(".")
        except PermissionError:
            file_path, filename, file_type = str(path), Path(str(path)).name, Path(str(path)).suffix.lstrip(".")
        return self._repo.create_dataset(
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            row_count=meta.get("num_rows", 0),
            column_count=meta.get("num_columns", 0),
            schema_json={"columns": meta.get("columns", []), "dtypes": meta.get("dtypes", {})},
            metadata_json=meta or None,
        )

    # ---- 运行期 ----
    def update_status(self, status: str, step: int | None = None) -> None:
        """把当前运行状态（及步数）实时更新到 Redis 会话，供外部查询进度。"""
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
        """记录 Agent 制定的分析计划（及选中的技能列表）到任务记录。"""
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
        """记录一次工具调用；对有价值的分析工具同时写一条 analysis_results。

        参数:
            name: 工具名。
            tool_type: 工具类型（local/mcp）。
            arguments: 调用参数字典。
            result: 工具结果文本（会按配置长度截断）。
            status: 调用状态。
            error: 失败时的错误信息。
        """
        if not self._db_ok():
            return
        try:
            # 结果文本过长时截断，避免单行过大、撑爆模型上下文
            truncated = (result or "")[: get_settings().output_truncate_chars]
            self._repo.add_tool_call(
                self.run_id, name, tool_type, arguments, truncated, status, error
            )
            # 仅画像/统计/相关性/异常值/图表类工具额外沉淀为结构化分析结果
            rt = result_type_of(name)
            if rt in _MEANINGFUL_RESULT_TYPES:
                self._repo.add_result(
                    self.task_id, rt, {"tool": name, "summary": truncated[:1000]}
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("记录工具调用失败：%s", e)

    def record_result(self, result_type: str, content: dict | None) -> None:
        """直接记录一条结构化分析结果（非工具触发的场景使用）。"""
        if not self._db_ok() or self.task_id is None:
            return
        try:
            self._repo.add_result(self.task_id, result_type, content)
        except Exception as e:  # noqa: BLE001
            logger.warning("记录分析结果失败：%s", e)

    def record_report(self, report_path: str | None, content: str | None) -> None:
        """记录最终报告（文件路径与/或正文）到数据库。"""
        if not self._db_ok() or self.task_id is None:
            return
        try:
            self._repo.add_report(self.task_id, report_path, content)
        except Exception as e:  # noqa: BLE001
            logger.warning("记录报告失败：%s", e)

    # ---- 完成 ----
    def complete(self, status: str = "completed", error: str | None = None, step_count: int | None = None) -> None:
        """收尾：把任务与运行置为终态，并同步 Redis 会话状态。

        参数:
            status: 终态，completed 或 failed。
            error: 失败原因；成功时为 None。
            step_count: 本次运行总步数，可选。
        """
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
                pass  # Redis 收尾失败无需处理，静默忽略

    # ---- 查询 ----
    @property
    def repo(self) -> Repository:
        """暴露内部 Repository，供需要直接查询历史记录的调用方使用。"""
        return self._repo
