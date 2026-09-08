"""Repository 层：封装数据库读写，统一短生命周期 Session。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from src.db.database import session_scope
from src.db.models import (
    AgentRun,
    AnalysisResult,
    AnalysisTask,
    Dataset,
    Report,
    ToolCall,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Repository:
    """统一 Repository：业务代码不直接接触 SQLAlchemy Session。"""

    # ---- 创建 ----
    def create_dataset(
        self,
        filename: str,
        file_path: str,
        file_type: str,
        row_count: int = 0,
        column_count: int = 0,
        schema_json: dict | None = None,
        metadata_json: dict | None = None,
    ) -> int:
        with session_scope() as s:
            existing = s.execute(
                select(Dataset).where(Dataset.file_path == file_path)
            ).scalar_one_or_none()
            if existing is not None:
                return existing.id
            ds = Dataset(
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                row_count=row_count,
                column_count=column_count,
                schema_json=schema_json,
                metadata_json=metadata_json,
            )
            s.add(ds)
            s.flush()
            return ds.id

    def create_task(
        self, dataset_id: int | None, user_request: str, status: str = "pending"
    ) -> int:
        with session_scope() as s:
            task = AnalysisTask(
                dataset_id=dataset_id, user_request=user_request, status=status
            )
            s.add(task)
            s.flush()
            return task.id

    def create_run(self, task_id: int | None, session_id: str | None) -> int:
        with session_scope() as s:
            run = AgentRun(task_id=task_id, session_id=session_id, status="running")
            s.add(run)
            s.flush()
            return run.id

    # ---- 更新 ----
    def update_task(self, task_id: int, **fields) -> None:
        with session_scope() as s:
            task = s.get(AnalysisTask, task_id)
            if task is None:
                return
            for k, v in fields.items():
                setattr(task, k, v)
            if "status" in fields and fields["status"] in ("completed", "failed"):
                task.completed_at = _now()

    def update_run(self, run_id: int, **fields) -> None:
        with session_scope() as s:
            run = s.get(AgentRun, run_id)
            if run is None:
                return
            for k, v in fields.items():
                setattr(run, k, v)
            if "status" in fields and fields["status"] in ("completed", "failed"):
                run.completed_at = _now()

    def add_tool_call(
        self,
        run_id: int | None,
        tool_name: str,
        tool_type: str,
        arguments: dict | None,
        result: str | None,
        status: str,
        error_message: str | None = None,
    ) -> int:
        with session_scope() as s:
            tc = ToolCall(
                run_id=run_id,
                tool_name=tool_name,
                tool_type=tool_type,
                arguments_json=arguments,
                result_json=result,
                status=status,
                error_message=error_message,
            )
            s.add(tc)
            s.flush()
            return tc.id

    def add_result(self, task_id: int | None, result_type: str, content: dict | None) -> int:
        with session_scope() as s:
            r = AnalysisResult(task_id=task_id, result_type=result_type, content_json=content)
            s.add(r)
            s.flush()
            return r.id

    def add_report(self, task_id: int | None, report_path: str | None, content: str | None) -> int:
        with session_scope() as s:
            r = Report(task_id=task_id, report_path=report_path, report_content=content)
            s.add(r)
            s.flush()
            return r.id

    # ---- 查询 ----
    def get_task(self, task_id: int) -> AnalysisTask | None:
        with session_scope() as s:
            return s.get(AnalysisTask, task_id)

    def get_run(self, run_id: int) -> AgentRun | None:
        with session_scope() as s:
            return s.get(AgentRun, run_id)

    def get_run_by_session(self, session_id: str) -> AgentRun | None:
        with session_scope() as s:
            return s.execute(
                select(AgentRun)
                .where(AgentRun.session_id == session_id)
                .order_by(AgentRun.id.desc())
            ).scalars().first()

    def list_tool_calls(self, run_id: int) -> list[ToolCall]:
        with session_scope() as s:
            return list(
                s.execute(
                    select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.id)
                ).scalars()
            )

    def list_results(self, task_id: int) -> list[AnalysisResult]:
        with session_scope() as s:
            return list(
                s.execute(
                    select(AnalysisResult)
                    .where(AnalysisResult.task_id == task_id)
                    .order_by(AnalysisResult.id)
                ).scalars()
            )

    def get_report(self, task_id: int) -> Report | None:
        with session_scope() as s:
            return s.execute(
                select(Report).where(Report.task_id == task_id).order_by(Report.id.desc())
            ).scalars().first()
