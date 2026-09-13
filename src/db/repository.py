"""Repository 层：封装数据库读写操作，统一使用短生命周期 Session。

业务代码只调用 Repository 的方法，不直接接触 SQLAlchemy Session 与查询语句，
便于集中管理 SQL、隔离持久化细节，也方便以后替换存储实现。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from src.db.database import session_scope
from src.db.models import (
    AgentRun,
    AnalysisResult,
    AnalysisTask,
    Dataset,
    LLMCall,
    Report,
    ToolCall,
)


def _now() -> datetime:
    """返回带时区的当前 UTC 时间，用于手动补时间戳。"""
    return datetime.now(timezone.utc)


class Repository:
    """统一数据访问对象：业务代码不直接接触 SQLAlchemy Session。

    每个方法内部通过 session_scope() 开启独立短事务，方法返回即提交关闭。
    依赖引擎配置的 expire_on_commit=False，返回的 ORM 对象在 Session
    关闭后仍可读取已加载的属性。
    """

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
        """登记一个数据集；同 file_path 已存在时幂等返回旧记录 ID。

        参数:
            filename: 原始文件名。
            file_path: 文件路径（作为去重键）。
            file_type: 文件类型后缀。
            row_count: 数据行数。
            column_count: 数据列数。
            schema_json: 列结构信息。
            metadata_json: 其他画像元信息。

        返回:
            数据集记录的主键 ID（新建或已有记录）。
        """
        with session_scope() as s:
            # 幂等去重：同一文件路径不重复入库
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
            # flush：先把 INSERT 发送到数据库以拿到自增主键，但事务尚未提交
            s.flush()
            return ds.id

    def create_task(
        self, dataset_id: int | None, user_request: str, status: str = "pending"
    ) -> int:
        """创建一条分析任务记录。

        参数:
            dataset_id: 关联数据集 ID，可为 None。
            user_request: 用户的自然语言分析需求。
            status: 初始状态，默认 pending。

        返回:
            新建任务的主键 ID。
        """
        with session_scope() as s:
            task = AnalysisTask(
                dataset_id=dataset_id, user_request=user_request, status=status
            )
            s.add(task)
            s.flush()
            return task.id

    def create_run(self, task_id: int | None, session_id: str | None) -> int:
        """创建一条 Agent 运行记录（初始状态 running）。

        参数:
            task_id: 关联任务 ID，可为 None。
            session_id: 对应的会话标识，可为 None。

        返回:
            新建运行记录的主键 ID。
        """
        with session_scope() as s:
            run = AgentRun(task_id=task_id, session_id=session_id, status="running")
            s.add(run)
            s.flush()
            return run.id

    # ---- 更新 ----
    def update_task(self, task_id: int, **fields) -> None:
        """按主键更新任务字段；任务不存在时静默跳过。

        参数:
            task_id: 要更新的任务 ID。
            **fields: 字段名=新值的任意键值对；当 status 变为
                completed/failed 时自动补 completed_at 时间戳。
        """
        with session_scope() as s:
            task = s.get(AnalysisTask, task_id)
            if task is None:
                return  # 目标记录不存在，直接返回不报错
            # 动态把传入的字段写到 ORM 对象上，由 Session 跟踪生成 UPDATE
            for k, v in fields.items():
                setattr(task, k, v)
            # 进入终态时自动补完成时间
            if "status" in fields and fields["status"] in ("completed", "failed"):
                task.completed_at = _now()

    def update_run(self, run_id: int, **fields) -> None:
        """按主键更新运行记录字段；记录不存在时静默跳过。

        参数:
            run_id: 要更新的运行 ID。
            **fields: 字段名=新值；status 变为 completed/failed 时自动补 completed_at。
        """
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
        duration_ms: int = 0,
    ) -> int:
        """插入一条工具调用记录。

        参数:
            run_id: 所属运行 ID，可为 None。
            tool_name: 工具名称。
            tool_type: 工具类型（local/mcp）。
            arguments: 调用参数字典（存 JSON）。
            result: 已截断的结果字符串。
            status: 调用状态（success/error 等）。
            error_message: 失败时的错误信息。
            duration_ms: 本次调用耗时（毫秒），用于定位慢工具。

        返回:
            新建工具调用记录的主键 ID。
        """
        with session_scope() as s:
            tc = ToolCall(
                run_id=run_id,
                tool_name=tool_name,
                tool_type=tool_type,
                arguments_json=arguments,
                result_json=result,
                status=status,
                error_message=error_message,
                duration_ms=duration_ms,
            )
            s.add(tc)
            s.flush()
            return tc.id

    def add_llm_call(
        self,
        run_id: int | None,
        node: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        duration_ms: int,
        attempts: int,
        status: str,
        error_message: str | None = None,
    ) -> int:
        """插入一条大模型调用记录。

        参数:
            run_id: 所属运行 ID，可为 None。
            node: 发起调用的节点名。
            model: 模型名。
            input_tokens / output_tokens / total_tokens: 接口返回的真实用量；
                接口未提供时传 None（不估算）。
            duration_ms: 本次调用耗时（含重试等待）。
            attempts: 实际尝试次数。
            status: success / failed。
            error_message: 失败原因。

        返回:
            新建记录的主键 ID。
        """
        with session_scope() as s:
            row = LLMCall(
                run_id=run_id,
                node=node,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
                attempts=attempts,
                status=status,
                error_message=error_message,
            )
            s.add(row)
            s.flush()
            return row.id

    def add_result(self, task_id: int | None, result_type: str, content: dict | None) -> int:
        """插入一条结构化分析结果。

        参数:
            task_id: 关联任务 ID，可为 None。
            result_type: 结果类型（profile/statistics/...）。
            content: 结果内容字典（存 JSON）。

        返回:
            新建结果记录的主键 ID。
        """
        with session_scope() as s:
            r = AnalysisResult(task_id=task_id, result_type=result_type, content_json=content)
            s.add(r)
            s.flush()
            return r.id

    def add_report(self, task_id: int | None, report_path: str | None, content: str | None) -> int:
        """插入一条报告记录（文件路径与正文至少其一）。

        参数:
            task_id: 关联任务 ID，可为 None。
            report_path: 报告文件路径，可为 None。
            content: 报告正文字符串，可为 None。

        返回:
            新建报告记录的主键 ID。
        """
        with session_scope() as s:
            r = Report(task_id=task_id, report_path=report_path, report_content=content)
            s.add(r)
            s.flush()
            return r.id

    # ---- 查询 ----
    def get_task(self, task_id: int) -> AnalysisTask | None:
        """按主键查询分析任务；不存在返回 None。"""
        with session_scope() as s:
            return s.get(AnalysisTask, task_id)

    def get_run(self, run_id: int) -> AgentRun | None:
        """按主键查询 Agent 运行记录；不存在返回 None。"""
        with session_scope() as s:
            return s.get(AgentRun, run_id)

    def get_run_by_session(self, session_id: str) -> AgentRun | None:
        """按会话 ID 查询最近一次运行记录；无记录返回 None。

        参数:
            session_id: 会话标识。
        """
        with session_scope() as s:
            # 同一会话可能有多条运行记录，取 ID 最大（最新）的一条
            return s.execute(
                select(AgentRun)
                .where(AgentRun.session_id == session_id)
                .order_by(AgentRun.id.desc())
            ).scalars().first()

    def list_tool_calls(self, run_id: int) -> list[ToolCall]:
        """查询某次运行下的全部工具调用，按调用先后（ID 升序）排列。"""
        with session_scope() as s:
            return list(
                s.execute(
                    select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.id)
                ).scalars()
            )

    def list_llm_calls(self, run_id: int) -> list[LLMCall]:
        """查询某次运行下的全部 LLM 调用，按调用先后（ID 升序）排列。"""
        with session_scope() as s:
            return list(
                s.execute(
                    select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.id)
                ).scalars()
            )

    def list_results(self, task_id: int) -> list[AnalysisResult]:
        """查询某任务下的全部结构化分析结果，按产生先后（ID 升序）排列。"""
        with session_scope() as s:
            return list(
                s.execute(
                    select(AnalysisResult)
                    .where(AnalysisResult.task_id == task_id)
                    .order_by(AnalysisResult.id)
                ).scalars()
            )

    def get_report(self, task_id: int) -> Report | None:
        """查询某任务最新的一份报告；无报告返回 None。"""
        with session_scope() as s:
            # 一个任务可能生成多版报告，取最新一条
            return s.execute(
                select(Report).where(Report.task_id == task_id).order_by(Report.id.desc())
            ).scalars().first()
