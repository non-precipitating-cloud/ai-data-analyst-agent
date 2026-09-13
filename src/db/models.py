"""SQLAlchemy ORM 模型模块：定义业务持久化的六张表。

表之间构成一条完整的分析溯源链：

Dataset（数据集）
  └─ AnalysisTask（分析任务：用户的一次自然语言分析请求）
       ├─ AgentRun（Agent 运行：一次任务可对应多次运行/续跑）
       │    └─ ToolCall（工具调用：运行过程中每一步工具执行的记录）
       ├─ AnalysisResult（分析产出：画像/统计/相关性/异常值/图表/洞察）
       └─ Report（最终分析报告）

这些类既是 create_all 建表的依据，也是 CRUD 操作所使用的 ORM 实体。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """返回带时区的当前 UTC 时间，作为时间字段 default 的回调函数。"""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """所有 ORM 模型的声明基类；其子类都会注册进同一套 metadata，供统一建表。"""

    pass


class Dataset(Base):
    """数据集表：记录被分析文件的基本信息与结构概要。"""

    __tablename__ = "datasets"

    # 主键，自增整数
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 原始文件名（不含目录）
    filename: Mapped[str] = mapped_column(String(512))
    # 文件在磁盘上的路径
    file_path: Mapped[str] = mapped_column(Text)
    # 文件类型后缀（csv / xlsx / json 等）
    file_type: Mapped[str] = mapped_column(String(32))
    # 数据行数，默认 0
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    # 数据列数，默认 0
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    # 列结构信息（列名、数据类型等），以 JSON 存储；允许为空
    schema_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # 其他画像元信息，以 JSON 存储；允许为空
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # 创建时间：插入行时自动取 UTC 当前时间（default 传函数引用，不能传调用结果）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisTask(Base):
    """分析任务表：一次自然语言分析请求的主记录，串联数据集与全部产出。"""

    __tablename__ = "analysis_tasks"

    # 主键，自增整数
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 关联的数据集 ID（外键 -> datasets.id）；允许为空表示未绑定具体文件
    dataset_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("datasets.id"), nullable=True
    )
    # 用户提出的原始分析需求（自然语言）
    user_request: Mapped[str] = mapped_column(Text)
    # 任务状态：pending/running/completed/failed 等
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # Agent 生成的分析计划，以 JSON 存储；允许为空
    analysis_plan_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # 如 selected_skills
    # 任务创建时间（UTC）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # 实际开始执行时间；未开始时为空
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 完成时间；进行中为空
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 失败时的错误信息；成功时为空
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentRun(Base):
    """Agent 运行表：任务的一次具体执行（同一任务可能多次运行/续跑）。"""

    __tablename__ = "agent_runs"

    # 主键，自增整数
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 关联的分析任务 ID（外键 -> analysis_tasks.id）；允许为空
    task_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("analysis_tasks.id"), nullable=True
    )
    # 会话标识（Redis 会话与本次运行的对应键）；允许为空
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 运行状态：running/completed/failed 等
    status: Mapped[str] = mapped_column(String(32), default="running")
    # 本次运行累计执行的步数
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    # 运行开始时间（UTC）
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # 运行结束时间；进行中为空
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 失败时的错误信息
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ToolCall(Base):
    """工具调用表：记录 Agent 运行过程中每一次工具调用的入参与结果。"""

    __tablename__ = "tool_calls"

    # 主键，自增整数
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 所属 Agent 运行 ID（外键 -> agent_runs.id）；允许为空
    run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("agent_runs.id"), nullable=True
    )
    # 被调用工具的名称
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_type: Mapped[str] = mapped_column(String(16), default="local")  # local / mcp
    # 调用参数（键值对），以 JSON 存储；允许为空
    arguments_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # 结果字符串（截断）
    # 调用结果状态：success/failed 等
    status: Mapped[str] = mapped_column(String(16), default="success")
    # 调用失败时的错误信息
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 调用开始时间（UTC）
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # 调用结束时间（UTC）
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisResult(Base):
    """分析结果表：工具/分析步骤产出的结构化结论。"""

    __tablename__ = "analysis_results"

    # 主键，自增整数
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 关联的分析任务 ID（外键 -> analysis_tasks.id）；允许为空
    task_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("analysis_tasks.id"), nullable=True
    )
    result_type: Mapped[str] = mapped_column(String(32))  # profile/statistics/correlation/outlier/chart/insight/other
    # 结果内容（结构随结果类型而不同），以 JSON 存储；允许为空
    content_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # 记录创建时间（UTC）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Report(Base):
    """报告表：分析任务最终生成的报告（文件路径或正文内容）。"""

    __tablename__ = "reports"

    # 主键，自增整数
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 关联的分析任务 ID（外键 -> analysis_tasks.id）；允许为空
    task_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("analysis_tasks.id"), nullable=True
    )
    # 报告文件在磁盘上的路径（落盘时使用）；允许为空
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 报告正文内容（直接入库时使用）；允许为空
    report_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 报告创建时间（UTC）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
