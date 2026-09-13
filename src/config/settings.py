"""全局配置。

所有配置项都通过环境变量 / .env 文件注入，禁止在代码中硬编码密钥。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 = src/config/settings.py 的上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """应用配置（字段名与 env 变量大小写不敏感映射）。"""

    # Pydantic Settings 的加载规则配置
    model_config = SettingsConfigDict(
        # 指定 .env 文件位置（项目根目录下）
        env_file=str(PROJECT_ROOT / ".env"),
        # .env 文件按 UTF-8 编码读取
        env_file_encoding="utf-8",
        # .env 中出现代码未声明的变量时直接忽略，而不是校验报错
        extra="ignore",
        # 环境变量名与字段名大小写不敏感（LLM_API_KEY 与 llm_api_key 等价）
        case_sensitive=False,
    )

    # ---- LLM（OpenAI 兼容接口）----
    # 大模型 API Key；默认空串，实际运行必须由环境变量/.env 注入，禁止硬编码
    llm_api_key: str = ""
    # OpenAI 兼容接口基础 URL，替换为其他厂商地址即可切换模型
    llm_base_url: str = "https://api.deepseek.com/v1"
    # 默认调用的模型名称
    llm_model: str = "deepseek-chat"
    # 采样温度：0 表示输出尽量确定、可复现，适合数据分析场景
    llm_temperature: float = 0.0
    # LLM HTTP 调用超时（秒）；超时后由内置重试机制处理
    llm_timeout_seconds: float = 60.0
    # LLM 瞬时错误（HTTP 429/5xx、连接抖动）自动重试次数（指数退避）
    llm_max_retries: int = 3

    # ---- Agent 控制 ----
    # 单次分析允许的最大推理/行动步数，防止 Agent 陷入死循环
    max_steps: int = 15
    # Python 代码执行工具的超时时间（秒）
    python_timeout_seconds: int = 30
    # 其他工具调用的超时时间（秒）。超时后该次调用返回超时错误给 LLM，
    # 由模型决定换参数或换工具，而不是让整个 Agent 挂死。
    tool_timeout_seconds: int = 60
    # 工具结果落库/回传给模型前的最大截断字符数，防止上下文超长
    output_truncate_chars: int = 4000
    # 同一工具**连续失败**达到此次数后，Agent 会收到「停止重试该工具」的
    # 明确指令，避免把全部步数耗在同一个错误上
    tool_max_consecutive_failures: int = 3
    # 单步（一次 LLM 决策）允许并行发起的工具调用数量上限
    max_tool_calls_per_step: int = 6
    # 单次 LLM 调用失败后的额外重试次数（针对超时/连接类瞬时故障）
    agent_llm_retries: int = 2
    # 回灌给 LLM 的历史工具结果保留条数；更早的结果会被替换为摘要占位，
    # 避免 15 步循环后把全部工具输出重复塞进每次请求（token 与稳定性问题）
    keep_recent_tool_results: int = 6
    # 数据文件体积上限（字节）。pandas 会整表读入内存，超大文件直接拒绝
    max_dataset_bytes: int = 512 * 1024 * 1024

    # ---- 连续对话（Agent Memory）----
    # 注入到新一轮分析的最近历史轮数上限（防止无关历史污染当前任务）
    memory_max_turns: int = 3
    # 历史上下文注入的最大字符数
    memory_max_chars: int = 4000
    # 除 datasets 目录外额外允许读取数据文件的根目录（环境变量传 JSON 数组），
    # 例如 Docker 中挂载外部数据卷：EXTRA_DATA_DIRS='["/data"]'
    # （变量名 = 字段名大写，写成 DATA_EXTRA_DIRS 会因 extra="ignore" 被静默忽略）
    extra_data_dirs: list[str] = []

    # ---- 路径（均基于 PROJECT_ROOT，可被同名环境变量覆盖）----
    datasets_dir: Path = PROJECT_ROOT / "datasets"              # 用户数据集存放目录
    reports_dir: Path = PROJECT_ROOT / "reports"                # 分析报告输出目录
    charts_dir: Path = PROJECT_ROOT / "reports" / "charts"      # 图表输出目录（报告子目录）
    knowledge_dir: Path = PROJECT_ROOT / "knowledge"            # RAG 知识库文档目录
    workspace_dir: Path = PROJECT_ROOT / "workspace"            # 代码执行等临时工作目录
    logs_dir: Path = PROJECT_ROOT / "logs"                      # 日志文件目录

    # ---- 基础设施 ----
    # SQLAlchemy 数据库连接串：postgresql+psycopg 表示用 psycopg3 驱动连 PostgreSQL
    database_url: str = "postgresql+psycopg://analyst:analyst@localhost:5432/analyst"
    # Redis 连接串，末尾 /0 表示使用 0 号逻辑库
    redis_url: str = "redis://localhost:6379/0"

    # ---- 持久化开关 ----
    db_enabled: bool = True           # 是否启用 PostgreSQL 业务持久化
    redis_enabled: bool = True        # 是否启用 Redis 会话/缓存
    redis_cache_ttl: int = 3600       # Redis 缓存默认 TTL（秒）

    # ---- RAG / 向量检索 ----
    # 文本向量化模型名（OpenAI 兼容的 embedding 接口）
    embedding_model: str = "text-embedding-3-small"
    # 向量化服务的基础 URL；为空时通常回退到 LLM_BASE_URL
    embedding_base_url: str = ""
    # 向量化服务的 API Key；留空则回退到离线词法向量（非语义）
    embedding_api_key: str = ""
    # Embedding 实现选择：
    #   auto（默认）= 有 EMBEDDING_API_KEY 用真实语义，否则回退离线词法并告警
    #   openai      = 强制真实语义（缺 Key 直接报错，避免误以为在语义检索）
    #   hash        = 强制离线词法回退（CI / 完全离线环境，检索仅字面匹配）
    #   test        = 测试替身，仅供测试代码显式指定
    embedding_provider: str = "auto"
    embedding_dim: int = 384          # 本地词法回退向量的维度
    rag_top_k: int = 5                # 检索返回条数

    # ---- MCP ----
    mcp_enabled: bool = False         # 是否启用 MCP 工具（MCP_ENABLED=true）

    def ensure_dirs(self) -> None:
        """确保运行所需的目录存在（不存在则递归创建，已存在不报错）。"""
        for d in (
            self.datasets_dir,
            self.reports_dir,
            self.charts_dir,
            self.knowledge_dir,
            self.workspace_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def resolve_dataset_path(self, path: str | Path) -> Path:
        """将用户输入的文件路径解析为绝对路径，并做目录边界安全校验。

        相对路径一律拼到项目根目录；绝对路径也必须位于 datasets 目录
        （或 extra_data_dirs 声明的额外数据根目录）之内。这样可以阻断
        ``../../.env``、``/etc/passwd`` 等路径穿越/任意文件读取尝试，
        避免沙箱借数据加载通道把密钥、系统文件读入 df 后回显。

        参数:
            path: 用户提供的数据集路径，可以是绝对路径或相对项目根的相对路径。

        返回:
            规范化（消除 .. 与软链）后的绝对 Path 对象。

        异常:
            PermissionError: 解析后的路径不在任何允许的数据根目录内。
        """
        p = Path(path)
        if not p.is_absolute():
            # 相对路径一律拼到项目根目录下，避免受进程当前工作目录影响
            p = PROJECT_ROOT / p
        p = p.resolve()

        # 允许的数据根：默认 datasets 目录 + 配置声明的外部数据目录
        allowed_roots = [self.datasets_dir.resolve()]
        allowed_roots.extend(Path(d).expanduser().resolve() for d in self.extra_data_dirs)

        # p 必须是某个允许根的子路径（relative_to 失败说明发生了越界）
        for root in allowed_roots:
            try:
                p.relative_to(root)
            except ValueError:
                continue
            return p

        # 所有允许根都不包含该路径：拒绝访问，提示中只回显根目录，不泄露系统结构
        raise PermissionError(
            f"路径越界：仅允许访问数据目录 {self.datasets_dir} 内的文件"
            + ("（或已配置的额外数据目录）" if self.extra_data_dirs else "")
        )


@lru_cache
def get_settings() -> Settings:
    """返回全局唯一的配置实例（带 lru_cache 单例缓存），并确保目录存在。

    返回:
        已加载环境变量/.env 的 Settings 单例；重复调用拿到同一对象。
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings
