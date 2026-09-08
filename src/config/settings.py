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

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM（OpenAI 兼容接口）----
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.0

    # ---- Agent 控制 ----
    max_steps: int = 15
    python_timeout_seconds: int = 30
    tool_timeout_seconds: int = 60
    output_truncate_chars: int = 4000

    # ---- 路径 ----
    datasets_dir: Path = PROJECT_ROOT / "datasets"
    reports_dir: Path = PROJECT_ROOT / "reports"
    charts_dir: Path = PROJECT_ROOT / "reports" / "charts"
    knowledge_dir: Path = PROJECT_ROOT / "knowledge"
    workspace_dir: Path = PROJECT_ROOT / "workspace"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # ---- 基础设施 ----
    database_url: str = "postgresql+psycopg://analyst:analyst@localhost:5432/analyst"
    redis_url: str = "redis://localhost:6379/0"

    # ---- 持久化开关 ----
    db_enabled: bool = True           # 是否启用 PostgreSQL 业务持久化
    redis_enabled: bool = True        # 是否启用 Redis 会话/缓存
    redis_cache_ttl: int = 3600       # Redis 缓存默认 TTL（秒）

    # ---- RAG / 向量检索 ----
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = 384          # 本地哈希回退向量的维度
    rag_top_k: int = 5                # 检索返回条数

    # ---- MCP ----
    mcp_enabled: bool = False         # 是否启用 MCP 工具（MCP_ENABLED=true）

    def ensure_dirs(self) -> None:
        """确保运行所需的目录存在。"""
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
        """将用户输入的文件路径解析为绝对路径（支持相对项目根）。"""
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()


@lru_cache
def get_settings() -> Settings:
    """返回单例配置，并确保目录存在。"""
    settings = Settings()
    settings.ensure_dirs()
    return settings
