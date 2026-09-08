# AI Data Analyst Agent

一个在本地 CLI 运行的数据分析智能体。用户提供 CSV / Excel / JSON 数据文件 + 自然语言需求，
Agent 自主规划、调用工具执行真实分析、根据中间结果循环决策，最终生成 Markdown 分析报告。

> 当前状态：**Phase 2–11 已完成**。核心链路「任务理解 → Skill 选择 → 数据画像 → 规划 →
> 工具循环 → 洞察 → 报告」已跑通，含 Python 沙箱、SQL 只读守卫、RAG 知识库检索（pgvector）、
> 可复用 Agent Skills、MCP（stdio）、PostgreSQL 业务落库 + Redis 会话/缓存（优雅降级）、
> 结构化可信报告（图表真实性校验 + 质量检查）。

---

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows (Git Bash)
# Linux/macOS: source .venv/bin/activate && pip install -e ".[dev]"

# 2. 配置 LLM（OpenAI 兼容接口：DeepSeek / Qwen / Kimi / GLM / OpenAI 等）
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY（必要时改 LLM_BASE_URL / LLM_MODEL）

# 3. 生成示例数据（已内置，可跳过；如需重新生成）
./.venv/Scripts/python.exe scripts/generate_sample_data.py

# 4. 运行
./.venv/Scripts/python.exe main.py
```

交互示例：

```
请输入数据文件路径：
> datasets/sales.csv

请输入你的分析需求：
> 分析今年销售额下降的主要原因，找出表现最差的地区和产品，检测异常数据，并生成完整报告。
```

## 示例数据

| 文件 | 说明 |
|---|---|
| `datasets/sales.csv` | 2160 行 × 12 列，月度×地区×产品销售，含 2025 下滑、华东大跌、异常值、销售-利润强相关 |
| `datasets/ecommerce.csv` | 4000 行订单，含退换货 / 评分 / 支付方式 |
| `datasets/financial.csv` | 720 行，部门×科目收入/支出 |

## 运行测试

```bash
./.venv/Scripts/python.exe -m pytest
```

## 项目结构

```
ai-data-analyst-agent/
├── main.py                  # 入口
├── pyproject.toml
├── .env.example
├── scripts/generate_sample_data.py
├── scripts/ingest_knowledge.py   # 知识库导入
├── datasets/                # 示例数据
├── knowledge/               # 知识库 Markdown 文档（7 篇）
├── reports/                 # 报告 + charts/ 图表
├── docker-compose.yml       # PostgreSQL(pgvector) + Redis
├── docker/init.sql
├── src/
│   ├── config/settings.py   # pydantic-settings 配置
│   ├── logging_config.py
│   ├── llm/factory.py       # OpenAI 兼容 ChatModel 工厂
│   ├── agent/               # LangGraph：state / prompts / graph / nodes
│   ├── tools/               # 文件 / Python 沙箱 / SQL 只读 / 统计 / 图表 / RAG / 报告
│   ├── rag/                 # embeddings / 向量存储 / loader / retriever
│   ├── skills/              # Agent Skills：*/SKILL.md + loader/selector
│   ├── mcp/                 # MCP Server（server.py）+ Client/Adapter（client.py）
│   ├── db/                  # SQLAlchemy 模型 / repository / 持久化 service
│   ├── cache/               # Redis 会话 / 缓存
│   └── cli/runner.py        # 交互式 CLI + 进度展示
└── tests/
```

## 已实现的安全限制

- **Python 沙箱**：子进程运行于 workspace、AST 模块白名单、禁用 `eval/exec/open/os.system` 等、
  禁用双下划线属性（阻断逃逸）、超时 + 输出截断。
- **SQL 只读**：仅允许 `SELECT / WITH / EXPLAIN`，静态拦截 `DROP/DELETE/UPDATE/INSERT/ALTER` 等，
  拒绝多语句。
- **Agent 防死循环**：`max_steps` 上限 + 工具失败不中断 + 错误回传 LLM 自纠。

## RAG（知识库检索）

完整链路：`knowledge/*.md` → 文档加载 → 文本切分 → Embedding → 向量存储 → Retriever → LLM。

- **向量存储**：PostgreSQL + pgvector（真实）；不可用时自动回退到内存实现（离线可跑、可测试）。
- **Embedding**：配置 `EMBEDDING_API_KEY` 用真实语义向量；留空用本地确定性哈希向量。
- **工具**：`retrieve_knowledge` 已加入 Agent 工具集，Agent 自主决定何时检索方法论知识。
- **导入**：`./.venv/Scripts/python.exe scripts/ingest_knowledge.py`（pgvector 时持久化；内存时提示不持久化）。
- **启动基础设施**：`docker compose up -d`（pgvector + Redis）。

## Agent Skills

可复用、可选择、可扩展的专业分析技能，把「怎么分析」从 Prompt 中独立出来。

- **目录**：`src/skills/<slug>/SKILL.md`，含 7 个技能（数据清洗 / 探索性分析 / 销售分析 /
  财务分析 / 异常检测 / 相关性分析 / 报告生成）。
- **Loader**：自动发现并解析 SKILL.md（frontmatter 元数据 + 正文），返回 `SkillMetadata`。
- **Selector**：LLM 语义选择（优先）+ 关键词匹配（回退），只把相关 Skill 注入上下文，不全部塞入。
- **协同**：Skill 提供「分析方法/流程/规则」，RAG 提供「方法论知识」，Tools 负责「实际执行」。
- **流程**：任务理解 → **Skill 选择** → 数据画像 → 规划（遵循 Skill 方法论）→ 工具循环 → 报告。

## MCP（Model Context Protocol）

把数据分析能力以标准化协议（stdio）对外暴露，Agent 通过 MCP Client 动态发现并调用。

- **MCP Server**（`src/mcp/server.py`）：暴露 7 个工具 `read_dataset / get_schema / execute_sql /
  run_analysis / detect_outliers / generate_chart / retrieve_knowledge`，复用现有底层实现（沙箱/只读 SQL 不变）。
- **MCP Client/Adapter**（`src/mcp/client.py`）：持久化 stdio 连接，`list_tools()` 动态发现，
  `build_mcp_tools()` 把 MCP 工具适配为 LangChain 工具（名称加 `mcp__` 前缀，与普通工具区分）。
- **启用**：`.env` 设 `MCP_ENABLED=true`。默认关闭。
- **启动 Server（手动调试）**：`.\.venv\Scripts\python.exe -m src.mcp.server`

两种工具的区分：

| 类型 | 链路 | 示例 |
|---|---|---|
| 普通 LangChain Tool | Agent → 本地函数 → 执行 | `execute_python` |
| MCP Tool | Agent → MCP Client → MCP Server → Tool → Result | `mcp__read_dataset` |

## 持久化（PostgreSQL + Redis）

业务数据落库 + 会话/缓存，全程优雅降级（DB/Redis 不可用时仅告警，不中断 Agent）。

- **PostgreSQL**（`src/db/`）：SQLAlchemy 2.0，6 张表 `datasets / analysis_tasks / agent_runs /
  tool_calls / analysis_results / reports`。初始化：`.\.venv\Scripts\python.exe -m src.db.init_db`。
- **Redis**（`src/cache/`）：会话 `agent:session:{id}`（status/step）+ 通用缓存（JSON + TTL）。
- **接入**：`PersistenceService` 在 Agent 运行中记录 task/run/tool_call（区分 local/mcp）/result/report，
  并实时更新 Redis session 状态；本地 `reports/*.md` 保留不变。
- **开关**：`.env` 设 `DB_ENABLED=true` / `REDIS_ENABLED=true` / `REDIS_CACHE_TTL=3600`。

## 报告系统（结构化、可信、可追溯）

`src/agent/report_builder.py` + `src/agent/nodes/report.py` 将报告升级为 17 章节结构化报告：

- **结构化上下文** `build_report_context()`：统一汇集 dataset_metadata / tool_results / generated_charts /
  insights / observations / skills_context / rag_results，数字全部来自真实工具结果，不凭空生成。
- **图表真实性**：`validate_chart_paths()` 只保留真实存在且扩展名合法的图表；文件名派生友好标题；
  报告只引用实际生成的图表，虚假图表引用被 `sanitize_report()` 自动剔除。
- **质量检查** `validate_report()`：检查非空、章节、占位符、JSON 残留、虚假图表引用（失败仅告警不中断）。

## 后续路线（未实现 Phase）

- Phase 12 测试补全、Phase 13 Docker 完善、Phase 14 README/Demo 完善。
