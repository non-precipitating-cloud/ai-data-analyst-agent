# 基础镜像：官方 Python 3.11 slim 精简版（体积小、自带 pip，适合生产镜像）
FROM python:3.11-slim

# Python 运行时环境变量（用一条 ENV 指令设置多项，减少镜像层数）：
# PYTHONUNBUFFERED=1          print/日志立即输出不缓冲，docker logs 可实时查看
# PYTHONDONTWRITEBYTECODE=1   不生成 .pyc 字节码文件，保持镜像整洁
# PYTHONUTF8 / LANG / LC_ALL  统一使用 UTF-8 编码，避免容器内中文乱码
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 设定容器内工作目录（不存在会自动创建），后续 COPY/RUN 都以 /app 为当前目录
WORKDIR /app

# 安装系统级中文字体：python:3.11-slim 默认不含任何 CJK 字体，
# matplotlib 绘制中文标题/轴标签时会渲染成方框（缺字豆腐块）。
# - fonts-noto-cjk：Google 思源黑体/宋体，覆盖简繁中文、日文、韩文
# - fontconfig：提供字体缓存与查询（fc-list / fc-cache），matplotlib 据此发现字体
# 用完立即删除 apt 索引列表，避免该层残留 /var/lib/apt/lists 增大镜像体积
RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-noto-cjk fontconfig && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖清单与源码再安装依赖：利用 Docker 分层缓存，清单未变时该层直接命中缓存
# 可编辑安装（-e .）让 src 以源码形式存在，保证 PROJECT_ROOT 解析为 /app
COPY pyproject.toml README.md ./
COPY src ./src
# 先升级 pip，再以可编辑模式安装本项目及全部运行时依赖（--no-cache-dir 不留安装缓存，减小镜像体积）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# 复制运行时需要的代码与数据（放在依赖安装之后：业务代码频繁变动不会使依赖层缓存失效）
# 应用入口脚本
COPY main.py ./
# 内置示例数据集
COPY datasets ./datasets
# RAG 向量检索使用的知识库文档
COPY knowledge ./knowledge
# 辅助脚本
COPY scripts ./scripts
# 离线评测脚本（可在容器内执行 `python -m eval.agent_eval` 验证镜像可用性）
COPY eval ./eval

# 创建非 root 用户并准备运行时目录（reports 是 docker-compose 的卷挂载点）
# 遵循最小权限原则：应用不以 root 身份运行，降低容器被攻破后的风险
RUN useradd --create-home appuser && \
    mkdir -p /app/reports /app/logs /app/workspace && \
    chown -R appuser:appuser /app
# 切换到非 root 用户，后续指令及容器进程都以 appuser 身份执行
USER appuser

# 保持容器前台存活不退出，交互式 CLI 通过 docker compose exec 进入容器使用
CMD ["tail", "-f", "/dev/null"]
