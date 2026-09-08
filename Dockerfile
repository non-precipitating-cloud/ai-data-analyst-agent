FROM python:3.11-slim

# UTF-8 环境 + 无缓冲输出
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# 先复制依赖清单与源码，安装依赖（可编辑安装，保证 PROJECT_ROOT 解析为 /app）
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# 复制运行时需要的代码与数据
COPY main.py ./
COPY datasets ./datasets
COPY knowledge ./knowledge
COPY scripts ./scripts

# 非 root 用户 + 运行时目录（reports 卷挂载点）
RUN useradd --create-home appuser && \
    mkdir -p /app/reports /app/logs /app/workspace && \
    chown -R appuser:appuser /app
USER appuser

# 保持容器存活，交互式 CLI 通过 docker compose exec 进入
CMD ["tail", "-f", "/dev/null"]
