"""报告工具：将 Agent 生成的 Markdown 分析报告落盘保存。

输入输出契约：
- 输入：报告正文 content（Markdown 字符串）与可选标题 title（用于文件名）。
- 输出（@tool 层）：成功返回 "报告已保存: <路径>"；失败返回中文错误提示。
- 文件写入配置项 reports_dir（默认 reports/），UTF-8 编码；
  文件名由“安全化标题 + 时间戳”组成，既防路径注入也避免覆盖历史报告。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from src.config.settings import get_settings
from src.tools.errors import KIND_EXECUTION_ERROR, tool_error


def save_report_file(content: str, title: str = "") -> str:
    """保存 Markdown 报告，返回文件路径。

    参数：
        content: Markdown 报告全文。
        title: 可选标题，会被清洗为文件名安全字符；为空时用 report。
    返回值：
        str: 报告文件的绝对路径（.md）。
    """
    settings = get_settings()
    # 确保报告目录存在（首次运行时自动创建）
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    # 文件名安全化：只保留单词字符、中文与连字符，其余（含 / \\ : 等）替换为下划线，
    # 防止标题里的路径分隔符造成目录穿越
    safe_title = re.sub(r"[^\w一-鿿\-]+", "_", title).strip("_") if title else "report"
    # 时间戳后缀：同一标题多次保存不会覆盖
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path: Path = settings.reports_dir / f"{safe_title}_{stamp}.md"

    # 显式 UTF-8 写入，保证 Windows 下中文报告不乱码
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


@tool
def save_report(content: str, title: str = "") -> str:
    """将 Markdown 格式的分析报告保存到 reports/ 目录，返回保存路径。

    返回值：
        str: 成功提示（含路径）；写盘异常时返回结构化错误文本，不向上抛异常。
    """
    try:
        out = save_report_file(content, title)
        return f"报告已保存: {out}"
    except Exception as e:  # noqa: BLE001 —— 工具层不抛异常
        return tool_error(
            KIND_EXECUTION_ERROR,
            f"报告保存失败：{type(e).__name__}: {e}",
            hint="请检查 reports/ 目录是否可写、磁盘是否已满。",
        )
