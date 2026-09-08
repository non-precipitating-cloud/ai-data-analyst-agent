"""报告工具：将 Markdown 报告保存到 reports/。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from src.config.settings import get_settings


def save_report_file(content: str, title: str = "") -> str:
    """保存 Markdown 报告，返回文件路径。"""
    settings = get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r"[^\w一-鿿\-]+", "_", title).strip("_") if title else "report"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path: Path = settings.reports_dir / f"{safe_title}_{stamp}.md"

    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


@tool
def save_report(content: str, title: str = "") -> str:
    """将 Markdown 格式的分析报告保存到 reports/ 目录，返回保存路径。"""
    try:
        out = save_report_file(content, title)
        return f"报告已保存: {out}"
    except Exception as e:
        return f"报告保存失败：{type(e).__name__}: {e}"
