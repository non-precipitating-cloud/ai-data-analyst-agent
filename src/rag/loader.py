"""知识库文档加载与切分。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 文件名 → 分类
CATEGORY_BY_FILE = {
    "data-analysis-basics": "数据分析基础",
    "statistics": "统计分析",
    "sales-analysis": "销售分析",
    "financial-analysis": "财务分析",
    "anomaly-detection": "异常检测",
    "data-cleaning": "数据清洗",
    "correlation-analysis": "相关性分析",
}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def load_knowledge_documents(knowledge_dir: str | Path) -> list[Document]:
    """加载 knowledge/ 目录下所有 Markdown 文档。"""
    docs: list[Document] = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        stem = path.stem
        category = CATEGORY_BY_FILE.get(stem, stem.replace("-", " ").title())
        title = stem
        first_line = text.lstrip().splitlines()
        if first_line and first_line[0].startswith("# "):
            title = first_line[0][2:].strip()
        docs.append(
            Document(
                text=text,
                metadata={"source": path.name, "category": category, "title": title},
            )
        )
    return docs


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符窗口切分，优先在换行处断开，带重叠。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            nl = chunk.rfind("\n")
            if nl > chunk_size * 0.5:
                end = start + nl
                chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[dict]:
    """把文档切分成带元数据的文本块。"""
    chunks: list[dict] = []
    for doc in docs:
        for i, text in enumerate(_split_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "text": text,
                    "metadata": {**doc.metadata, "chunk_index": i},
                }
            )
    return chunks
