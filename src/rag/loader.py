"""知识库文档加载与切分（RAG 链路前两步）。

- load_knowledge_documents：扫描 knowledge/ 目录下的 Markdown 文件，
  读取全文并附上来源文件名、业务分类、标题等元数据；
- chunk_documents：把整篇文档按字符窗口切成带重叠的小文本块（chunk），
  块大小控制在 embedding/检索的有效范围内，重叠避免语义被硬切断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 文件名（不含扩展名）→ 知识库业务分类，加载时写入 metadata 供检索结果标注
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
    """一篇加载后的知识文档。

    Attributes:
        text: Markdown 全文。
        metadata: 文档级元数据（source 文件名、category 分类、title 标题），
            切分后会原样继承到每个文本块。
    """

    text: str
    metadata: dict = field(default_factory=dict)


def load_knowledge_documents(knowledge_dir: str | Path) -> list[Document]:
    """加载 knowledge/ 目录下所有 Markdown 文档。

    Args:
        knowledge_dir: 知识文档目录路径。

    Returns:
        Document 列表（按文件名排序，保证导入顺序稳定）；
        每个文档的 metadata 含来源文件、业务分类与标题。
    """
    docs: list[Document] = []
    # sorted 保证每次导入块顺序、chunk_index 稳定可复现
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        stem = path.stem
        # 优先用内置映射得到中文分类；未知文件则把 slug 美化成标题式英文名
        category = CATEGORY_BY_FILE.get(stem, stem.replace("-", " ").title())
        title = stem
        # 若文档以 “# 标题” 开头，则用该一级标题作为文档标题
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
    """按字符窗口切分，优先在换行处断开，带重叠。

    Args:
        text: 待切分的整篇文本。
        chunk_size: 单块最大字符数。
        overlap: 相邻块之间重叠的字符数（保留上下文、避免语义被硬切断）。

    Returns:
        文本块字符串列表；空文本返回空列表。
    """
    text = text.strip()
    if not text:
        return []
    # 短文不切，整体作为一个块
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    # 滑动窗口：每次推进 chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            # 在窗口内找最后一个换行：若落在后半段，则改在换行处断开，保持段落完整
            nl = chunk.rfind("\n")
            if nl > chunk_size * 0.5:
                end = start + nl
                chunk = text[start:end]
        # 跳过切分后只剩空白的块
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        # 下一块回退 overlap 个字符，形成块间重叠
        start = end - overlap
    return chunks


def chunk_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[dict]:
    """把文档切分成带元数据的文本块（embedding 入库的基本单位）。

    Args:
        docs: load_knowledge_documents 加载的文档列表。
        chunk_size: 单块最大字符数，默认 800。
        chunk_overlap: 块间重叠字符数，默认 100。

    Returns:
        字典列表，每项形如 ``{"text": 块文本, "metadata": {...}}``；
        metadata 在文档元数据基础上追加 chunk_index（块在文档内的序号）。
    """
    chunks: list[dict] = []
    for doc in docs:
        for i, text in enumerate(_split_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "text": text,
                    # 继承文档级元数据并补块序号，便于溯源到具体文件与位置
                    "metadata": {**doc.metadata, "chunk_index": i},
                }
            )
    return chunks
