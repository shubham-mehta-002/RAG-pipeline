"""Plain text parser — reads a .txt file as a single text element."""

from pathlib import Path

from rag.ingestion.models import Document, Element
from rag.ingestion.utils import make_doc_id


def parse_text(path: str) -> Document:
    """Parse a plain text file and return a Document with one text element."""
    abs_path = str(Path(path).resolve())
    doc_id = make_doc_id(abs_path)

    content = Path(abs_path).read_text(encoding="utf-8", errors="replace").strip()

    document = Document(
        id=doc_id,
        source=abs_path,
        mime_type="text/plain",
        title=Path(path).stem,
    )

    document.elements.append(
        Element(
            id=f"{doc_id}-e0",
            type="text",
            content=content,
        )
    )

    return document
