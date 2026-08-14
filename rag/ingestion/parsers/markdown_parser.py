"""Markdown parser — parses .md file preserving heading structure.

Each heading becomes a "text" element tagged with is_heading=True and
its level. Non-heading elements are stored as-is without section_path —
that is added downstream by the hierarchy processor.

Example output for:
  # Auth
  JWT is used...
  ## Tokens
  Token expires in 24h.

  Element 0: type=text, content="Auth",           metadata={"level": 1, "is_heading": True}
  Element 1: type=text, content="JWT is used...", metadata={}
  Element 2: type=text, content="Tokens",         metadata={"level": 2, "is_heading": True}
  Element 3: type=text, content="Token expires...", metadata={}
"""

import re
from pathlib import Path

from rag.ingestion.models import Document, Element
from rag.ingestion.utils import make_doc_id


def _heading_level(line: str) -> int | None:
    """Return the heading level (1-6) ie number of #'s; if the line is a heading, else None."""
    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else None


def _heading_text(line: str) -> str:
    """Strip the leading # characters from a heading line."""
    return re.sub(r"^#{1,6}\s+", "", line).strip()


def parse_markdown(path: str) -> Document:
    """Parse a Markdown file and return a Document with structured elements.

    Headings are stored as elements with is_heading=True and their level.
    Non-heading elements are stored with an empty metadata dict — section_path
    is attached downstream by the hierarchy processor.
    """
    abs_path = str(Path(path).resolve())
    doc_id = make_doc_id(abs_path)

    raw = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    document = Document(
        id=doc_id,
        source=abs_path,
        mime_type="text/plain",
        title=Path(path).stem,
    )

    element_index = 0

    for line in lines:
        line = line.rstrip()

        if not line:
            continue

        level = _heading_level(line)

        if level is not None:
            # Heading element — tag with level and is_heading only.
            # section_path is added downstream by the hierarchy processor.
            heading_text = _heading_text(line)
            document.elements.append(
                Element(
                    id=f"{doc_id}-e{element_index}",
                    type="text",
                    content=heading_text,
                    metadata={"level": level, "is_heading": True},
                )
            )
        else:
            # Body element — no section_path yet; hierarchy processor adds it.
            document.elements.append(
                Element(
                    id=f"{doc_id}-e{element_index}",
                    type="text",
                    content=line,
                    metadata={},
                )
            )

        element_index += 1

    return document
