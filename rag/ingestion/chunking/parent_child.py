"""Parent-child chunker.

Produces two levels of chunks from a Document's elements:

  Parent chunk — one per section (all non-heading elements under the
                 same section_path grouped together). Stored for LLM
                 context retrieval but NOT embedded.

  Child chunks — fixed-size token splits of each parent. These are
                 embedded and stored in the vector DB. Each child
                 carries the parent's metadata plus its own parent_id
                 so the retrieval layer can fetch the full parent.

Why two levels?
  Vector search uses small child chunks for precision.
  The LLM receives the full parent chunk for context.

Token counting uses a simple word-split approximation (word count × 1.3)
to avoid a hard dependency on tiktoken at ingestion time. Close enough
for chunking purposes.

Output:
  chunk_documents(document) -> tuple[list[Chunk], list[Chunk]]
                                       parents      children
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from rag.ingestion.models import Document, Element


# --- Chunk model ---

@dataclass
class Chunk:
    """A piece of content ready for embedding or LLM context."""

    id: str
    content: str
    parent_id: str | None   # None for parent chunks; set for child chunks
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Constants ---

CHILD_CHUNK_TOKENS = 200    # target size for child chunks
CHILD_CHUNK_OVERLAP = 20    # token overlap between consecutive children


# --- Helpers ---

def _approx_tokens(text: str) -> int:
    """Approximate token count using word count × 1.3."""
    return int(len(text.split()) * 1.3)


def _split_into_children(text: str, parent_id: str, base_metadata: dict) -> list[Chunk]:
    """Split *text* into overlapping child chunks of ~CHILD_CHUNK_TOKENS tokens."""
    words = text.split()
    # Convert token targets to word counts
    chunk_words = int(CHILD_CHUNK_TOKENS / 1.3)
    overlap_words = int(CHILD_CHUNK_OVERLAP / 1.3)

    children = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_text = " ".join(words[start:end])

        children.append(Chunk(
            id=str(uuid.uuid4()),
            content=chunk_text,
            parent_id=parent_id,
            metadata={**base_metadata, "chunk_type": "child"},
        ))

        if end == len(words):
            break
        start += chunk_words - overlap_words  # slide forward with overlap

    return children


def _group_by_section(elements: list[Element]) -> list[list[Element]]:
    """Group consecutive non-heading elements by their section_path.

    A new group starts whenever the section_path changes.
    Heading elements are skipped (they define the section, not content).
    """
    groups: list[list[Element]] = []
    current_path: list[str] | None = None
    current_group: list[Element] = []

    for element in elements:
        if element.metadata.get("is_heading"):
            continue  # headings define structure, not content

        path = tuple(element.metadata.get("section_path", []))

        if path != current_path:
            if current_group:
                groups.append(current_group)
            current_group = [element]
            current_path = path
        else:
            current_group.append(element)

    if current_group:
        groups.append(current_group)

    return groups


# --- Public API ---

def chunk_document(document: Document) -> tuple[list[Chunk], list[Chunk]]:
    """Split a Document into parent and child chunks.

    Returns (parents, children).
    Parents are full-section chunks for LLM context.
    Children are small overlapping chunks for vector search.
    """
    groups = _group_by_section(document.elements)
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for group in groups:
        # Build the parent chunk by joining all element content in the section
        combined_text = " ".join(
            e.content for e in group if e.content
        ).strip()

        if not combined_text:
            continue

        # Pull only the fields that belong in a chunk's metadata.
        # We read directly from the Document object and the first element
        # rather than spreading element.metadata, which may contain
        # parser-internal flags (is_heading, level, ocr, etc.).
        first = group[0]
        base_meta = {
            "document_id": document.id,
            "source": document.source,
            "mime_type": document.mime_type,
            "title": document.title,
            "section_path": first.metadata.get("section_path", []),
            "page": first.page,
        }

        parent_id = str(uuid.uuid4())
        parent = Chunk(
            id=parent_id,
            content=combined_text,
            parent_id=None,
            metadata={**base_meta, "chunk_type": "parent"},
        )
        parents.append(parent)

        # Split the parent into child chunks
        child_meta = {**base_meta, "parent_id": parent_id}
        group_children = _split_into_children(combined_text, parent_id, child_meta)
        children.extend(group_children)

    return parents, children
