"""Canonical document model for the RAG ingestion pipeline.

Every parser — PDF, text, Markdown — must return a Document
containing a list of Elements. Downstream stages (chunking,
embedding) only deal with this model, never raw parser output.
"""

from dataclasses import dataclass, field
from typing import Any


# --- Element ---

@dataclass
class Element:
    """A single piece of content extracted from a document."""

    id: str               # unique within the document, e.g. "doc-abc-p1-e0"
    type: str             # "text", "table", or "image"
    content: str | None = None   # text content; None for image elements
    page: int | None = None      # 1-indexed page number (None for non-paged sources)
    parent_id: str | None = None # id of the parent element (e.g. section heading)
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Document ---

@dataclass
class Document:
    """A parsed document ready for downstream processing."""

    id: str               # unique document id, derived from file hash or path
    source: str           # absolute path to the original file
    mime_type: str        # e.g. "application/pdf", "text/plain"
    title: str | None = None
    elements: list[Element] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
