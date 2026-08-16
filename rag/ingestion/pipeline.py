"""Ingestion pipeline — wires all stages end-to-end.

Stage order:
  1. Validate        — FileValidator checks existence, MIME, size, duplicates
  2. Route           — DocumentRouter identifies the file type
  3. Parse           — appropriate parser converts the file to a Document
  4. Normalize       — clean unicode artifacts and collapse whitespace
  5. Build Hierarchy — attach section_path to every non-heading element
  6. Chunk           — split into parent/child Chunk pairs

Usage:
    from rag.ingestion.pipeline import ingest

    parents, children = ingest("/path/to/file.pdf")
"""

from rag.ingestion.validators.file_validator import FileValidator
from rag.ingestion.router import DocumentRouter
from rag.ingestion.parsers.pdf_parser import parse_pdf
from rag.ingestion.parsers.text_parser import parse_text
from rag.ingestion.parsers.markdown_parser import parse_markdown
from rag.ingestion.processors.normalizer import normalize
from rag.ingestion.processors.hierarchy import build_hierarchy
from rag.ingestion.chunking.parent_child import chunk_document, Chunk


# Module-level singletons — created once, reused for every call
_validator = FileValidator()
_router = DocumentRouter()


def ingest(path: str) -> tuple[list[Chunk], list[Chunk]]:
    """Run the full ingestion pipeline on a single file.

    Args:
        path: Path to the file to ingest (absolute or relative).

    Returns:
        (parents, children) — two lists of Chunk objects.
        parents: one chunk per document section, for LLM context.
        children: small overlapping chunks, for vector search.

    Raises:
        ValueError: if validation fails or the file type is unsupported.
    """
    # --- Stage 1: Validate ---
    result = _validator.validate(path)
    if not result.valid:
        raise ValueError(f"Validation failed: {result.reason}")

    # --- Stage 2: Route ---
    route = _router.route(path)
    if route == "unsupported":
        raise ValueError(f"Unsupported file type: {path}")

    # --- Stage 3: Parse ---
    if route == "pdf":
        document = parse_pdf(path)
    elif route == "markdown":
        document = parse_markdown(path)
    elif route == "text":
        document = parse_text(path)
    else:
        raise ValueError(f"Unknown route '{route}' for: {path}")

    # --- Stage 4: Normalize ---
    document = normalize(document)

    # --- Stage 5: Build Hierarchy ---
    document = build_hierarchy(document)

    # --- Stage 6: Chunk ---
    parents, children = chunk_document(document)

    return parents, children
