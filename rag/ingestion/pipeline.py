"""Ingestion pipeline — wires all stages end-to-end.

Stage order:
  1. Validate        — FileValidator checks existence, MIME, size, duplicates
  2. Route           — DocumentRouter identifies the file type
  3. Parse           — appropriate parser converts the file to a Document
  4. Normalize       — clean unicode artifacts and collapse whitespace
  5. Build Hierarchy — attach section_path to every non-heading element
  6. Chunk           — split into parent/child Chunk pairs
  7. Embed           — embed child chunks via text-embedding-3-small
  8. Store           — persist children (with vectors) + parents in Qdrant

Usage:
    from rag.ingestion.pipeline import ingest

    ingest("/path/to/file.pdf")
"""

from rag.ingestion.validators.file_validator import FileValidator
from rag.ingestion.router import DocumentRouter
from rag.ingestion.parsers.pdf_parser import parse_pdf
from rag.ingestion.parsers.text_parser import parse_text
from rag.ingestion.parsers.markdown_parser import parse_markdown
from rag.ingestion.processors.normalizer import normalize
from rag.ingestion.processors.hierarchy import build_hierarchy
from rag.ingestion.chunking.parent_child import chunk_document
from rag.embeddings.embedder import embed
from rag.embeddings.vector_store import ensure_collections, store_children, store_parents
from rag.retrieval.retriever import invalidate_bm25_cache


# Module-level singletons — created once, reused for every call
_validator = FileValidator()
_router = DocumentRouter()


def ingest(path: str) -> None:
    """Run the full ingestion pipeline on a single file.

    Validates, parses, normalizes, chunks, embeds, and stores the file
    in Qdrant. After this call the file's content is searchable.

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

    # --- Stage 7: Embed children ---
    vectors = embed(children)

    # --- Stage 8: Store in Qdrant ---
    ensure_collections()
    store_children(children, vectors)
    store_parents(parents)

    # Invalidate BM25 cache so next query picks up the new chunks
    invalidate_bm25_cache()
