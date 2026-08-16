"""Hierarchy builder — attaches section_path metadata to elements.

For Markdown documents, the parser tags headings with is_heading=True
and their level. This module walks those tags and builds the section_path
for all non-heading elements.

For PDF documents, headings are detected by font size: any text block
whose font size is notably larger than the document average is treated
as a heading. This is a heuristic — it works well for structured PDFs
but may misfire on stylised documents.

Plain text documents have no heading signals, so all elements get an
empty section_path.

After this step every non-heading element has:
  metadata["section_path"] = ["Top heading", "Sub heading", ...]
"""

import fitz

from rag.ingestion.models import Document


# --- Markdown hierarchy ---

def _build_markdown_hierarchy(document: Document) -> Document:
    """Attach section_path to non-heading elements using the heading stack."""
    heading_stack: list[str] = []

    for element in document.elements:
        if element.metadata.get("is_heading"):
            level = element.metadata.get("level", 1)
            heading_stack = heading_stack[:level - 1]
            heading_stack.append(element.content or "")
        else:
            element.metadata["section_path"] = list(heading_stack)

    return document


# --- PDF hierarchy ---

def _get_pdf_font_sizes(source: str) -> dict[int, list[float]]:
    """Return a mapping of page_number → list of font sizes on that page."""
    sizes: dict[int, list[float]] = {}
    pdf = fitz.open(source)
    for page_num, page in enumerate(pdf, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_sizes = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    page_sizes.append(span["size"])
        sizes[page_num] = page_sizes
    pdf.close()
    return sizes


def _average_font_size(sizes: dict[int, list[float]]) -> float:
    """Compute the average font size across all pages."""
    all_sizes = [s for page in sizes.values() for s in page]
    return sum(all_sizes) / len(all_sizes) if all_sizes else 12.0


def _detect_pdf_headings(document: Document) -> Document:
    """Tag PDF text elements as headings if their font size is above average.

    Uses PyMuPDF to read font sizes per page, then compares each element's
    page average against the document-wide average.
    A page whose average font size is > 1.2× the document average is tagged
    as a heading. This is a heuristic — good enough for structured PDFs.
    """
    try:
        font_sizes = _get_pdf_font_sizes(document.source)
    except Exception:
        # If we can't read font sizes, skip heading detection
        return document

    doc_avg = _average_font_size(font_sizes)

    # Build a set of page numbers that look like heading pages
    heading_pages: set[int] = set()
    for page_num, sizes in font_sizes.items():
        if sizes:
            page_avg = sum(sizes) / len(sizes)
            if page_avg > doc_avg * 1.2:
                heading_pages.add(page_num)

    for element in document.elements:
        if element.type == "text" and element.page in heading_pages:
            element.metadata["is_heading"] = True
            element.metadata["level"] = 1  # PDFs don't have nested heading levels

    return document


def _build_pdf_hierarchy(document: Document) -> Document:
    """Detect headings by font size, then attach section_path to body elements."""
    document = _detect_pdf_headings(document)

    heading_stack: list[str] = []
    for element in document.elements:
        if element.metadata.get("is_heading"):
            heading_stack = [element.content or ""]
        else:
            element.metadata["section_path"] = list(heading_stack)

    return document


# --- Public API ---

def build_hierarchy(document: Document) -> Document:
    """Attach section_path metadata to all non-heading elements.

    Dispatches to the correct strategy based on MIME type.
    """
    if document.mime_type == "text/plain":
        # Check if it was parsed as Markdown (elements will have is_heading tags)
        has_headings = any(e.metadata.get("is_heading") for e in document.elements)
        if has_headings:
            return _build_markdown_hierarchy(document)
        # Plain .txt — no heading signals, attach empty section_path
        for element in document.elements:
            element.metadata.setdefault("section_path", [])
        return document

    if document.mime_type == "application/pdf":
        return _build_pdf_hierarchy(document)

    return document
