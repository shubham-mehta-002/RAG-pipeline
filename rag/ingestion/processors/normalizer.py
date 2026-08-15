"""Normalizer — cleans up elements after parsing.

Applies three passes to every text element in a Document:
  1. Unicode ligature/quote fixes (e.g. fi, fl, smart quotes)
  2. Whitespace collapse (multiple spaces/newlines → single space)
  3. Empty element removal (drop elements with no usable content)

Image elements (content=None) are kept as-is.
"""

import re
import unicodedata

from rag.ingestion.models import Document, Element


# --- Unicode fixes ---

# Common PDF ligatures and smart quotes that don't convert cleanly
_UNICODE_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",   # right single quotation mark
    "\u2018": "'",   # left single quotation mark
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u2013": "-",   # en dash
    "\u2014": "--",  # em dash
    "\u00a0": " ",   # non-breaking space
}


def _fix_unicode(text: str) -> str:
    """Replace known problematic unicode characters with ASCII equivalents."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Normalize remaining unicode to closest ASCII representation
    return unicodedata.normalize("NFKC", text)


def _collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace (including newlines) with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_element(element: Element) -> Element | None:
    """Clean a single element. Returns None if the element should be dropped."""
    # Image elements have no text content — keep them unchanged
    if element.type == "image" or element.content is None:
        return element

    cleaned = _fix_unicode(element.content)
    cleaned = _collapse_whitespace(cleaned)

    # Drop if nothing useful remains
    if not cleaned:
        return None

    element.content = cleaned
    return element


# --- Public API ---

def normalize(document: Document) -> Document:
    """Normalize all elements in a Document in-place and return it.

    Empty elements are removed. Text content is cleaned of unicode
    artifacts and excess whitespace.
    """
    cleaned_elements = []
    for element in document.elements:
        result = _normalize_element(element)
        if result is not None:
            cleaned_elements.append(result)

    document.elements = cleaned_elements
    return document
