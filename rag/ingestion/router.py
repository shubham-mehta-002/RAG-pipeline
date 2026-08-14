"""Document Router

Receives a validated local file path and returns a route string:
  "pdf"         — application/pdf
  "text"        — text/plain with .txt extension
  "markdown"    — text/plain with .md extension
  "unsupported" — anything else, or when MIME detection fails
"""

from pathlib import Path

import magic


# --- Constants ---

_ROUTE_PDF = "pdf"
_ROUTE_TEXT = "text"
_ROUTE_MARKDOWN = "markdown"
_ROUTE_UNSUPPORTED = "unsupported"


# --- Router ---

class DocumentRouter:
    """Determines the document type of a validated local file."""

    def route(self, path: str) -> str:
        """Return the route string for the file at *path*.

        Does not validate the file — assumes File Validator has
        already accepted it. Never raises; returns "unsupported" on
        any detection failure.
        """
        abs_path = Path(path).resolve()
        mime = self._detect_mime(str(abs_path))

        if mime is None:
            return _ROUTE_UNSUPPORTED

        if mime == "application/pdf":
            return _ROUTE_PDF

        if mime == "text/plain":
            return self._route_plain_text(abs_path)

        return _ROUTE_UNSUPPORTED

    # --- Private helpers ---

    def _detect_mime(self, abs_path: str) -> str | None:
        """Return content-detected MIME type, or None on failure."""
        try:
            return magic.from_file(abs_path, mime=True)
        except Exception:
            return None

    def _route_plain_text(self, abs_path: Path) -> str:
        """Disambiguate text/plain files using the file extension."""
        ext = abs_path.suffix.lower()
        if ext == ".txt":
            return _ROUTE_TEXT
        if ext == ".md":
            return _ROUTE_MARKDOWN
        return _ROUTE_UNSUPPORTED
