"""Shared utilities for the ingestion pipeline."""

import hashlib


def make_doc_id(path: str) -> str:
    """Derive a short, stable document id from the file's absolute path."""
    return hashlib.md5(path.encode()).hexdigest()[:12]
