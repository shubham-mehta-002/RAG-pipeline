"""File validation module for the RAG ingestion pipeline.

Validates a local file against four sequential checks:
existence, MIME type, file size, and duplicate-content detection.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import magic


# --- Result type ---

@dataclass
class ValidationResult:
    """Returned by FileValidator.validate() for every call."""
    valid: bool   # True only when all checks pass
    reason: str   # "ok" on success; descriptive message on failure


# --- Validator ---

class FileValidator:
    """Entry gate for the RAG ingestion pipeline.

    Applies four sequential checks and returns a ValidationResult.
    The hash store (seen_hashes.json) is only written when all checks pass.
    """

    # Allowed content-detected MIME types (PDF and plain text).
    # Note: python-magic always returns "text/plain" for .md files — the
    # router disambiguates .md vs .txt using the file extension.
    _ALLOWED_MIME_TYPES: frozenset = frozenset({
        "application/pdf",
        "text/plain",
    })

    # Maximum accepted file size (20 MB)
    _SIZE_LIMIT: int = 20 * 1024 * 1024

    # Read chunk size used when computing SHA-256
    _CHUNK_SIZE: int = 8192

    # Path to the JSON file that stores previously seen hashes
    _HASH_STORE_PATH: str = "seen_hashes.json"

    # --- Public API ---

    def validate(self, path: str) -> ValidationResult:
        """Validate a file at *path* against all four checks.

        Returns a ValidationResult. Checks are applied in order and the
        first failure short-circuits the rest.
        """
        # 1. Resolve to absolute path
        abs_path = Path(path).resolve()

        # 2. Check existence
        if not abs_path.exists():
            return ValidationResult(valid=False, reason=f"File not found: {abs_path}")

        # 3. Check it is a regular file
        if not abs_path.is_file():
            return ValidationResult(valid=False, reason=f"Path is not a regular file: {abs_path}")

        # 4. MIME type check
        mime = self._detect_mime(str(abs_path))
        if mime is None:
            return ValidationResult(valid=False, reason=f"Cannot determine MIME type: {abs_path}")
        if mime not in self._ALLOWED_MIME_TYPES:
            return ValidationResult(valid=False, reason=f"Unsupported MIME type: {mime}")

        # 5. File size check
        size = abs_path.stat().st_size
        if size == 0:
            return ValidationResult(valid=False, reason=f"File is empty: {abs_path}")
        if size > self._SIZE_LIMIT:
            return ValidationResult(
                valid=False,
                reason=f"File too large: {size} bytes (limit {self._SIZE_LIMIT} bytes)",
            )

        # 6. Duplicate content check
        file_hash = self._compute_sha256(str(abs_path))
        hashes = self._load_hashes()

        if file_hash in hashes:
            return ValidationResult(valid=False, reason="Duplicate file: content already ingested")

        hashes.add(file_hash)
        try:
            self._save_hashes(hashes)
        except OSError as e:
            return ValidationResult(valid=False, reason=f"Failed to write hash store: {e}")

        return ValidationResult(valid=True, reason="ok")

    # --- Private helpers ---

    def _detect_mime(self, abs_path: str) -> str | None:
        """Return the content-detected MIME type, or None on failure."""
        try:
            return magic.from_file(abs_path, mime=True)
        except Exception:
            return None

    def _compute_sha256(self, abs_path: str) -> str:
        """Return the lowercase hex SHA-256 digest of the file at abs_path."""
        hasher = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while chunk := f.read(self._CHUNK_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_hashes(self) -> set:
        """Load previously seen SHA-256 hashes from the store.

        Returns an empty set if the file is missing or malformed.
        """
        try:
            with open(self._HASH_STORE_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_hashes(self, hashes: set) -> None:
        """Persist *hashes* to seen_hashes.json as a JSON array.

        Raises OSError on write failure.
        """
        with open(self._HASH_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(list(hashes), f)
