"""Entry point for the RAG ingestion pipeline.

Usage:
    python main.py <file_path>

Example:
    python main.py docs/report.pdf
    python main.py docs/readme.md
    python main.py docs/notes.txt
"""

import sys

from rag.ingestion.pipeline import ingest


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path>")
        sys.exit(1)

    path = sys.argv[1]

    try:
        parents, children = ingest(path)
        print(f"Ingestion complete.")
        print(f"  Parent chunks : {len(parents)}")
        print(f"  Child chunks  : {len(children)}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
