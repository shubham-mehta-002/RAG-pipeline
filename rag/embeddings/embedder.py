"""Embedder — converts text chunks into vectors using OpenAI embeddings.

Model: text-embedding-3-small
  - 1536 dimensions
  - Best cost/quality ratio (~$0.00002 per 1K tokens)
  - Same API key used for GPT-4o

Only child chunks are embedded — parent chunks are stored as plain text
in the vector store for context retrieval, not for similarity search.

Batching: OpenAI allows up to 2048 inputs per request. We batch in
groups of 100 to stay well within limits and avoid timeouts.
"""

import os
from openai import OpenAI

from rag.ingestion.chunking.parent_child import Chunk


# --- Constants ---

_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# --- Public API ---

def embed(chunks: list[Chunk]) -> list[list[float]]:
    """Embed a list of chunks and return their vectors in the same order.

    Sends text in batches of _BATCH_SIZE to avoid large single requests.
    Returns a flat list of vectors — one per chunk, same order as input.
    """
    vectors: list[list[float]] = []

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i: i + _BATCH_SIZE]
        texts = [chunk.content or "" for chunk in batch]

        # Skip batch if all texts are empty
        if not any(texts):
            vectors.extend([[0.0] * 1536] * len(batch))
            continue

        response = _client.embeddings.create(
            model=_MODEL,
            input=texts,
        )

        # Response data is ordered to match the input
        batch_vectors = [item.embedding for item in response.data]
        vectors.extend(batch_vectors)

    return vectors
