"""Vector store — persists chunks in Qdrant Cloud.

Two collections:
  children  — child chunks with vectors (used for similarity search)
  parents   — parent chunks without vectors (fetched by parent_id at retrieval)

Children are stored with full metadata so the retrieval layer can
filter by document, section, page etc. before or after vector search.

Parents are stored as payload-only points using a deterministic integer
id derived from their UUID so Qdrant can store them without a vector.
"""

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from rag.ingestion.chunking.parent_child import Chunk


# --- Constants ---

_CHILDREN_COLLECTION = "children"
_PARENTS_COLLECTION = "parents"
_VECTOR_SIZE = 1536  # text-embedding-3-small dimensions


# --- Client ---

def _get_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )


# --- Collection setup ---

def ensure_collections() -> None:
    """Create collections if they don't already exist.

    Safe to call on every startup — skips creation if already present.
    """
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}

    if _CHILDREN_COLLECTION not in existing:
        client.create_collection(
            collection_name=_CHILDREN_COLLECTION,
            vectors_config=VectorParams(
                size=_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    if _PARENTS_COLLECTION not in existing:
        # Parents are stored without vectors — use a dummy single-dim vector
        # Qdrant requires a vector config even for payload-only collections,
        # so we use size=1 and never query by vector in this collection.
        client.create_collection(
            collection_name=_PARENTS_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )


# --- Helpers ---

def _uuid_to_int(id_str: str) -> int:
    """Convert a UUID string to a stable integer for use as Qdrant point id."""
    return uuid.UUID(id_str).int % (2**63)


# --- Public API ---

def store_children(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Store child chunks with their embedding vectors in Qdrant."""
    client = _get_client()

    points = [
        PointStruct(
            id=_uuid_to_int(chunk.id),
            vector=vector,
            payload={
                "chunk_id": chunk.id,
                "content": chunk.content,
                **chunk.metadata,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=_CHILDREN_COLLECTION, points=points)


def store_parents(chunks: list[Chunk]) -> None:
    """Store parent chunks as payload-only points (no vector search on parents)."""
    client = _get_client()

    points = [
        PointStruct(
            id=_uuid_to_int(chunk.id),
            vector=[0.0],  # dummy vector — parents are never searched by vector
            payload={
                "chunk_id": chunk.id,
                "content": chunk.content,
                **chunk.metadata,
            },
        )
        for chunk in chunks
    ]

    client.upsert(collection_name=_PARENTS_COLLECTION, points=points)


def get_parent(parent_id: str) -> dict | None:
    """Fetch a parent chunk by its id. Returns the payload dict or None."""
    client = _get_client()

    results = client.retrieve(
        collection_name=_PARENTS_COLLECTION,
        ids=[_uuid_to_int(parent_id)],
        with_payload=True,
    )

    return results[0].payload if results else None
