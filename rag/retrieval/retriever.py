"""Hybrid retriever — combines vector search and BM25, fused with RRF.

Flow:
  1. Embed the query with text-embedding-3-small
  2. Vector search  — top-k from Qdrant children collection
  3. BM25 search    — top-k from in-memory BM25 index built from Qdrant payloads
  4. RRF fusion     — merge the two ranked lists into one
  5. Fetch parents  — replace each child with its full parent chunk

The BM25 index is built lazily on first query by loading all child chunk
texts from Qdrant. It stays in memory for the process lifetime.

RRF formula:  score(d) = Σ 1 / (k + rank(d))
  k=60 is the standard constant that dampens the influence of very high ranks.
"""

import os
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from rag.embeddings.embedder import embed
from rag.embeddings.vector_store import get_parent
from rag.ingestion.chunking.parent_child import Chunk


# --- Constants ---

_CHILDREN_COLLECTION = "children"
_TOP_K = 20          # candidates retrieved from each method before fusion
_RRF_K = 60          # RRF damping constant (standard value)
_FINAL_TOP_N = 10    # results returned after fusion (before reranker)


# --- Result model ---

@dataclass
class RetrievalResult:
    """A single retrieval result with its parent context and fusion score."""
    child_id: str
    child_content: str
    parent_content: str | None   # None if parent fetch fails
    metadata: dict
    rrf_score: float


# --- BM25 index (lazy, in-memory) ---

_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[dict] | None = None   # raw payloads parallel to index


def _build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Load all child chunks from Qdrant and build a BM25 index.

    Called once on first query. Subsequent queries reuse the cached index.
    """
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )

    # Scroll through all points in the children collection
    chunks = []
    offset = None
    while True:
        response = client.scroll(
            collection_name=_CHILDREN_COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=100,
            offset=offset,
        )
        # qdrant-client 1.9+ returns a ScrollResult object; unpack safely
        points = response[0] if isinstance(response, tuple) else response.points
        next_offset = response[1] if isinstance(response, tuple) else response.next_page_offset
        chunks.extend([r.payload for r in points])
        if next_offset is None:
            break
        offset = next_offset

    # Tokenise by whitespace for BM25
    tokenised = [c.get("content", "").lower().split() for c in chunks]
    return BM25Okapi(tokenised), chunks


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    """Return the cached BM25 index, building it on first call."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is None:
        _bm25_index, _bm25_chunks = _build_bm25_index()
    return _bm25_index, _bm25_chunks


def invalidate_bm25_cache() -> None:
    """Clear the in-memory BM25 index so it is rebuilt on the next query.

    Call this after ingesting new documents to keep BM25 in sync with Qdrant.
    """
    global _bm25_index, _bm25_chunks
    _bm25_index = None
    _bm25_chunks = None


# --- RRF fusion ---

def _rrf_fuse(
    vector_ids: list[str],
    bm25_ids: list[str],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Returns a list of (chunk_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(vector_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, chunk_id in enumerate(bm25_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# --- Public API ---

def retrieve(query: str, top_n: int = _FINAL_TOP_N) -> list[RetrievalResult]:
    """Run hybrid retrieval for *query* and return the top results.

    Each result contains the matched child chunk, its parent's full text,
    and the fused RRF score.
    """
    # --- Step 1: Embed the query ---
    # embed() only uses .content — parent_id and id are not used for embedding
    query_chunk = Chunk(id=str(uuid.uuid4()), content=query, parent_id=None)
    query_vector = embed([query_chunk])[0]

    # --- Step 2: Vector search ---
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    vector_hits = client.search(
        collection_name=_CHILDREN_COLLECTION,
        query_vector=query_vector,
        limit=_TOP_K,
        with_payload=True,
    )
    vector_ids = [hit.payload["chunk_id"] for hit in vector_hits]
    vector_payloads = {hit.payload["chunk_id"]: hit.payload for hit in vector_hits}

    # --- Step 3: BM25 search ---
    bm25, bm25_chunks = _get_bm25()
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)

    # Get top-k indices by score
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:_TOP_K]

    bm25_ids = [bm25_chunks[i]["chunk_id"] for i in top_bm25_indices]
    bm25_payloads = {bm25_chunks[i]["chunk_id"]: bm25_chunks[i] for i in top_bm25_indices}

    # --- Step 4: RRF fusion ---
    fused = _rrf_fuse(vector_ids, bm25_ids)[:top_n]

    # --- Step 5: Build results with parent context ---
    results = []
    all_payloads = {**vector_payloads, **bm25_payloads}

    for chunk_id, rrf_score in fused:
        payload = all_payloads.get(chunk_id, {})
        parent_id = payload.get("parent_id")
        parent = get_parent(parent_id) if parent_id else None

        results.append(RetrievalResult(
            child_id=chunk_id,
            child_content=payload.get("content", ""),
            parent_content=parent.get("content") if parent else None,
            metadata={k: v for k, v in payload.items() if k not in ("content", "chunk_id")},
            rrf_score=rrf_score,
        ))

    return results
