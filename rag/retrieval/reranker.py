"""Cross-encoder reranker — scores query-chunk pairs for precise relevance.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80MB, runs locally via sentence-transformers
  - Trained on MS MARCO passage ranking (standard IR benchmark)
  - Takes (query, passage) pair and returns a single relevance score

Why rerank after hybrid search?
  Vector search and BM25 both score independently — they measure similarity
  or keyword overlap, not true query-document relevance. A cross-encoder
  sees query and document together, giving a much more accurate relevance
  score. We only run it on the top results (not the full index) so speed
  is not a concern.

The model is loaded once and cached for the process lifetime.
"""

from sentence_transformers import CrossEncoder

from rag.retrieval.retriever import RetrievalResult


# --- Model (loaded once on first use) ---

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


# --- Public API ---

def rerank(query: str, results: list[RetrievalResult], top_n: int = 5) -> list[RetrievalResult]:
    """Rerank *results* by cross-encoder relevance score.

    Uses the parent content if available (full section context) — this gives
    the cross-encoder more signal than the small child chunk alone.
    Falls back to child content if parent is not available.

    Returns the top_n results sorted by cross-encoder score descending.
    """
    if not results:
        return []

    model = _get_model()

    # Build (query, passage) pairs — prefer parent text for richer context
    pairs = [
        (query, r.parent_content or r.child_content)
        for r in results
    ]

    scores = model.predict(pairs)

    # Attach scores and sort
    scored = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True,
    )

    return [result for _, result in scored[:top_n]]
