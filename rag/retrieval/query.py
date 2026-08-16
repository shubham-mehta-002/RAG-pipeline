"""Query pipeline — single entry point for the full RAG query flow.

Flow:
  1. retrieve()  — hybrid search (vector + BM25 + RRF fusion) → top 10
  2. rerank()    — cross-encoder scoring → top 5
  3. generate()  — GPT-4o answer from parent context

Usage:
    from rag.retrieval.query import ask

    answer = ask("What is the token expiry time?")
    print(answer)
"""

from rag.retrieval.retriever import retrieve
from rag.retrieval.reranker import rerank
from rag.retrieval.generator import generate


def ask(query: str) -> str:
    """Run the full RAG query pipeline and return the LLM's answer."""
    # Step 1: Hybrid retrieval — top 10 candidates
    results = retrieve(query, top_n=10)

    # Step 2: Rerank — cross-encoder narrows to top 5
    results = rerank(query, results, top_n=5)

    # Step 3: Generate answer from parent context
    return generate(query, results)
