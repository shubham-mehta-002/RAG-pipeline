"""LLM response generator — builds a prompt from retrieved context and calls GPT-4o.

Takes the reranked results, uses their parent content as context (full section
text), builds a system + user prompt, and returns the LLM's answer.

The prompt is structured so the LLM:
  - Only answers from the provided context
  - Cites which source/section the answer came from
  - Says "I don't know" if the context doesn't contain the answer
    (avoids hallucination)

Model: gpt-4o — best reasoning quality, same API key already in use.
"""

import os
from openai import OpenAI

from rag.retrieval.retriever import RetrievalResult


_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
_MODEL = "gpt-4o"


# --- Prompt builder ---

def _build_context_block(results: list[RetrievalResult]) -> str:
    """Format retrieved results into a numbered context block for the prompt."""
    blocks = []
    for i, result in enumerate(results, start=1):
        # Use parent content if available — it's the full section
        content = result.parent_content or result.child_content
        section = " > ".join(result.metadata.get("section_path", []))
        source = result.metadata.get("title", result.metadata.get("source", "unknown"))
        page = result.metadata.get("page")

        header = f"[{i}] Source: {source}"
        if section:
            header += f" | Section: {section}"
        if page:
            header += f" | Page: {page}"

        blocks.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(blocks)


# --- Public API ---

def generate(query: str, results: list[RetrievalResult]) -> str:
    """Generate an answer to *query* using the retrieved context.

    Returns the LLM's response as a plain string.
    If no results are provided, returns a fallback message.
    """
    if not results:
        return "I could not find any relevant information to answer your question."

    context = _build_context_block(results)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided context. If the answer is not in the context, say "
        "'I don't have enough information to answer this question.' "
        "Always mention which source and section your answer comes from."
    )

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature — factual, consistent answers
    )

    return response.choices[0].message.content.strip()
