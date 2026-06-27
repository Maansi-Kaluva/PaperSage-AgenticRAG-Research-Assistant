"""
Centralized guardrails for the RAG pipeline, applied at three stages:

1. INPUT GUARDRAILS  — validate the user query before it enters the graph
   (length limits, empty-input check, basic safety filtering).
2. RETRIEVAL GUARDRAILS — bound retrieval cost/size (max chunks, max
   context length) so a single query can't blow up latency or token cost.
3. GENERATION GUARDRAILS — cap LLM output length and provide a safe
   fallback message if generation produces nothing usable.

These are intentionally simple, deterministic checks (no extra LLM calls)
so they add negligible latency/cost while preventing the most common
failure modes: empty queries, oversized inputs, runaway context, and
unbounded output.
"""

# 1. INPUT GUARDRAILS

MIN_QUERY_LENGTH = 2
MAX_QUERY_LENGTH = 1000  # characters; protects against pasted-document-as-query


def validate_input_query(query: str) -> tuple[bool, str]:
    """
    Returns (is_valid, message). If invalid, `message` is a user-facing
    explanation that should be shown directly instead of running the graph.
    """
    if query is None or not query.strip():
        return False, "Please enter a question — the message can't be empty."

    stripped = query.strip()

    if len(stripped) < MIN_QUERY_LENGTH:
        return False, "Your question is too short. Please provide a bit more detail."

    if len(stripped) > MAX_QUERY_LENGTH:
        return (
            False,
            f"Your question is too long ({len(stripped)} characters). "
            f"Please shorten it to under {MAX_QUERY_LENGTH} characters "
            f"(e.g. summarize instead of pasting full text)."
        )

    return True, ""


# 2. RETRIEVAL GUARDRAILS

MAX_RETRIEVED_DOCS = 5          # hard cap on chunks passed to the LLM
MAX_CONTEXT_CHARS = 8000        # ≈ 2000 tokens — keeps context "medium", avoids
                                # context-window overflow and lost-in-the-middle


def cap_retrieved_docs(docs: list) -> list:
    """Hard cap on number of chunks, regardless of what retrieval returns."""
    return docs[:MAX_RETRIEVED_DOCS]


def build_context_with_citations(docs: list, max_chars: int = MAX_CONTEXT_CHARS):
    pieces, citations, total_len = [], [], 0
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Unknown source")
        page = doc.metadata.get("page_number", doc.metadata.get("page_index", "?"))
        tagged_text = f"[{i}] (Source: {title}, page {page})\n{doc.page_content}"
        if total_len + len(tagged_text) > max_chars:
            remaining = max_chars - total_len
            if remaining > 100:
                pieces.append(tagged_text[:remaining])
                citations.append({"index": i, "title": title, "page": page})
            break
        pieces.append(tagged_text)
        citations.append({"index": i, "title": title, "page": page})
        total_len += len(tagged_text)
    return "\n\n---\n\n".join(pieces), citations


def format_citations_block(citations: list) -> str:
    if not citations:
        return ""
    lines = [f"[{c['index']}] {c['title']}, page {c['page']}" for c in citations]
    return "\n\n**Sources:**\n" + "\n".join(lines)

# 3. GENERATION GUARDRAILS

# Output token cap — set on ChatOpenAI(max_tokens=...) at LLM init time.
# Kept here as a single source of truth so all LLM instances stay in sync.
MAX_OUTPUT_TOKENS = 1024

NO_ANSWER_FALLBACK = (
    "I wasn't able to generate a reliable answer for this question. "
    "This could be due to insufficient context in the uploaded papers, "
    "or a temporary issue with the language model. Please try rephrasing "
    "your question or asking again in a moment."
)


def validate_generation_output(answer: str) -> str:
    """
    Last-line guardrail on the generated answer: if the LLM returned
    nothing usable (empty string, whitespace-only, or None), substitute
    a safe, user-friendly fallback message instead of showing a blank
    response.
    """
    if not answer or not answer.strip():
        return NO_ANSWER_FALLBACK
    return answer