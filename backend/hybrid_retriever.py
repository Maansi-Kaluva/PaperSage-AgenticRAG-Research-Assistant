from rank_bm25 import BM25Okapi

from backend.vector_store import search
from backend.reranker import (
    rerank_documents,
)

def hybrid_search(query: str, session_id: str, paper_title: str | None = None):

    vector_docs = search(
        query=query,
        session_id=session_id,
        k=15,
        paper_title = paper_title,
    )
    if not vector_docs:
        return [], 0.0

    corpus = [
        d.page_content
        for d in vector_docs
    ]

    tokenized = [
        text.split()
        for text in corpus
    ]

    bm25 = BM25Okapi(tokenized)

    bm25_scores = bm25.get_scores(
        query.split()
    )

    top_bm25 = sorted(
        zip(bm25_scores, vector_docs),
        key=lambda x: x[0],
        reverse=True,
    )[:10]

    merged = vector_docs + [
        d for _, d in top_bm25
    ]

    unique_docs = []

    seen = set()

    for d in merged:
        text = d.page_content[:200]

        if text in seen:
            continue

        seen.add(text)

        unique_docs.append(d)

    return rerank_documents(
        query,
        unique_docs,
        top_k=5,
    )