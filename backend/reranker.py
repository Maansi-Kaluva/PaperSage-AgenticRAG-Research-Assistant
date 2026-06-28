import os

ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"

if ENABLE_RERANKER:
    from sentence_transformers import CrossEncoder
    
    _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

    try:
        _cross_encoder.predict([("warmup", "warmup")])
    except Exception:
        pass
else:
    _cross_encoder = None


def rerank_documents(query: str, docs: list, top_k: int = 5) -> tuple[list, float]:
    if not docs:
        return [], 0.0

    # Skip reranking on low-memory deployments (e.g. EC2 free tier)
    if not ENABLE_RERANKER:
        return docs[:top_k], 0.0

    pairs = [(query, doc.page_content[:512]) for doc in docs]
    scores = _cross_encoder.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top = ranked[:top_k]

    avg_score = round(float(sum(s for s, _ in top) / len(top)), 4)

    return [doc for _, doc in top], avg_score

# with warmup:  App starts -> load reranker immediately -> first query asked -> answer quick with low latency
# without warmup:  App starts -> first query asked -> load reranker  -> answer with latency