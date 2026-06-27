from sentence_transformers import CrossEncoder

_cross_encoder = CrossEncoder("BAAI/bge-reranker-large")
try:
    _cross_encoder.predict(
        [("warmup", "warmup")]
    )
except Exception:
    pass

def rerank_documents(query: str, docs: list, top_k: int = 5) -> list:
    if not docs:
        return [], 0.0
    pairs = [(query, doc.page_content[:512]) for doc in docs]
    scores = _cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top = ranked[:top_k]
    avg_score = round(float(sum(s for s, _ in top) / len(top)), 4)
    return [doc for _, doc in top], avg_score

# with warmup:  App starts -> load reranker immediately -> first query asked -> answer quick with low latency
# without warmup:  App starts -> first query asked -> load reranker  -> answer with latency