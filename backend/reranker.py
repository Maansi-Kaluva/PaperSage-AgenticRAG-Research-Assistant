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
        return []
    pairs = [(query, doc.page_content[:512]) for doc in docs]
    scores = _cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# with warmup:  App starts -> load reranker immediately -> first query asked -> answer quick with low latency
# without warmup:  App starts -> first query asked -> load reranker  -> answer with latency