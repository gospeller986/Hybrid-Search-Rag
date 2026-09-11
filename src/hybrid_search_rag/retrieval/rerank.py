from functools import lru_cache

from sentence_transformers import CrossEncoder

# A cross-encoder scores (query, chunk) jointly in one forward pass, rather
# than comparing two independently-computed vectors like dense.py does. This
# is far more accurate at judging relevance, but too expensive to run over an
# entire corpus — which is exactly why it only re-scores the candidate pool
# fusion.search() already narrowed things down to, not the whole collection.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    pairs = [(query, candidate["document"]) for candidate in candidates]
    scores = _get_model().predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [{**candidate, "score": float(score)} for candidate, score in ranked[:top_k]]
