from functools import lru_cache

from sentence_transformers import SentenceTransformer

from hybrid_search_rag.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    return _get_model().encode(query, show_progress_bar=False).tolist()
