from functools import lru_cache

from sentence_transformers import SentenceTransformer

from hybrid_search_rag.config import settings
from hybrid_search_rag.storage import vector_store


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    return _get_model().encode(query, show_progress_bar=False).tolist()


def search(query: str, n_results: int = 5) -> list[dict]:
    results = vector_store.query(embed_query(query), n_results=n_results)
    return [
        {"id": id_, "document": document, "metadata": metadata, "score": distance}
        for id_, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
