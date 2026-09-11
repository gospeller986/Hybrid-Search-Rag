from hybrid_search_rag.storage.keyword_index import KeywordIndex


def search(query: str, n_results: int = 5) -> list[dict]:
    return KeywordIndex.from_vector_store().search(query, n_results=n_results)
