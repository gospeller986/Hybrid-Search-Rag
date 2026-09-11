from hybrid_search_rag.retrieval import dense, sparse

# Standard constant from the original RRF paper (Cormack et al., 2009) — large
# enough that rank 1 vs rank 2 doesn't dominate the score, so a chunk ranked
# decently by *both* methods can outscore one ranked #1 by only one of them.
RRF_K = 60

# Each underlying search pulls more candidates than we ultimately return,
# since fusion needs a wide-enough pool from each method to find chunks that
# rank well overall even if they weren't #1 anywhere individually.
CANDIDATE_POOL = 20


def _rrf_scores(results: list[dict]) -> dict[str, float]:
    return {result["id"]: 1.0 / (RRF_K + rank) for rank, result in enumerate(results, start=1)}


def search(query: str, n_results: int = 5) -> list[dict]:
    dense_results = dense.search(query, n_results=CANDIDATE_POOL)
    sparse_results = sparse.search(query, n_results=CANDIDATE_POOL)

    dense_scores = _rrf_scores(dense_results)
    sparse_scores = _rrf_scores(sparse_results)

    lookup = {result["id"]: result for result in dense_results + sparse_results}

    fused_scores = {
        id_: dense_scores.get(id_, 0.0) + sparse_scores.get(id_, 0.0) for id_ in lookup
    }
    ranked_ids = sorted(fused_scores, key=lambda id_: fused_scores[id_], reverse=True)[:n_results]

    return [{**lookup[id_], "score": fused_scores[id_]} for id_ in ranked_ids]
