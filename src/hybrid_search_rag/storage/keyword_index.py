import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from hybrid_search_rag.storage import vector_store

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class KeywordIndex:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    bm25: BM25Okapi

    @classmethod
    def from_vector_store(cls) -> "KeywordIndex":
        # rank-bm25 has no persistence of its own, so rather than maintaining
        # a second on-disk index that could drift out of sync, we rebuild it
        # in memory from Chroma, which stays the single source of truth for
        # chunk text and metadata.
        data = vector_store.get_collection().get(include=["documents", "metadatas"])
        documents = data["documents"]
        return cls(
            ids=data["ids"],
            documents=documents,
            metadatas=data["metadatas"],
            bm25=BM25Okapi([_tokenize(doc) for doc in documents]),
        )

    def search(self, query_text: str, n_results: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [
            {
                "id": self.ids[i],
                "document": self.documents[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            for i in ranked
        ]
