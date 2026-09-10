import chromadb

from hybrid_search_rag.config import settings
from hybrid_search_rag.ingestion.chunking import Chunk

COLLECTION_NAME = "documents"


def chunk_id(source: str, index: int) -> str:
    return f"{source}::{index}"


def _get_client() -> chromadb.ClientAPI:
    persist_dir = settings.processed_data_dir / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection():
    return _get_client().get_or_create_collection(COLLECTION_NAME)


def _metadata_for(chunk: Chunk) -> dict:
    return {"source": chunk.source, "page_start": chunk.page_start, "page_end": chunk.page_end}


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> list[str]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    ids = [chunk_id(chunk.source, i) for i, chunk in enumerate(chunks)]
    get_collection().upsert(
        ids=ids,
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[_metadata_for(chunk) for chunk in chunks],
    )
    return ids


def query(query_embedding: list[float], n_results: int = 5) -> dict:
    return get_collection().query(query_embeddings=[query_embedding], n_results=n_results)
