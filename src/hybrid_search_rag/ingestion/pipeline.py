from pathlib import Path

from hybrid_search_rag.ingestion.chunking import chunk_pages
from hybrid_search_rag.ingestion.loaders import load_pdf
from hybrid_search_rag.retrieval.dense import embed_texts
from hybrid_search_rag.storage import vector_store


def ingest_pdf(path: Path) -> int:
    pages = load_pdf(path)
    chunks = chunk_pages(pages)
    embeddings = embed_texts([chunk.text for chunk in chunks])
    vector_store.add_chunks(chunks, embeddings)
    return len(chunks)
