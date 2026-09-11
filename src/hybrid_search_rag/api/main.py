from fastapi import FastAPI

from hybrid_search_rag.api.routes import ingest, search

app = FastAPI(title="Hybrid Search RAG")

app.include_router(search.router, tags=["search"])
app.include_router(ingest.router, tags=["ingest"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
