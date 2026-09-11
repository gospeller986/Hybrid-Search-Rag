import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from hybrid_search_rag.api.schemas import QueryRequest, QueryResponse
from hybrid_search_rag.generation import llm

router = APIRouter()


@router.post("/search", response_model=QueryResponse)
def search(request: QueryRequest) -> dict:
    return llm.answer(request.question, n_results=request.n_results)


def _event_stream(question: str, n_results: int):
    # Newline-delimited JSON events rather than a plain-text body: sources
    # (which now carry full chunk text, not just page numbers) can be
    # arbitrarily long, so they don't belong in an HTTP header. The client
    # reads the first line for sources, then streams token events after it.
    sources, tokens = llm.stream_answer(question, n_results=n_results)
    yield json.dumps({"type": "sources", "sources": sources}) + "\n"
    for token in tokens:
        yield json.dumps({"type": "token", "text": token}) + "\n"
    yield json.dumps({"type": "done"}) + "\n"


@router.post("/search/stream")
def search_stream(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request.question, request.n_results),
        media_type="application/x-ndjson",
    )
