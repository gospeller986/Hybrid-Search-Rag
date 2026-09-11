from fastapi import APIRouter

from hybrid_search_rag.api.schemas import QueryRequest, QueryResponse
from hybrid_search_rag.generation import llm

router = APIRouter()


@router.post("/search", response_model=QueryResponse)
def search(request: QueryRequest) -> dict:
    return llm.answer(request.question, n_results=request.n_results)
