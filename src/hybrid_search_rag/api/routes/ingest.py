from fastapi import APIRouter, HTTPException

from hybrid_search_rag.api.schemas import IngestRequest, IngestResponse
from hybrid_search_rag.config import settings
from hybrid_search_rag.ingestion.pipeline import ingest_pdf

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    raw_dir = settings.raw_data_dir.resolve()
    path = (raw_dir / request.filename).resolve()

    if not path.is_relative_to(raw_dir):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found in data/raw: {request.filename}")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported")

    chunk_count = ingest_pdf(path)
    return IngestResponse(source=path.name, chunks_indexed=chunk_count)
