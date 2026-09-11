from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5


class SourceRef(BaseModel):
    source: str
    page_start: int
    page_end: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class IngestRequest(BaseModel):
    filename: str


class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int
