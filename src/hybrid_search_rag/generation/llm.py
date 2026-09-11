import json
from collections.abc import Iterator

import httpx

from hybrid_search_rag.config import settings
from hybrid_search_rag.retrieval import dense, fusion

# Cosine distance (lower = closer) from the top dense match, above which we
# treat the query as having no relevant content in the corpus at all. Chosen
# from real measurements: genuinely relevant queries scored 0.55-0.96, clearly
# unrelated ones (e.g. "capital of France") scored 1.45-1.84 — a real gap, not
# a guess. This exists because RRF's fused score only reflects rank position,
# not match quality, so it can't tell "best of great matches" apart from
# "best of terrible matches" — only the raw distance carries that signal.
MAX_RELEVANT_DISTANCE = 1.2

NO_CONTEXT_ANSWER = "I couldn't find anything relevant to this question in the indexed documents."

SYSTEM_PROMPT = (
    "You are a clinical reference assistant. Answer the user's question using ONLY the "
    "provided context excerpts below. Cite the page number(s) for every claim, in the form "
    "(p. X) or (pp. X-Y). If the context does not contain enough information to answer, say "
    "so explicitly rather than guessing. This is for informational purposes only and is not "
    "a substitute for professional medical judgment."
)


def _page_label(metadata: dict) -> str:
    start, end = metadata["page_start"], metadata["page_end"]
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{_page_label(chunk['metadata'])}] {chunk['document']}" for chunk in chunks)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {query}"},
    ]


def _sources_for(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source": chunk["metadata"]["source"],
            "page_start": chunk["metadata"]["page_start"],
            "page_end": chunk["metadata"]["page_end"],
            "text": chunk["document"],
        }
        for chunk in chunks
    ]


def _call_ollama(messages: list[dict]) -> str:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={"model": settings.ollama_model, "messages": messages, "stream": False},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _stream_ollama(messages: list[dict]) -> Iterator[str]:
    with httpx.stream(
        "POST",
        f"{settings.ollama_base_url}/api/chat",
        json={"model": settings.ollama_model, "messages": messages, "stream": True},
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done"):
                break


def _has_relevant_context(query: str) -> bool:
    top_match = dense.search(query, n_results=1)
    return bool(top_match) and top_match[0]["score"] <= MAX_RELEVANT_DISTANCE


def answer(query: str, n_results: int = 5) -> dict:
    if not _has_relevant_context(query):
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}

    chunks = fusion.search(query, n_results=n_results)
    response_text = _call_ollama(build_messages(query, chunks))
    return {"answer": response_text, "sources": _sources_for(chunks)}


def stream_answer(query: str, n_results: int = 5) -> tuple[list[dict], Iterator[str]]:
    """Returns (sources, token_stream). Sources are known before generation
    starts, since retrieval happens first — callers can show them immediately
    rather than waiting for the streamed answer to finish.
    """
    if not _has_relevant_context(query):
        return [], iter([NO_CONTEXT_ANSWER])

    chunks = fusion.search(query, n_results=n_results)
    return _sources_for(chunks), _stream_ollama(build_messages(query, chunks))
