import httpx

from hybrid_search_rag.config import settings
from hybrid_search_rag.retrieval import fusion

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


def _call_ollama(messages: list[dict]) -> str:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={"model": settings.ollama_model, "messages": messages, "stream": False},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def answer(query: str, n_results: int = 5) -> dict:
    chunks = fusion.search(query, n_results=n_results)
    response_text = _call_ollama(build_messages(query, chunks))
    return {
        "answer": response_text,
        "sources": [
            {
                "source": chunk["metadata"]["source"],
                "page_start": chunk["metadata"]["page_start"],
                "page_end": chunk["metadata"]["page_end"],
            }
            for chunk in chunks
        ],
    }
