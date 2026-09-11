import json
from collections.abc import Iterator

import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Hybrid Search RAG", page_icon="🔍")
st.title("Hybrid Search RAG")
st.caption("WHO Guideline for the Pharmacological Treatment of Hypertension in Adults")

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources):
            page_label = (
                f"p. {source['page_start']}"
                if source["page_start"] == source["page_end"]
                else f"pp. {source['page_start']}-{source['page_end']}"
            )
            st.markdown(f"**{source['source']}** — {page_label}")
            st.markdown(f"> {source['text']}")
            if i < len(sources) - 1:
                st.divider()


def stream_answer(question: str) -> tuple[list[dict], Iterator[str]]:
    # Manually open the streamed response (rather than a `with` block) so we
    # can read the first NDJSON line (sources) immediately and hand back a
    # generator that only reads the rest of the body lazily, event by event,
    # as the caller consumes it.
    client = httpx.Client(timeout=120.0)
    response = client.send(
        client.build_request("POST", f"{API_URL}/search/stream", json={"question": question}),
        stream=True,
    )
    response.raise_for_status()

    lines = response.iter_lines()
    sources_event = json.loads(next(lines))
    sources = sources_event.get("sources", [])

    def token_generator() -> Iterator[str]:
        try:
            for line in lines:
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "done":
                    break
        finally:
            response.close()
            client.close()

    return sources, token_generator()


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            render_sources(turn["sources"])

question = st.chat_input("Ask a question about the guideline...")

if question:
    st.session_state.history.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            sources, tokens = stream_answer(question)
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the backend: {exc}")
        else:
            full_answer = st.write_stream(tokens)
            render_sources(sources)
            st.session_state.history.append(
                {"role": "assistant", "content": full_answer, "sources": sources}
            )
