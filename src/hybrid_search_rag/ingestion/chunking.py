from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from hybrid_search_rag.config import settings
from hybrid_search_rag.ingestion.loaders import PageContent

# Sized against the embedding model's own limit (see config.embedding_model_name),
# not an arbitrary guess: all-MiniLM-L6-v2 truncates at 256 tokens.
TARGET_CHUNK_TOKENS = 200
OVERLAP_TOKENS = 30


@dataclass
class Chunk:
    source: str
    page_start: int
    page_end: int
    text: str
    token_count: int


@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(f"sentence-transformers/{settings.embedding_model_name}")


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


@lru_cache(maxsize=1)
def _get_splitter() -> RecursiveCharacterTextSplitter:
    # Measures chunk size in real tokens from our embedding model's own
    # tokenizer (not characters), and falls back all the way to per-character
    # splitting if a stretch of text (e.g. a table row) has no paragraph,
    # sentence, or line separators to split on.
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _get_tokenizer(),
        chunk_size=TARGET_CHUNK_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
    )


def _page_for_offset(offset: int, page_offsets: list[tuple[int, int, int]]) -> int:
    for page_number, start, end in page_offsets:
        if start <= offset < end:
            return page_number
    return page_offsets[-1][0]


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    full_text = ""
    page_offsets: list[tuple[int, int, int]] = []
    for page in pages:
        start = len(full_text)
        full_text += page.text + "\n"
        page_offsets.append((page.page_number, start, len(full_text)))

    texts = _get_splitter().split_text(full_text)

    chunks = []
    search_from = 0
    for text in texts:
        # split_text doesn't report offsets, so we recover each chunk's
        # position by searching forward from the previous chunk's start —
        # not its end, since overlapping chunks can start before that.
        offset = full_text.find(text, search_from)
        if offset == -1:
            offset = search_from
        search_from = offset

        chunks.append(
            Chunk(
                source=pages[0].source,
                page_start=_page_for_offset(offset, page_offsets),
                page_end=_page_for_offset(max(offset, offset + len(text) - 1), page_offsets),
                text=text,
                token_count=_count_tokens(text),
            )
        )

    return chunks
