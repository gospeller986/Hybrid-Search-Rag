from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# A header/footer line is treated as running boilerplate if it repeats as a
# page's first line on at least this fraction of pages.
RUNNING_HEADER_THRESHOLD = 0.3


@dataclass
class PageContent:
    source: str
    page_number: int
    text: str


def _find_running_header(pages_text: list[str]) -> str | None:
    first_lines = Counter()
    for text in pages_text:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            first_lines[lines[0]] += 1

    if not first_lines:
        return None

    header, count = first_lines.most_common(1)[0]
    if count / len(pages_text) >= RUNNING_HEADER_THRESHOLD:
        return header
    return None


def _strip_running_header(text: str, header: str | None) -> str:
    if header is None:
        return text.strip()

    lines = text.split("\n")
    if lines and lines[0].strip() == header:
        lines = lines[1:]
    return "\n".join(lines).strip()


def load_pdf(path: Path) -> list[PageContent]:
    doc = pymupdf.open(path)
    raw_pages_text = [page.get_text() for page in doc]

    running_header = _find_running_header(raw_pages_text)

    return [
        PageContent(
            source=path.name,
            page_number=i + 1,
            text=_strip_running_header(text, running_header),
        )
        for i, text in enumerate(raw_pages_text)
    ]
