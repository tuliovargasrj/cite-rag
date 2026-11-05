"""Split PEPs into citable chunks along their reStructuredText sections."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from cite_rag.corpus import Document, parse_headers

MAX_CHUNK_CHARS = 1200

_ADORNMENT = re.compile(r"^([=\-~^\"*+#])\1{2,}$")
_METADATA_FIELDS = ("Title", "Author", "Status", "Type", "Created", "Python-Version")
# These sections hold links, credits, or legal text, never an answer about the proposal.
_SKIPPED_SECTIONS = {
    "acknowledgements",
    "acknowledgments",
    "copyright",
    "footnotes",
    "references",
}


@dataclass(frozen=True)
class Chunk:
    id: str
    pep: int
    title: str
    section: str
    text: str

    @property
    def search_text(self) -> str:
        """Text to index. The PEP and section names give a short passage its context."""
        return f"PEP {self.pep}: {self.title}\n{self.section}\n\n{self.text}"


@dataclass(frozen=True)
class _Section:
    path: str
    body: str


def chunk_corpus(documents: Iterable[Document]) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document)]


def chunk_document(document: Document) -> list[Chunk]:
    pieces = [("Metadata", _metadata_text(document))]
    for section in _split_sections(document.text):
        pieces.extend(
            (section.path, text) for text in _pack(_paragraphs(section.body), "\n\n")
        )
    return [
        Chunk(
            id=f"pep-{document.pep:04d}:{ordinal:04d}",
            pep=document.pep,
            title=document.title,
            section=path,
            text=text,
        )
        for ordinal, (path, text) in enumerate(pieces)
    ]


def _metadata_text(document: Document) -> str:
    # Authors and target versions live only in this header, so it must be retrievable.
    headers = parse_headers(document.text)
    lines = [f"PEP: {document.pep}"]
    lines += [f"{name}: {headers[name]}" for name in _METADATA_FIELDS if name in headers]
    return "\n".join(lines)


def _split_sections(text: str) -> list[_Section]:
    """Walk the headings, skipping the header block and the unwanted sections."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if not line.strip()), len(lines))

    adornments: list[str] = []
    stack: list[str] = []
    sections: list[_Section] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        body.clear()
        if content and stack and not _is_skipped(stack):
            sections.append(_Section(" > ".join(stack), content))

    index = start
    while index < len(lines):
        heading = _heading_at(lines, index)
        if heading is None:
            body.append(lines[index])
            index += 1
            continue
        title, style, consumed = heading
        flush()
        if style not in adornments:
            adornments.append(style)
        level = adornments.index(style)
        stack[level:] = [title]
        index += consumed
    flush()
    return sections


def _heading_at(lines: list[str], index: int) -> tuple[str, str, int] | None:
    """Return (title, style, lines consumed) if a heading starts at this line."""
    line = lines[index]
    following = lines[index + 1] if index + 1 < len(lines) else ""
    after = lines[index + 2] if index + 2 < len(lines) else ""

    overline = _ADORNMENT.match(line)
    if overline and following.strip() and after == line:
        # The style includes the overline flag: "=" with and without one are two levels.
        return following.strip(), overline.group(1) * 2, 3

    is_title = line.strip() and not line[0].isspace() and not _ADORNMENT.match(line)
    underline = _ADORNMENT.match(following)
    if is_title and underline and len(following) >= len(line.rstrip()):
        return line.strip(), underline.group(1), 2
    return None


def _is_skipped(stack: list[str]) -> bool:
    return any(title.lower() in _SKIPPED_SECTIONS for title in stack)


def _pack(pieces: Iterable[str], separator: str) -> Iterator[str]:
    """Join pieces greedily so that no result exceeds MAX_CHUNK_CHARS."""
    current = ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
            continue
        if current:
            yield current
        current = piece
    if current:
        yield current


def _paragraphs(body: str) -> Iterator[str]:
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip("\n")
        if paragraph.strip():
            yield from _pack(_line_pieces(paragraph), "\n")


def _line_pieces(paragraph: str) -> Iterator[str]:
    """Yield each line, cut mid-line only if it exceeds the limit on its own."""
    for line in paragraph.splitlines():
        for start in range(0, len(line), MAX_CHUNK_CHARS):
            yield line[start : start + MAX_CHUNK_CHARS]
