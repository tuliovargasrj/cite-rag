"""Load the PEP corpus and enforce that every file is public domain or CC0."""

import hashlib
import json
import re
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path

DEFAULT_CORPUS_DIR = Path("data/corpus")

_COPYRIGHT_HEADING = re.compile(
    r"^ ?Copyright\n[=\-~^]{3,}\n(.*)", re.MULTILINE | re.DOTALL
)


class CorpusError(Exception):
    """The corpus on disk does not match its recorded sources."""


@dataclass(frozen=True)
class Document:
    pep: int
    title: str
    text: str


def normalize_text(text: str) -> str:
    """Collapse whitespace, so that line wrapping never affects a text comparison."""
    return " ".join(text.split())


def parse_headers(text: str) -> dict[str, str]:
    """Read the RFC 822 style header block at the top of a PEP."""
    message = Parser().parsestr(text, headersonly=True)
    return {name: normalize_text(value) for name, value in message.items()}


def detect_license(text: str) -> str | None:
    """Return the license stated in the PEP's Copyright section, if it is allowed."""
    match = _COPYRIGHT_HEADING.search(text)
    if match is None:
        return None
    statement = normalize_text(match.group(1)).lower()
    public_domain = "public domain" in statement
    cc0 = "cc0" in statement
    if public_domain and cc0:
        return "Public domain or CC0-1.0"
    if cc0:
        return "CC0-1.0"
    return "Public domain" if public_domain else None


def load_corpus(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Document]:
    """Load every document listed in sources.json, verifying hash and license."""
    manifest = corpus_dir / "sources.json"
    documents = []
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        raw = (corpus_dir / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise CorpusError(f"{entry['file']} does not match its recorded sha256")
        text = raw.decode("utf-8")
        if detect_license(text) != entry["license"]:
            raise CorpusError(f"{entry['file']} does not state its recorded license")
        documents.append(Document(entry["pep"], entry["title"], text))
    return documents
