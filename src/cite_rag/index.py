"""Build, save, and load the embedding matrix for a list of chunks."""

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cite_rag.chunking import Chunk
from cite_rag.embeddings import Embedder, Vectors

DEFAULT_INDEX_DIR = Path("data/index")

_EMBEDDINGS_FILE = "embeddings.npy"
_META_FILE = "meta.json"


class StaleIndexError(Exception):
    """The saved index is missing or was built from other chunks or another model."""


@dataclass(frozen=True)
class Index:
    chunks: Sequence[Chunk]
    embeddings: Vectors


def build_index(chunks: Sequence[Chunk], embedder: Embedder) -> Index:
    return Index(chunks, embedder.embed_documents([c.search_text for c in chunks]))


def save_index(
    index: Index, model_name: str, directory: Path = DEFAULT_INDEX_DIR
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / _EMBEDDINGS_FILE, index.embeddings)
    meta = {"model": model_name, "fingerprint": fingerprint(index.chunks)}
    (directory / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(
    chunks: Sequence[Chunk], model_name: str, directory: Path = DEFAULT_INDEX_DIR
) -> Index:
    """Load the matrix, refusing one whose rows may not line up with these chunks."""
    meta_path = directory / _META_FILE
    if not meta_path.exists():
        raise StaleIndexError(f"no index in {directory}; run `cite-rag index`")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {"model": model_name, "fingerprint": fingerprint(chunks)}
    if meta != expected:
        raise StaleIndexError("index does not match the corpus; run `cite-rag index`")
    return Index(chunks, np.load(directory / _EMBEDDINGS_FILE))


def fingerprint(chunks: Iterable[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.id}\0{chunk.search_text}\0".encode())
    return digest.hexdigest()
