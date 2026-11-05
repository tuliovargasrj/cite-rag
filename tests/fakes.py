"""Deterministic stand-ins for the embedding model and the LLM."""

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cite_rag.embeddings import Vectors, normalize
from cite_rag.prompts import ModelAnswer, ModelCitation, ModelClaim

_DIMENSIONS = 256


class FakeEmbedder:
    """Hash each word to one dimension, so shared words mean similar vectors."""

    model_name = "fake-embedder"

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return normalize(np.array([_bag_of_words(text) for text in texts]))

    def embed_query(self, text: str) -> Vectors:
        return normalize(_bag_of_words(text))


class FakeLLM:
    """Return one canned reply and record every prompt it receives."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._reply


def model_reply(
    *claims: tuple[str, list[tuple[str, str]]], answerable: bool = True
) -> str:
    """Build a reply in the model's contract from (text, [(chunk_id, quote)]) claims."""
    return ModelAnswer(
        answerable=answerable,
        claims=[
            ModelClaim(
                text=text,
                citations=[ModelCitation(chunk_id=i, quote=q) for i, q in citations],
            )
            for text, citations in claims
        ],
    ).model_dump_json()


def write_corpus(
    directory: Path, text: str, license_name: str = "Public domain", pep: int = 1
) -> None:
    """Write a one-document corpus with a matching sources.json."""
    raw = text.encode("utf-8")
    name = f"pep-{pep:04}.rst"
    (directory / name).write_bytes(raw)
    entry = {
        "file": name,
        "pep": pep,
        "title": "Example",
        "status": "Final",
        "license": license_name,
        "source_url": f"https://example.test/{name}",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (directory / "sources.json").write_text(json.dumps([entry]), encoding="utf-8")


def _bag_of_words(text: str) -> Vectors:
    vector = np.zeros(_DIMENSIONS, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:4]) % _DIMENSIONS] += 1
    return vector
