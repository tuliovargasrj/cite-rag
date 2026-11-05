"""Local text embeddings. The model is downloaded once and then runs on the CPU."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models are trained with this prefix on queries only, and fastembed does not add it.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Measured speed barely changes with batch size, while fastembed's default of 256 pads
# long passages into several gigabytes of memory.
_BATCH_SIZE = 32

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Return one unit-length row per text."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """Return one unit-length vector."""
        ...


class FastEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Imported here because it is slow to load and most commands' help never needs it.
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        rows = self._model.embed(list(texts), batch_size=_BATCH_SIZE)
        return normalize(np.array(list(rows), dtype=np.float32))

    def embed_query(self, text: str) -> Vectors:
        vector = next(iter(self._model.embed([_QUERY_PREFIX + text])))
        return normalize(np.asarray(vector, dtype=np.float32))


def normalize(rows: Vectors) -> Vectors:
    """Scale rows to unit length so that a dot product is a cosine similarity."""
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return np.asarray(rows / np.where(norms == 0, 1, norms), dtype=np.float32)
