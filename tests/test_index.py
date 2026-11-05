from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cite_rag.chunking import Chunk
from cite_rag.index import StaleIndexError, build_index, load_index, save_index
from tests.fakes import FakeEmbedder

CHUNKS = [
    Chunk("pep-0001:0000", 1, "One", "Abstract", "first text"),
    Chunk("pep-0001:0001", 1, "One", "Rationale", "second text"),
]


def test_saved_index_round_trips(tmp_path: Path) -> None:
    index = build_index(CHUNKS, FakeEmbedder())
    save_index(index, "model", tmp_path)
    loaded = load_index(CHUNKS, "model", tmp_path)
    assert loaded.chunks == CHUNKS
    np.testing.assert_array_equal(loaded.embeddings, index.embeddings)


def test_rows_are_unit_length() -> None:
    norms = np.linalg.norm(build_index(CHUNKS, FakeEmbedder()).embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)


def test_missing_index_is_reported(tmp_path: Path) -> None:
    with pytest.raises(StaleIndexError, match="cite-rag index"):
        load_index(CHUNKS, "model", tmp_path)


def test_changed_chunks_invalidate_the_index(tmp_path: Path) -> None:
    save_index(build_index(CHUNKS, FakeEmbedder()), "model", tmp_path)
    edited = [replace(CHUNKS[0], text="edited text"), CHUNKS[1]]
    with pytest.raises(StaleIndexError):
        load_index(edited, "model", tmp_path)


def test_changed_model_invalidates_the_index(tmp_path: Path) -> None:
    save_index(build_index(CHUNKS, FakeEmbedder()), "model", tmp_path)
    with pytest.raises(StaleIndexError):
        load_index(CHUNKS, "another-model", tmp_path)
