"""Keyword, embedding, and hybrid search over the indexed chunks."""

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from rank_bm25 import BM25Okapi

from cite_rag.chunking import Chunk
from cite_rag.embeddings import Embedder
from cite_rag.index import Index

DEFAULT_TOP_K = 5
# How deep each ranking goes before fusion. Wider than top_k so that a chunk ranked
# modestly by both methods can still beat one ranked well by a single method.
_FUSION_DEPTH = 50
# The constant from the original RRF paper. Keeping it fixed means no weight is tuned
# against the evaluation questions.
_RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9_]+")


class SearchMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, index: Index, embedder: Embedder) -> None:
        self._chunks = index.chunks
        self._embeddings = index.embeddings
        self._embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.search_text) for c in index.chunks])

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[Hit]:
        if mode is SearchMode.BM25:
            ranked = self._rank_bm25(query)
        elif mode is SearchMode.DENSE:
            ranked = self._rank_dense(query)
        else:
            ranked = _reciprocal_rank_fusion(
                self._rank_bm25(query)[:_FUSION_DEPTH],
                self._rank_dense(query)[:_FUSION_DEPTH],
            )
        return [Hit(self._chunks[i], score) for i, score in ranked[:top_k]]

    def _rank_bm25(self, query: str) -> list[tuple[int, float]]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        # A chunk that shares no term with the query is not a keyword match at all.
        return [(i, score) for i, score in _ranked(scores) if score > 0]

    def _rank_dense(self, query: str) -> list[tuple[int, float]]:
        scores = self._embeddings @ self._embedder.embed_query(query)
        return _ranked(scores)


def _ranked(scores: np.ndarray) -> list[tuple[int, float]]:
    # A stable sort keeps results deterministic when scores tie.
    order = np.argsort(-scores, kind="stable")
    return [(int(i), float(scores[i])) for i in order]


def _reciprocal_rank_fusion(
    *rankings: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Combine rankings by position, since BM25 and cosine scores are not comparable."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (chunk_index, _) in enumerate(ranking, start=1):
            fused[chunk_index] += 1 / (_RRF_K + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
