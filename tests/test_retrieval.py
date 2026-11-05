import numpy as np

from cite_rag.chunking import Chunk
from cite_rag.embeddings import Vectors
from cite_rag.index import Index, build_index
from cite_rag.retrieval import Retriever, SearchMode, tokenize
from tests.fakes import FakeEmbedder


def chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(f"pep-0001:{ordinal:04d}", 1, "Example", "Section", text)


CHUNKS = [
    chunk(0, "limit lines to seventy nine characters"),
    chunk(1, "coroutines are declared with async def"),
    chunk(2, "the walrus operator assigns inside an expression"),
    chunk(3, "dataclasses generate an init method"),
]


def retriever() -> Retriever:
    embedder = FakeEmbedder()
    return Retriever(build_index(CHUNKS, embedder), embedder)


def test_tokenize_lowercases_and_keeps_identifiers() -> None:
    assert tokenize("Use __init__ in PEP-8!") == ["use", "__init__", "in", "pep", "8"]


def test_bm25_finds_the_chunk_sharing_a_rare_word() -> None:
    hits = retriever().search("walrus", mode=SearchMode.BM25)
    assert hits[0].chunk.id == "pep-0001:0002"


def test_bm25_returns_nothing_when_no_word_matches() -> None:
    assert retriever().search("zebra", mode=SearchMode.BM25) == []


def test_dense_ranks_by_cosine_similarity() -> None:
    hits = retriever().search("async def coroutines", mode=SearchMode.DENSE)
    assert hits[0].chunk.id == "pep-0001:0001"
    assert [hit.score for hit in hits] == sorted((h.score for h in hits), reverse=True)


def test_top_k_limits_the_results() -> None:
    assert len(retriever().search("the", top_k=2, mode=SearchMode.DENSE)) == 2


class ScriptedEmbedder:
    """Return a fixed query vector, to control the dense ranking exactly."""

    model_name = "scripted"

    def __init__(self, query_vector: list[float]) -> None:
        self._query_vector = np.array(query_vector, dtype=np.float32)

    def embed_documents(self, texts: object) -> Vectors:
        raise NotImplementedError

    def embed_query(self, text: str) -> Vectors:
        return self._query_vector


def test_hybrid_prefers_a_chunk_found_by_both_methods() -> None:
    chunks = [
        chunk(0, "alpha alpha alpha"),  # Best keyword match, worst embedding match.
        chunk(1, "unrelated words here"),  # Best embedding match, no keyword match.
        chunk(2, "alpha and more words"),  # Second in both rankings.
        # BM25 gives a term no weight when it appears in half of the chunks or more.
        chunk(3, "filler text one"),
        chunk(4, "filler text two"),
        chunk(5, "filler text three"),
    ]
    embeddings = np.array(
        [[0, 1], [1, 0], [0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        dtype=np.float32,
    )
    index = Index(chunks, embeddings)
    search = Retriever(index, ScriptedEmbedder([1, 0])).search
    assert search("alpha", top_k=1, mode=SearchMode.BM25)[0].chunk.id == "pep-0001:0000"
    assert search("alpha", top_k=1, mode=SearchMode.DENSE)[0].chunk.id == "pep-0001:0001"
    assert search("alpha", top_k=1)[0].chunk.id == "pep-0001:0002"


def test_hybrid_still_returns_dense_results_without_keyword_matches() -> None:
    hits = retriever().search("zebra", top_k=2)
    assert len(hits) == 2


def test_results_are_deterministic() -> None:
    search = retriever().search
    assert search("the method") == search("the method")
