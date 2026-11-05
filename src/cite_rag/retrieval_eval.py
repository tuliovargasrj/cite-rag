"""Measure how well each search mode finds the evidence. Needs no API key."""

from collections.abc import Iterable, Sequence
from typing import Any

from cite_rag.chunking import Chunk
from cite_rag.questions import Question, relevant_chunk_ids
from cite_rag.retrieval import Retriever, SearchMode

HIT_RATE_CUTOFFS = (1, 3, 5)
MRR_DEPTH = 10


def first_relevant_rank(ranked_ids: Iterable[str], relevant: set[str]) -> int | None:
    """Return the 1-based position of the first relevant chunk, if any."""
    return next((r for r, i in enumerate(ranked_ids, start=1) if i in relevant), None)


def summarize_ranks(ranks: Sequence[int | None]) -> dict[str, float]:
    """Compute hit rate at each cutoff and mean reciprocal rank."""
    total = len(ranks)
    metrics = {
        f"hit_rate_at_{cutoff}": sum(r is not None and r <= cutoff for r in ranks) / total
        for cutoff in HIT_RATE_CUTOFFS
    }
    metrics["mrr"] = sum(1 / r for r in ranks if r is not None) / total
    return {name: round(value, 4) for name, value in metrics.items()}


def evaluate_retrieval(
    retriever: Retriever, questions: Sequence[Question], chunks: Sequence[Chunk]
) -> dict[str, Any]:
    """Score every mode on the answerable questions, keeping each question's rank."""
    answerable = [question for question in questions if question.answerable]
    modes: dict[str, Any] = {}
    for mode in SearchMode:
        ranks = {}
        for question in answerable:
            hits = retriever.search(question.question, MRR_DEPTH, mode)
            ranks[question.id] = first_relevant_rank(
                [hit.chunk.id for hit in hits], relevant_chunk_ids(question, chunks)
            )
        modes[mode.value] = {
            "metrics": summarize_ranks(list(ranks.values())),
            "ranks": ranks,
        }
    return {"questions": len(answerable), "mrr_depth": MRR_DEPTH, "modes": modes}
