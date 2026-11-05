from pathlib import Path

import pytest

from cite_rag.chunking import Chunk, chunk_corpus
from cite_rag.corpus import load_corpus
from cite_rag.index import build_index
from cite_rag.questions import (
    Evidence,
    MissingEvidenceError,
    Question,
    check_evidence,
    load_questions,
    relevant_chunk_ids,
)
from cite_rag.results import save_result, saved_results
from cite_rag.retrieval import Retriever
from cite_rag.retrieval_eval import (
    evaluate_retrieval,
    first_relevant_rank,
    summarize_ranks,
)
from tests.fakes import FakeEmbedder

CHUNKS = [
    Chunk("pep-0008:0000", 8, "Style", "Layout", "Limit all lines to 79\n  characters."),
    Chunk("pep-0572:0000", 572, "Walrus", "Syntax", "Limit all lines to 79 characters."),
    Chunk("pep-0572:0001", 572, "Walrus", "Syntax", "The walrus operator assigns."),
]


def question(pep: int, text: str, question_text: str = "Q?") -> Question:
    return Question(
        id="q",
        question=question_text,
        answerable=True,
        evidence=[Evidence(pep=pep, text=text)],
    )


def test_relevance_ignores_whitespace_and_respects_the_pep() -> None:
    labelled = question(8, "Limit all lines to 79 characters.")
    assert relevant_chunk_ids(labelled, CHUNKS) == {"pep-0008:0000"}


def test_check_evidence_rejects_a_label_found_in_no_chunk() -> None:
    with pytest.raises(MissingEvidenceError, match="q"):
        check_evidence([question(8, "text that is not there")], CHUNKS)


def test_first_relevant_rank_is_one_based() -> None:
    assert first_relevant_rank(["a", "b", "c"], {"b", "c"}) == 2
    assert first_relevant_rank(["a"], {"b"}) is None


def test_summarize_ranks_computes_hit_rate_and_mrr() -> None:
    metrics = summarize_ranks([1, 2, 4, None])
    assert metrics == {
        "hit_rate_at_1": 0.25,
        "hit_rate_at_3": 0.5,
        "hit_rate_at_5": 0.75,
        "mrr": round((1 + 1 / 2 + 1 / 4) / 4, 4),
    }


def test_evaluate_retrieval_scores_every_mode_and_skips_unanswerable() -> None:
    embedder = FakeEmbedder()
    retriever = Retriever(build_index(CHUNKS, embedder), embedder)
    questions = [
        question(572, "The walrus operator", "what does the walrus operator do"),
        Question(id="u", question="what is the walrus", answerable=False),
    ]
    report = evaluate_retrieval(retriever, questions, CHUNKS)
    assert report["questions"] == 1
    assert set(report["modes"]) == {"bm25", "dense", "hybrid"}
    assert report["modes"]["bm25"]["ranks"] == {"q": 1}


def test_results_round_trip_and_latest_wins(tmp_path: Path) -> None:
    older = {"date": "2026-01-01", "git_commit": "aaa", "value": 1}
    newer = {"date": "2026-02-01", "git_commit": "bbb", "value": 2}
    save_result("retrieval", newer, tmp_path)
    path = save_result("retrieval", older, tmp_path)
    assert path.name == "retrieval-2026-01-01-aaa.json"
    assert saved_results("retrieval", tmp_path)[-1] == newer


def test_saved_results_reports_a_missing_kind(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="answers"):
        saved_results("answers", tmp_path)


def test_every_answerable_question_has_evidence_in_the_real_corpus() -> None:
    check_evidence(load_questions(), chunk_corpus(load_corpus()))


def test_question_set_is_well_formed() -> None:
    questions = load_questions()
    assert len({q.id for q in questions}) == len(questions)
    assert sum(not q.answerable for q in questions) >= 20
    for q in questions:
        if q.answerable:
            assert q.evidence
            assert q.must_mention
        else:
            assert q.kind in {"off_topic", "near_miss", "false_premise", "injection"}
            assert not q.evidence
