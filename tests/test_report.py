import re
from pathlib import Path
from typing import Any

import pytest

from cite_rag.report import (
    END_MARKER,
    README_PATH,
    START_MARKER,
    render_report,
    render_saved_report,
    replace_results_section,
)
from cite_rag.results import save_result


def metrics(value: float) -> dict[str, float]:
    return {
        "hit_rate_at_1": value,
        "hit_rate_at_3": value,
        "hit_rate_at_5": value,
        "mrr": value,
    }


RETRIEVAL: dict[str, Any] = {
    "date": "2026-01-02T030405Z",
    "git_commit": "abc1234",
    "retrieval": {
        "questions": 67,
        "modes": {
            "bm25": {"metrics": metrics(0.5)},
            "dense": {"metrics": metrics(0.6)},
            "hybrid": {"metrics": metrics(0.7564)},
        },
    },
}
ANSWERS: dict[str, Any] = {
    "date": "2026-01-02T030500Z",
    "git_commit": "abc1234",
    "git_dirty": False,
    "model": "gpt-test",
    "top_k": 5,
    "corpus": {"documents": 41, "chunks": 1665},
    "answers": {
        "metrics": {
            "answerable_questions": 67,
            "unanswerable_questions": 20,
            "answer_rate": 0.8,
            "answer_correctness": 0.75,
            "evidence_cited_rate": 0.9,
            "refusal_recall": 0.95,
            "refusal_precision": 0.7,
            "citation_validity": 0.99,
            "citations_checked": 120,
        },
        "by_kind": {
            "off_topic": {"total": 5, "refused": 5},
            "near_miss": {"total": 9, "refused": 8},
            "false_premise": {"total": 4, "refused": 4},
            "injection": {"total": 2, "refused": 2},
        },
        "injection": {"cases": 6, "resisted": 5},
    },
}


def test_report_carries_the_saved_values() -> None:
    report = render_report(RETRIEVAL, [ANSWERS])
    assert "| Hybrid | 75.6% | 75.6% | 75.6% | 0.756 |" in report
    assert "| Refused, of unanswerable questions | 95.0% |" in report
    assert "| Model citations that passed verification | 99.0% of 120 |" in report
    assert "| Hostile passages resisted | 5 of 6 |" in report
    assert "| Near miss | 8 of 9 |" in report
    assert "`gpt-test`" in report
    assert "retrieval-2026-01-02T030405Z-abc1234.json" in report


def test_report_details_the_latest_run_and_lists_every_run() -> None:
    older = ANSWERS | {
        "date": "2026-01-01T000000Z",
        "git_commit": "old1234",
        "git_dirty": True,
    }
    report = render_report(RETRIEVAL, [older, ANSWERS])
    assert "| 2026-01-01T000000Z | `old1234` (dirty) |" in report
    assert "| 2026-01-02T030500Z | `abc1234` |" in report
    assert "answers-2026-01-02T030500Z-abc1234.json" in report
    assert "answers-2026-01-01T000000Z-old1234.json" not in report


def test_a_later_free_retrieval_run_does_not_change_the_report(tmp_path: Path) -> None:
    save_result("retrieval", RETRIEVAL, tmp_path)
    save_result("answers", ANSWERS, tmp_path)
    expected = render_saved_report(tmp_path)
    save_result("retrieval", RETRIEVAL | {"date": "2026-01-03T000000Z"}, tmp_path)
    assert render_saved_report(tmp_path) == expected


def test_replace_results_section_touches_only_the_marked_block() -> None:
    readme = f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n"
    updated = replace_results_section(readme, "new")
    assert updated == f"before\n{START_MARKER}\nnew\n{END_MARKER}\nafter\n"


def test_replace_results_section_requires_the_markers() -> None:
    with pytest.raises(ValueError, match="markers"):
        replace_results_section("no markers here", "new")


def test_readme_results_match_the_latest_saved_files() -> None:
    expected = render_saved_report()
    readme = README_PATH.read_text(encoding="utf-8")
    assert f"{START_MARKER}\n{expected}\n{END_MARKER}" in readme


def test_readme_cites_saved_files_that_exist() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    cited = re.findall(r"evals/results/[\w.-]+\.json", readme)
    assert cited
    assert all(Path(path).exists() for path in cited)
