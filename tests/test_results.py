import subprocess
from pathlib import Path

import pytest

from cite_rag.chunking import Chunk
from cite_rag.results import DEFAULT_RESULTS_DIR, NO_COMMIT, run_metadata

CHUNKS = [Chunk("pep-0008:0000", 8, "Style", "Layout", "Text.")]


def git(*args: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.test"]
    subprocess.run(["git", *identity, *args], check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "questions.jsonl").write_text("{}", encoding="utf-8")
    git("init")
    git("add", ".")
    git("commit", "-m", "start")
    return tmp_path


def metadata() -> dict[str, object]:
    return run_metadata(CHUNKS, Path("questions.jsonl"), "embedder")


def test_saved_results_and_docs_do_not_mark_the_code_as_changed(repository: Path) -> None:
    results = repository / DEFAULT_RESULTS_DIR
    results.mkdir(parents=True)
    (results / "retrieval.json").write_text("{}", encoding="utf-8")
    (repository / "README.md").write_text("Results.\n", encoding="utf-8")
    assert metadata()["git_dirty"] is False


def test_changed_code_is_recorded_as_dirty(repository: Path) -> None:
    (repository / "src").mkdir()
    (repository / "src" / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert metadata()["git_dirty"] is True


def test_a_directory_without_commits_is_reported_as_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "questions.jsonl").write_text("{}", encoding="utf-8")
    git("init")
    assert metadata()["git_commit"] == NO_COMMIT
