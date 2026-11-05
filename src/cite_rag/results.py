"""Save evaluation results with enough context to reproduce and audit them."""

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cite_rag.chunking import Chunk
from cite_rag.index import fingerprint

DEFAULT_RESULTS_DIR = Path("evals/results")
NO_COMMIT = "uncommitted"
# Questions, corpus, and prompt have their own hashes, so only the code is checked here.
_CODE_PATHS = ("src", "pyproject.toml", "uv.lock")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_metadata(
    chunks: Sequence[Chunk], questions_path: Path, embedding_model: str
) -> dict[str, Any]:
    """Describe the run. The hashes expose any later edit to the questions or corpus."""
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "embedding_model": embedding_model,
        "questions_sha256": sha256_hex(questions_path.read_bytes()),
        "corpus_fingerprint": fingerprint(chunks),
        "corpus": {
            "documents": len({chunk.pep for chunk in chunks}),
            "chunks": len(chunks),
        },
    }


def result_file_name(kind: str, result: dict[str, Any]) -> str:
    return f"{kind}-{result['date']}-{result['git_commit']}.json"


def save_result(
    kind: str, result: dict[str, Any], directory: Path = DEFAULT_RESULTS_DIR
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / result_file_name(kind, result)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def saved_results(
    kind: str, directory: Path = DEFAULT_RESULTS_DIR
) -> list[dict[str, Any]]:
    """Return every result of a kind, oldest first. File names sort by date."""
    paths = sorted(directory.glob(f"{kind}-*.json"))
    if not paths:
        raise FileNotFoundError(f"no {kind} result in {directory}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _git_commit() -> str:
    completed = _git("rev-parse", "--short", "HEAD")
    return completed.stdout.strip() if completed.returncode == 0 else NO_COMMIT


def _git_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", *_CODE_PATHS).stdout.strip())
