"""The hand-written evaluation questions and their ground truth.

Relevance is defined by literal evidence text rather than chunk IDs, so the labels stay
valid when chunking changes and can never be fitted to one retrieval run.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from cite_rag.chunking import Chunk
from cite_rag.corpus import normalize_text

DEFAULT_QUESTIONS_PATH = Path("evals/questions.jsonl")


class MissingEvidenceError(Exception):
    """A question's evidence does not appear in any chunk, so its label is wrong."""


class Evidence(BaseModel):
    pep: int
    text: str


class Question(BaseModel):
    id: str
    question: str
    answerable: bool
    evidence: list[Evidence] = []
    # Every group must be matched by at least one of its alternatives.
    must_mention: list[list[str]] = []
    kind: str | None = None


def load_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[Question]:
    return load_jsonl(path, Question)


def relevant_chunk_ids(question: Question, chunks: Iterable[Chunk]) -> set[str]:
    return {
        chunk.id
        for chunk in chunks
        for evidence in question.evidence
        if chunk.pep == evidence.pep
        and normalize_text(evidence.text) in normalize_text(chunk.text)
    }


def check_evidence(questions: Iterable[Question], chunks: Iterable[Chunk]) -> None:
    """Fail loudly on a bad label instead of letting it count as a retrieval miss."""
    chunk_list = list(chunks)
    missing = [
        question.id
        for question in questions
        if question.answerable and not relevant_chunk_ids(question, chunk_list)
    ]
    if missing:
        raise MissingEvidenceError(f"evidence found in no chunk: {', '.join(missing)}")
