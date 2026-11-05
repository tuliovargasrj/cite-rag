"""Generate an answer and accept it only if code can verify every citation.

Everything here fails closed: output that cannot be verified becomes a refusal, so the
model can never present an unsupported claim as a cited answer.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import ValidationError

from cite_rag.chunking import Chunk
from cite_rag.corpus import normalize_text
from cite_rag.llm import LLMClient
from cite_rag.prompts import SYSTEM_PROMPT, ModelAnswer, ModelCitation, build_user_message

REFUSAL_MESSAGE = "The documents do not contain an answer to this question."
# Rejects quotes so short that they would appear in a chunk by chance.
_MIN_QUOTE_CHARS = 10
_CLOSING_PUNCTUATION = ".,;:!?…"
# Backticks delimit RST literals, and colons mark RST literal blocks.
_RST_MARKUP = str.maketrans("", "", "`:")


class RefusalReason(StrEnum):
    NO_CONTEXT = "no_context"
    MODEL_DECLINED = "model_declined"
    MALFORMED_OUTPUT = "malformed_output"
    NO_CLAIMS = "no_claims"
    UNCITED_CLAIM = "uncited_claim"
    INVALID_CITATION = "invalid_citation"


@dataclass(frozen=True)
class Citation:
    chunk: Chunk
    quote: str


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Answer:
    question: str
    claims: tuple[Claim, ...] = ()
    refusal: RefusalReason | None = None
    # Counted even on refusal, because they measure the model and not the final output.
    citations_checked: int = 0
    citations_invalid: int = 0
    raw_output: str = field(default="", repr=False)

    @property
    def answered(self) -> bool:
        return self.refusal is None

    @property
    def text(self) -> str:
        if self.refusal is not None:
            return REFUSAL_MESSAGE
        return " ".join(claim.text for claim in self.claims)


def generate_answer(question: str, chunks: Sequence[Chunk], llm: LLMClient) -> Answer:
    if not question.strip() or not chunks:
        return Answer(question, refusal=RefusalReason.NO_CONTEXT)
    raw = llm.complete(SYSTEM_PROMPT, build_user_message(question, chunks))
    return validate_output(question, raw, chunks)


def validate_output(question: str, raw: str, chunks: Sequence[Chunk]) -> Answer:
    """Turn raw model text into an Answer, or into a refusal if anything fails."""
    try:
        parsed = ModelAnswer.model_validate_json(raw)
    except ValidationError:
        return Answer(question, refusal=RefusalReason.MALFORMED_OUTPUT, raw_output=raw)
    if not parsed.answerable:
        return Answer(question, refusal=RefusalReason.MODEL_DECLINED, raw_output=raw)
    if not parsed.claims:
        return Answer(question, refusal=RefusalReason.NO_CLAIMS, raw_output=raw)

    # Only the retrieved chunks count, so a real ID that was never shown is rejected too.
    context = {chunk.id: chunk for chunk in chunks}
    claims = []
    checked = invalid = 0
    for claim in parsed.claims:
        citations = []
        for citation in claim.citations:
            checked += 1
            verified = _verify(citation, context)
            if verified is None:
                invalid += 1
            else:
                citations.append(verified)
        claims.append(Claim(claim.text, tuple(citations)))

    refusal = None
    if invalid:
        refusal = RefusalReason.INVALID_CITATION
    elif not all(claim.citations for claim in claims):
        refusal = RefusalReason.UNCITED_CLAIM
    return Answer(
        question,
        claims=tuple(claims) if refusal is None else (),
        refusal=refusal,
        citations_checked=checked,
        citations_invalid=invalid,
        raw_output=raw,
    )


def _verify(citation: ModelCitation, context: dict[str, Chunk]) -> Citation | None:
    chunk = context.get(citation.chunk_id)
    if chunk is None:
        return None
    # A quote cut mid-sentence often gains a closing period that the source lacks.
    quote = _comparable(citation.quote).rstrip(_CLOSING_PUNCTUATION)
    if len(quote) < _MIN_QUOTE_CHARS or quote not in _comparable(chunk.text):
        return None
    return Citation(chunk, citation.quote)


def _comparable(text: str) -> str:
    """Ignore what a faithful quote may lose: line wrapping, case, and RST markup."""
    return normalize_text(text.translate(_RST_MARKUP)).casefold()


def render_answer(answer: Answer) -> str:
    """Format an answer for the terminal, numbering each distinct cited chunk."""
    if answer.refusal is not None:
        return f"{REFUSAL_MESSAGE} ({answer.refusal.value})"
    # Each cited chunk maps to its distinct quotes. Dicts keep first-seen order.
    cited: dict[Chunk, dict[str, None]] = {}
    for claim in answer.claims:
        for citation in claim.citations:
            cited.setdefault(citation.chunk, {})[normalize_text(citation.quote)] = None
    numbers = {chunk.id: number for number, chunk in enumerate(cited, start=1)}

    sentences = [
        claim.text + " " + "".join(f"[{numbers[c.chunk.id]}]" for c in claim.citations)
        for claim in answer.claims
    ]
    lines = [" ".join(sentences), ""]
    for chunk, quotes in cited.items():
        lines.append(
            f"[{numbers[chunk.id]}] {chunk.id}  PEP {chunk.pep}, {chunk.section}"
        )
        lines += [f'    "{quote}"' for quote in quotes]
    return "\n".join(lines)
