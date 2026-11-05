import json
from pathlib import Path

import pytest

from cite_rag.corpus import (
    CorpusError,
    Document,
    detect_license,
    load_corpus,
    parse_headers,
)
from tests.fakes import write_corpus

PUBLIC_DOMAIN = "Copyright\n=========\n\nThis document is in the public domain.\n"


def test_parse_headers_joins_continuation_lines() -> None:
    text = "PEP: 1\nAuthor: A,\n        B\nStatus: Final\n\nBody: not a header\n"
    assert parse_headers(text) == {"PEP": "1", "Author": "A, B", "Status": "Final"}


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("This document has been placed in the public domain.", "Public domain"),
        ("Licensed under CC0-1.0.", "CC0-1.0"),
        (
            "Placed in the public domain or under the\nCC0-1.0-Universal license.",
            "Public domain or CC0-1.0",
        ),
        ("This document is under the Open Publication License.", None),
    ],
)
def test_detect_license(statement: str, expected: str | None) -> None:
    assert detect_license(f"Copyright\n=========\n\n{statement}\n") == expected


def test_detect_license_handles_overlined_heading() -> None:
    text = "===========\n Copyright\n===========\n\nPlaced in the public domain.\n"
    assert detect_license(text) == "Public domain"


def test_detect_license_ignores_the_phrase_outside_the_copyright_section() -> None:
    assert detect_license("Abstract\n========\n\nThe public domain is nice.\n") is None


def test_load_corpus_returns_documents(tmp_path: Path) -> None:
    write_corpus(tmp_path, PUBLIC_DOMAIN)
    documents = load_corpus(tmp_path)
    assert documents == [Document(pep=1, title="Example", text=PUBLIC_DOMAIN)]


def test_load_corpus_rejects_a_modified_file(tmp_path: Path) -> None:
    write_corpus(tmp_path, PUBLIC_DOMAIN)
    (tmp_path / "pep-0001.rst").write_text("tampered", encoding="utf-8")
    with pytest.raises(CorpusError, match="sha256"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_a_file_without_an_allowed_license(tmp_path: Path) -> None:
    write_corpus(tmp_path, "Copyright\n=========\n\nAll rights reserved.\n")
    with pytest.raises(CorpusError, match="license"):
        load_corpus(tmp_path)


def test_committed_corpus_records_a_source_for_every_file() -> None:
    corpus_dir = Path("data/corpus")
    documents = load_corpus(corpus_dir)
    assert len(documents) == len(list(corpus_dir.glob("*.rst")))
    sources = json.loads((corpus_dir / "sources.json").read_text(encoding="utf-8"))
    assert all(source["source_url"].startswith("https://") for source in sources)
