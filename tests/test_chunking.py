from cite_rag.chunking import MAX_CHUNK_CHARS, chunk_corpus, chunk_document
from cite_rag.corpus import Document, load_corpus

PEP_TEXT = """\
PEP: 7
Title: Example
Author: Ada Lovelace,
        Grace Hopper
Status: Final
Post-History: 01-Jan-2020

Abstract
========

The abstract.

Rationale
=========

Why it exists.

A subsection
------------

Details here.

Second subsection
-----------------

More details.

Back at the top
===============

Final words.

::

    code = "indented"
    ------------

References
==========

.. [1] https://example.test

Copyright
=========

Placed in the public domain.
"""


def document(text: str = PEP_TEXT) -> Document:
    return Document(7, "Example", text)


def test_ids_are_unique_ordinals_within_a_pep() -> None:
    chunks = chunk_document(document())
    assert [chunk.id for chunk in chunks] == [
        f"pep-0007:{ordinal:04d}" for ordinal in range(len(chunks))
    ]


def test_first_chunk_is_retrievable_metadata_without_noise() -> None:
    first = chunk_document(document())[0]
    assert first.section == "Metadata"
    assert "Author: Ada Lovelace, Grace Hopper" in first.text
    assert "Post-History" not in first.text


def test_section_paths_follow_heading_levels() -> None:
    sections = [chunk.section for chunk in chunk_document(document())]
    assert sections == [
        "Metadata",
        "Abstract",
        "Rationale",
        "Rationale > A subsection",
        "Rationale > Second subsection",
        "Back at the top",
    ]


def test_skipped_sections_are_left_out() -> None:
    joined = "\n".join(chunk.text for chunk in chunk_document(document()))
    assert "public domain" not in joined
    assert "example.test" not in joined


def test_overlined_headings_are_recognized() -> None:
    text = (
        "PEP: 7\nTitle: Example\n\n"
        "==========\n Top level\n==========\n\nBody.\n\n"
        "Nested\n======\n\nInner.\n"
    )
    assert [c.section for c in chunk_document(document(text))] == [
        "Metadata",
        "Top level",
        "Top level > Nested",
    ]


def test_indented_adornment_in_a_code_block_is_not_a_heading() -> None:
    chunks = chunk_document(document())
    assert '    code = "indented"\n    ------------' in chunks[-1].text


def test_long_sections_are_split_within_the_limit_without_losing_text() -> None:
    paragraphs = [f"Paragraph {n} " + "word " * 80 for n in range(20)]
    body = "\n\n".join(paragraphs)
    text = f"PEP: 7\nTitle: Example\n\nLong\n====\n\n{body}\n"
    chunks = chunk_document(document(text))[1:]
    assert len(chunks) > 1
    assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)
    joined = "\n\n".join(chunk.text for chunk in chunks)
    assert joined.split() == body.split()


def test_oversized_paragraph_and_oversized_line_are_split() -> None:
    long_line = "x" * (MAX_CHUNK_CHARS * 2 + 10)
    block = "\n".join(["short line"] * 200)
    text = f"PEP: 7\nTitle: Example\n\nLong\n====\n\n{block}\n\n{long_line}\n"
    chunks = chunk_document(document(text))[1:]
    assert all(0 < len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).count("x") == len(long_line)


def test_search_text_carries_the_pep_and_section_context() -> None:
    chunk = chunk_document(document())[3]
    assert chunk.search_text.startswith("PEP 7: Example\nRationale > A subsection\n\n")


def test_real_corpus_produces_unique_ids_within_the_limit() -> None:
    chunks = chunk_corpus(load_corpus())
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(0 < len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)
