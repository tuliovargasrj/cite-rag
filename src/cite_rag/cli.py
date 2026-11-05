"""Command line interface. Only `ask` and `run-evals` call the paid model."""

import os
from pathlib import Path
from typing import Any, NoReturn

import typer
from dotenv import load_dotenv

from cite_rag.answer_eval import (
    DEFAULT_INJECTIONS_PATH,
    evaluate_answers,
    load_injections,
)
from cite_rag.answering import generate_answer, render_answer
from cite_rag.chunking import Chunk, chunk_corpus
from cite_rag.corpus import DEFAULT_CORPUS_DIR, load_corpus, normalize_text
from cite_rag.embeddings import DEFAULT_MODEL as EMBEDDING_MODEL
from cite_rag.embeddings import FastEmbedder
from cite_rag.index import (
    DEFAULT_INDEX_DIR,
    StaleIndexError,
    build_index,
    load_index,
    save_index,
)
from cite_rag.llm import DEFAULT_MODEL, OpenAIClient
from cite_rag.prompts import SYSTEM_PROMPT
from cite_rag.questions import (
    DEFAULT_QUESTIONS_PATH,
    Question,
    check_evidence,
    load_questions,
)
from cite_rag.report import (
    README_PATH,
    render_saved_report,
    replace_results_section,
)
from cite_rag.results import run_metadata, save_result, sha256_hex
from cite_rag.retrieval import DEFAULT_TOP_K, Retriever, SearchMode
from cite_rag.retrieval_eval import evaluate_retrieval

app = typer.Typer(no_args_is_help=True, add_completion=False)

_SNIPPET_CHARS = 160


def _fail(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _load_chunks(corpus_dir: Path) -> list[Chunk]:
    return chunk_corpus(load_corpus(corpus_dir))


def _load_retriever(chunks: list[Chunk], index_dir: Path) -> Retriever:
    embedder = FastEmbedder()
    try:
        index = load_index(chunks, embedder.model_name, index_dir)
    except StaleIndexError as error:
        _fail(str(error))
    return Retriever(index, embedder)


def _load_llm(model: str) -> OpenAIClient:
    load_dotenv(".env")
    if not os.environ.get("OPENAI_API_KEY"):
        _fail("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAIClient(model)


def _load_checked_questions(path: Path, chunks: list[Chunk]) -> list[Question]:
    questions = load_questions(path)
    check_evidence(questions, chunks)
    return questions


def _context(retriever: Retriever, question: str, top_k: int) -> list[Chunk]:
    return [hit.chunk for hit in retriever.search(question, top_k)]


def _run_retrieval_eval(
    retriever: Retriever,
    questions: list[Question],
    chunks: list[Chunk],
    metadata: dict[str, Any],
) -> None:
    report = evaluate_retrieval(retriever, questions, chunks)
    path = save_result("retrieval", metadata | {"retrieval": report})
    for mode, scores in report["modes"].items():
        typer.echo(f"{mode:>6}  {scores['metrics']}")
    typer.echo(f"Saved {path}")


@app.command()
def index(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> None:
    """Chunk the corpus and embed it locally. No API key needed."""
    chunks = _load_chunks(corpus_dir)
    embedder = FastEmbedder()
    save_index(build_index(chunks, embedder), embedder.model_name, index_dir)
    typer.echo(f"Indexed {len(chunks)} chunks into {index_dir}")


@app.command()
def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    mode: SearchMode = SearchMode.HYBRID,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> None:
    """Show the chunks retrieved for a query. No API key needed."""
    retriever = _load_retriever(_load_chunks(corpus_dir), index_dir)
    for hit in retriever.search(query, top_k, mode):
        snippet = normalize_text(hit.chunk.text)[:_SNIPPET_CHARS]
        typer.echo(f"{hit.score:.4f}  {hit.chunk.id}  {hit.chunk.section}")
        typer.echo(f"        {snippet}")


@app.command()
def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> None:
    """Answer from the documents with verified citations. Makes one paid call."""
    llm = _load_llm(model)
    retriever = _load_retriever(_load_chunks(corpus_dir), index_dir)
    context = _context(retriever, question, top_k)
    typer.echo(render_answer(generate_answer(question, context, llm)))


@app.command("eval-retrieval")
def eval_retrieval(
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> None:
    """Compare BM25, embeddings, and hybrid search. No API key needed."""
    chunks = _load_chunks(corpus_dir)
    retriever = _load_retriever(chunks, index_dir)
    questions = _load_checked_questions(questions_path, chunks)
    metadata = run_metadata(chunks, questions_path, EMBEDDING_MODEL)
    _run_retrieval_eval(retriever, questions, chunks, metadata)


@app.command("run-evals")
def run_evals(
    model: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    injections_path: Path = DEFAULT_INJECTIONS_PATH,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> None:
    """Run the retrieval and answer evaluations. Makes paid calls."""
    llm = _load_llm(model)
    chunks = _load_chunks(corpus_dir)
    retriever = _load_retriever(chunks, index_dir)
    # A bad label must stop the run before any paid call.
    questions = _load_checked_questions(questions_path, chunks)
    metadata = run_metadata(chunks, questions_path, EMBEDDING_MODEL)
    _run_retrieval_eval(retriever, questions, chunks, metadata)

    report = evaluate_answers(
        questions,
        load_injections(injections_path),
        lambda question: _context(retriever, question, top_k),
        chunks,
        llm,
    )
    result = metadata | {
        "model": model,
        "top_k": top_k,
        "prompt_sha256": sha256_hex(SYSTEM_PROMPT.encode()),
        "injections_sha256": sha256_hex(injections_path.read_bytes()),
        "answers": report,
    }
    path = save_result("answers", result)
    typer.echo(f"{report['metrics']}")
    typer.echo(f"Injection: {report['injection']}")
    typer.echo(f"Saved {path}")


@app.command()
def report() -> None:
    """Rewrite the README results section from the latest saved files."""
    try:
        section = render_saved_report()
    except FileNotFoundError as error:
        _fail(str(error))
    readme = README_PATH.read_text(encoding="utf-8")
    README_PATH.write_text(replace_results_section(readme, section), encoding="utf-8")
    typer.echo(f"Updated {README_PATH}")
