#!/usr/bin/env python3
"""Evaluate FTS, semantic embeddings, and hybrid retrieval."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from local_rag.database import add_database_argument, resolve_database  # noqa: E402
from local_rag.embeddings.build_embeddings import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
)
from local_rag.embeddings.search_embeddings import (  # noqa: E402
    search_embeddings,
)
from local_rag.retrieval.hybrid import (  # noqa: E402
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_FTS_WEIGHT,
    DEFAULT_RRF_K,
    HybridResult,
    fuse_results,
)
from local_rag.retrieval.search import search as search_fts  # noqa: E402


DEFAULT_QUERIES = Path("data/evaluation/test_queries.yaml")
DEFAULT_REPORT = Path("data/evaluation/retrieval_evaluation_hybrid.md")
CUTOFFS = (1, 3, 5, 10)


class RetrievalResult(Protocol):
    rank: int
    filename: str
    path: str
    page_number: int | None
    chunk_id: int
    document_id: int


@dataclass(frozen=True)
class TestQuery:
    query_id: str
    query: str
    known_relevant_documents: tuple[str, ...]
    category: str | None = None
    evaluation_mode: str | None = None
    relevance_confidence: str | None = None


@dataclass(frozen=True)
class QueryEvaluation:
    test_query: TestQuery
    fts_results: list[RetrievalResult]
    embedding_results: list[RetrievalResult]
    hybrid_results: list[RetrievalResult]
    fts_first_hit_rank: int | None
    embedding_first_hit_rank: int | None
    hybrid_first_hit_rank: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SQLite FTS5, embeddings, and hybrid retrieval with Hit@k metrics."
    )
    add_database_argument(parser)
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help=f"test queries YAML path (default: {DEFAULT_QUERIES})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=max(CUTOFFS),
        help=f"number of results to evaluate (default: {max(CUTOFFS)})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--prefer-reference",
        action="store_true",
        help="apply the FTS reference boost used by search_chunks.py",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="candidate pool size per method before hybrid fusion (default: 50)",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=DEFAULT_RRF_K,
        help=f"RRF smoothing constant for hybrid search (default: {DEFAULT_RRF_K:g})",
    )
    parser.add_argument(
        "--fts-weight",
        type=float,
        default=DEFAULT_FTS_WEIGHT,
        help=f"FTS contribution weight for hybrid search (default: {DEFAULT_FTS_WEIGHT:g})",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=DEFAULT_EMBEDDING_WEIGHT,
        help=f"embedding contribution weight for hybrid search (default: {DEFAULT_EMBEDDING_WEIGHT:g})",
    )
    parser.add_argument(
        "--no-rebuild-fts",
        action="store_true",
        help="use the existing chunks_fts table without rebuilding it first",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"write markdown report to this path (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="print the evaluation only; do not write a markdown report",
    )
    return parser.parse_args()


def unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_test_queries(path: Path) -> list[TestQuery]:
    """Load the small YAML subset used by data/evaluation/test_queries.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"Test queries file not found: {path}")

    items: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    active_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0 and stripped.startswith("- "):
            if current is not None:
                items.append(current)
            current = {}
            active_list_key = None
            payload = stripped[2:]
            if payload:
                key, _, value = payload.partition(":")
                current[key.strip()] = unquote_yaml_scalar(value)
            continue

        if current is None:
            raise ValueError(f"Unexpected YAML line before first item: {raw_line}")

        if indent == 2 and ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                current[key] = unquote_yaml_scalar(value)
                active_list_key = None
            else:
                current[key] = []
                active_list_key = key
            continue

        if indent >= 4 and stripped.startswith("- ") and active_list_key:
            values = current.setdefault(active_list_key, [])
            if not isinstance(values, list):
                raise ValueError(f"YAML key is not a list: {active_list_key}")
            values.append(unquote_yaml_scalar(stripped[2:]))
            continue

        raise ValueError(f"Unsupported YAML line: {raw_line}")

    if current is not None:
        items.append(current)

    queries: list[TestQuery] = []
    for index, item in enumerate(items, start=1):
        query = str(item.get("query", "")).strip()
        relevant = item.get("known_relevant_documents", [])
        if not query:
            raise ValueError(f"Query item #{index} has no query")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"Query item #{index} has no known_relevant_documents")
        queries.append(
            TestQuery(
                query_id=str(item.get("id") or f"q{index:03d}"),
                query=query,
                known_relevant_documents=tuple(str(value) for value in relevant),
                category=str(item["category"]) if item.get("category") else None,
                evaluation_mode=(
                    str(item["evaluation_mode"]) if item.get("evaluation_mode") else None
                ),
                relevance_confidence=(
                    str(item["relevance_confidence"])
                    if item.get("relevance_confidence")
                    else None
                ),
            )
        )
    return queries


def normalize_document_name(value: str) -> str:
    value = Path(value).name
    value = value.casefold().replace("\\", "/")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_relevant_result(
    result: RetrievalResult,
    known_relevant_documents: tuple[str, ...],
) -> bool:
    candidates = {
        normalize_document_name(result.filename),
        normalize_document_name(result.path),
    }
    for document in known_relevant_documents:
        normalized = normalize_document_name(document)
        if normalized in candidates:
            return True
    return False


def first_hit_rank(
    results: list[RetrievalResult],
    known_relevant_documents: tuple[str, ...],
) -> int | None:
    for result in results:
        if is_relevant_result(result, known_relevant_documents):
            return result.rank
    return None


def hit_at(rank: int | None, cutoff: int) -> int:
    return int(rank is not None and rank <= cutoff)


def format_hit_vector(rank: int | None) -> str:
    return " ".join(f"Hit@{cutoff}={hit_at(rank, cutoff)}" for cutoff in CUTOFFS)


def format_top_results(results: list[RetrievalResult], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for result in results[:limit]:
        page = result.page_number if result.page_number is not None else "-"
        lines.append(
            f"{result.rank}. {result.filename} "
            f"(page {page}, chunk {result.chunk_id}, document {result.document_id})"
        )
    return lines


def evaluate(
    *,
    database: Path,
    queries: list[TestQuery],
    top_k: int,
    model: str,
    ollama_url: str,
    rebuild_fts: bool,
    prefer_reference: bool,
    pool_size: int,
    rrf_k: float,
    fts_weight: float,
    embedding_weight: float,
) -> list[QueryEvaluation]:
    evaluations: list[QueryEvaluation] = []
    should_rebuild_fts = rebuild_fts

    for test_query in queries:
        fts_results, _, _ = search_fts(
            database=database,
            query=test_query.query,
            top_k=pool_size,
            rebuild_index=should_rebuild_fts,
            prefer_reference=prefer_reference,
        )
        should_rebuild_fts = False

        embedding_results = search_embeddings(
            database=database,
            query=test_query.query,
            top_k=pool_size,
            model=model,
            ollama_url=ollama_url,
        )
        hybrid_results: list[HybridResult] = fuse_results(
            fts_results=fts_results,
            embedding_results=embedding_results,
            top_k=top_k,
            rrf_k=rrf_k,
            fts_weight=fts_weight,
            embedding_weight=embedding_weight,
        )

        evaluations.append(
            QueryEvaluation(
                test_query=test_query,
                fts_results=fts_results[:top_k],
                embedding_results=embedding_results[:top_k],
                hybrid_results=hybrid_results,
                fts_first_hit_rank=first_hit_rank(
                    fts_results[:top_k], test_query.known_relevant_documents
                ),
                embedding_first_hit_rank=first_hit_rank(
                    embedding_results[:top_k], test_query.known_relevant_documents
                ),
                hybrid_first_hit_rank=first_hit_rank(
                    hybrid_results, test_query.known_relevant_documents
                ),
            )
        )

    return evaluations


def mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def method_summary(
    evaluations: list[QueryEvaluation],
    *,
    method: str,
) -> dict[int, float]:
    ranks = [
        {
            "fts": evaluation.fts_first_hit_rank,
            "embeddings": evaluation.embedding_first_hit_rank,
            "hybrid": evaluation.hybrid_first_hit_rank,
        }[method]
        for evaluation in evaluations
    ]
    return {
        cutoff: mean([hit_at(rank, cutoff) for rank in ranks])
        for cutoff in CUTOFFS
    }


def winner(evaluation: QueryEvaluation) -> str:
    ranks = {
        "fts": evaluation.fts_first_hit_rank,
        "embeddings": evaluation.embedding_first_hit_rank,
        "hybrid": evaluation.hybrid_first_hit_rank,
    }
    hit_ranks = {method: rank for method, rank in ranks.items() if rank is not None}
    if not hit_ranks:
        return "tie"
    best_rank = min(hit_ranks.values())
    winners = [
        method for method, rank in hit_ranks.items() if rank == best_rank
    ]
    if len(winners) != 1:
        return "tie"
    return winners[0]


def build_report(
    evaluations: list[QueryEvaluation],
    *,
    database: Path,
    queries_path: Path,
    top_k: int,
    model: str,
    prefer_reference: bool,
    pool_size: int,
    rrf_k: float,
    fts_weight: float,
    embedding_weight: float,
) -> str:
    fts_summary = method_summary(evaluations, method="fts")
    embedding_summary = method_summary(evaluations, method="embeddings")
    hybrid_summary = method_summary(evaluations, method="hybrid")
    winner_counts = {
        name: sum(1 for evaluation in evaluations if winner(evaluation) == name)
        for name in ("fts", "embeddings", "hybrid", "tie")
    }

    lines = [
        "# Retrieval Evaluation",
        "",
        "## Configuration",
        "",
        f"- Database: `{database}`",
        f"- Queries: `{queries_path}`",
        f"- Top-k: {top_k}",
        f"- Candidate pool size: {pool_size}",
        f"- Embedding model: `{model}`",
        f"- FTS prefer reference boost: `{str(prefer_reference).lower()}`",
        f"- Hybrid fusion: `RRF k={rrf_k:g}, fts_weight={fts_weight:g}, embedding_weight={embedding_weight:g}`",
        "",
        "## Evaluation Summary",
        "",
        f"Queries: {len(evaluations)}",
        "",
        "FTS",
        "-----",
    ]
    lines.extend(
        f"Hit@{cutoff}: {fts_summary[cutoff]:.2f}" for cutoff in CUTOFFS
    )
    lines.extend(["", "Embeddings", "----------"])
    lines.extend(
        f"Hit@{cutoff}: {embedding_summary[cutoff]:.2f}" for cutoff in CUTOFFS
    )
    lines.extend(["", "Hybrid", "------"])
    lines.extend(
        f"Hit@{cutoff}: {hybrid_summary[cutoff]:.2f}" for cutoff in CUTOFFS
    )
    lines.extend(
        [
            "",
            "Winner by first relevant rank",
            "-----------------------------",
            f"FTS: {winner_counts['fts']}",
            f"Embeddings: {winner_counts['embeddings']}",
            f"Hybrid: {winner_counts['hybrid']}",
            f"Tie: {winner_counts['tie']}",
            "",
            "## Per-query Results",
            "",
        ]
    )

    for evaluation in evaluations:
        test_query = evaluation.test_query
        lines.extend(
            [
                f"### {test_query.query_id}",
                "",
                f"Query: {test_query.query}",
                "",
                "Known relevant documents:",
            ]
        )
        lines.extend(
            f"- {document}" for document in test_query.known_relevant_documents
        )
        lines.extend(
            [
                "",
                (
                    f"FTS first hit: "
                    f"{evaluation.fts_first_hit_rank or 'none'} "
                    f"({format_hit_vector(evaluation.fts_first_hit_rank)})"
                ),
                (
                    f"Embeddings first hit: "
                    f"{evaluation.embedding_first_hit_rank or 'none'} "
                    f"({format_hit_vector(evaluation.embedding_first_hit_rank)})"
                ),
                (
                    f"Hybrid first hit: "
                    f"{evaluation.hybrid_first_hit_rank or 'none'} "
                    f"({format_hit_vector(evaluation.hybrid_first_hit_rank)})"
                ),
                f"Winner: {winner(evaluation)}",
                "",
                "FTS top results:",
            ]
        )
        lines.extend(f"- {line}" for line in format_top_results(evaluation.fts_results))
        lines.append("")
        lines.append("Embeddings top results:")
        lines.extend(
            f"- {line}" for line in format_top_results(evaluation.embedding_results)
        )
        lines.append("")
        lines.append("Hybrid top results:")
        lines.extend(
            f"- {line}" for line in format_top_results(evaluation.hybrid_results)
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def print_console_summary(report: str) -> None:
    sections = report.split("## Per-query Results", maxsplit=1)
    print(sections[0].replace("# Retrieval Evaluation\n\n", "").rstrip())


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)
    queries_path = args.queries.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    if args.top_k < max(CUTOFFS):
        print(f"--top-k must be at least {max(CUTOFFS)} for Hit@10 evaluation.")
        return 2
    if args.pool_size < args.top_k:
        print("--pool-size must be greater than or equal to --top-k.")
        return 2

    try:
        queries = load_test_queries(queries_path)
        evaluations = evaluate(
            database=database,
            queries=queries,
            top_k=args.top_k,
            model=args.model,
            ollama_url=args.ollama_url,
            rebuild_fts=not args.no_rebuild_fts,
            prefer_reference=args.prefer_reference,
            pool_size=args.pool_size,
            rrf_k=args.rrf_k,
            fts_weight=args.fts_weight,
            embedding_weight=args.embedding_weight,
        )
    except (FileNotFoundError, RuntimeError, sqlite3.Error, ValueError) as error:
        print(error)
        return 1

    report = build_report(
        evaluations,
        database=database,
        queries_path=queries_path,
        top_k=args.top_k,
        model=args.model,
        prefer_reference=args.prefer_reference,
        pool_size=args.pool_size,
        rrf_k=args.rrf_k,
        fts_weight=args.fts_weight,
        embedding_weight=args.embedding_weight,
    )
    print_console_summary(report)

    if not args.no_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print()
        print(f"Report written: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
