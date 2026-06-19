"""Hybrid retrieval with Reciprocal Rank Fusion over FTS and embeddings."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from local_rag.database import add_database_argument, resolve_database
from local_rag.embeddings.build_embeddings import DEFAULT_MODEL, DEFAULT_OLLAMA_URL
from local_rag.embeddings.search_embeddings import SemanticResult, search_embeddings
from local_rag.retrieval.search import SearchResult, search as search_fts


DEFAULT_TOP_K = 10
DEFAULT_POOL_SIZE = 50
DEFAULT_RRF_K = 60.0
DEFAULT_FTS_WEIGHT = 1.5
DEFAULT_EMBEDDING_WEIGHT = 1.0


@dataclass(frozen=True)
class HybridResult:
    rank: int
    score: float
    chunk_id: int
    document_id: int
    filename: str
    path: str
    relative_path: str | None
    absolute_path: str | None
    scan_root: str | None
    page_number: int | None
    document_type: str
    content_category: str | None
    chunk_strategy: str
    text: str
    fts_rank: int | None
    embedding_rank: int | None
    fts_score: float | None
    embedding_similarity: float | None


@dataclass
class FusionCandidate:
    chunk_id: int
    result: SearchResult | SemanticResult
    score: float = 0.0
    fts_rank: int | None = None
    embedding_rank: int | None = None
    fts_score: float | None = None
    embedding_similarity: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search chunks with hybrid FTS + embeddings rank fusion."
    )
    add_database_argument(parser)
    parser.add_argument("query", help="hybrid search query")
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"number of results to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help=f"candidate pool size per method (default: {DEFAULT_POOL_SIZE})",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=DEFAULT_RRF_K,
        help=f"RRF smoothing constant (default: {DEFAULT_RRF_K:g})",
    )
    parser.add_argument(
        "--fts-weight",
        type=float,
        default=DEFAULT_FTS_WEIGHT,
        help=f"FTS contribution weight (default: {DEFAULT_FTS_WEIGHT:g})",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=DEFAULT_EMBEDDING_WEIGHT,
        help=f"embedding contribution weight (default: {DEFAULT_EMBEDDING_WEIGHT:g})",
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
        help="apply the FTS reference boost before fusion",
    )
    parser.add_argument(
        "--no-rebuild-fts",
        action="store_true",
        help="use existing chunks_fts table without rebuilding it first",
    )
    return parser.parse_args()


def compact_text(text: str, limit: int = 700) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def reciprocal_rank(rank: int, *, rrf_k: float) -> float:
    return 1.0 / (rrf_k + rank)


def fuse_results(
    *,
    fts_results: list[SearchResult],
    embedding_results: list[SemanticResult],
    top_k: int,
    rrf_k: float = DEFAULT_RRF_K,
    fts_weight: float = DEFAULT_FTS_WEIGHT,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
) -> list[HybridResult]:
    candidates: dict[int, FusionCandidate] = {}

    for result in fts_results:
        candidate = candidates.setdefault(
            result.chunk_id,
            FusionCandidate(chunk_id=result.chunk_id, result=result),
        )
        candidate.result = result
        candidate.fts_rank = result.rank
        candidate.fts_score = result.score
        candidate.score += fts_weight * reciprocal_rank(result.rank, rrf_k=rrf_k)

    for result in embedding_results:
        candidate = candidates.setdefault(
            result.chunk_id,
            FusionCandidate(chunk_id=result.chunk_id, result=result),
        )
        candidate.embedding_rank = result.rank
        candidate.embedding_similarity = result.similarity
        candidate.score += embedding_weight * reciprocal_rank(result.rank, rrf_k=rrf_k)

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.score,
            -(candidate.fts_rank or 10**9),
            -(candidate.embedding_rank or 10**9),
        ),
        reverse=True,
    )

    results: list[HybridResult] = []
    for rank, candidate in enumerate(sorted_candidates[:top_k], start=1):
        source = candidate.result
        results.append(
            HybridResult(
                rank=rank,
                score=candidate.score,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                filename=source.filename,
                path=source.path,
                relative_path=source.relative_path,
                absolute_path=source.absolute_path,
                scan_root=source.scan_root,
                page_number=source.page_number,
                document_type=source.document_type,
                content_category=source.content_category,
                chunk_strategy=source.chunk_strategy,
                text=source.text,
                fts_rank=candidate.fts_rank,
                embedding_rank=candidate.embedding_rank,
                fts_score=candidate.fts_score,
                embedding_similarity=candidate.embedding_similarity,
            )
        )
    return results


def search_hybrid(
    *,
    database: Path,
    query: str,
    top_k: int,
    pool_size: int,
    model: str,
    ollama_url: str,
    rebuild_fts: bool,
    prefer_reference: bool,
    rrf_k: float = DEFAULT_RRF_K,
    fts_weight: float = DEFAULT_FTS_WEIGHT,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
) -> tuple[list[HybridResult], list[SearchResult], list[SemanticResult]]:
    fts_results, _, _ = search_fts(
        database=database,
        query=query,
        top_k=pool_size,
        rebuild_index=rebuild_fts,
        prefer_reference=prefer_reference,
    )
    embedding_results = search_embeddings(
        database=database,
        query=query,
        top_k=pool_size,
        model=model,
        ollama_url=ollama_url,
    )
    hybrid_results = fuse_results(
        fts_results=fts_results,
        embedding_results=embedding_results,
        top_k=top_k,
        rrf_k=rrf_k,
        fts_weight=fts_weight,
        embedding_weight=embedding_weight,
    )
    return hybrid_results, fts_results, embedding_results


def print_results(
    *,
    query: str,
    results: list[HybridResult],
    model: str,
    rrf_k: float,
    fts_weight: float,
    embedding_weight: float,
) -> None:
    print(f"Query: {query}")
    print(f"Model: {model}")
    print(f"Fusion: RRF k={rrf_k:g}, fts_weight={fts_weight:g}, embedding_weight={embedding_weight:g}")

    if not results:
        print("\nNo results.")
        return

    for result in results:
        print()
        print(f"{result.rank}. hybrid_score: {result.score:.6f}")
        print(f"   file: {result.filename}")
        print(f"   path: {result.relative_path or result.filename}")
        print(f"   page: {result.page_number}")
        print(f"   type: {result.document_type}")
        print(f"   category: {result.content_category or 'unknown'}")
        print(f"   strategy: {result.chunk_strategy}")
        print(f"   fts_rank: {result.fts_rank or '-'}")
        print(f"   embedding_rank: {result.embedding_rank or '-'}")
        if result.fts_score is not None:
            print(f"   fts_score: {result.fts_score:.6f}")
        if result.embedding_similarity is not None:
            print(f"   embedding_similarity: {result.embedding_similarity:.6f}")
        print(f"   chunk_id: {result.chunk_id}")
        print(f"   document_id: {result.document_id}")
        print(f"   text: {compact_text(result.text)}")


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)
    try:
        results, _, _ = search_hybrid(
            database=database,
            query=args.query,
            top_k=args.top_k,
            pool_size=max(args.pool_size, args.top_k),
            model=args.model,
            ollama_url=args.ollama_url,
            rebuild_fts=not args.no_rebuild_fts,
            prefer_reference=args.prefer_reference,
            rrf_k=args.rrf_k,
            fts_weight=args.fts_weight,
            embedding_weight=args.embedding_weight,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(error)
        return 1

    print_results(
        query=args.query,
        results=results,
        model=args.model,
        rrf_k=args.rrf_k,
        fts_weight=args.fts_weight,
        embedding_weight=args.embedding_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
