"""Semantic search over chunk embeddings stored in SQLite."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from local_rag.database import (
    add_database_argument,
    connect_database,
    resolve_database,
)
from local_rag.embeddings.build_embeddings import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    blob_to_embedding,
    cosine_similarity,
    embed_text,
    initialize_database,
)
from local_rag.rag.source_formatter import format_source


DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class SemanticResult:
    rank: int
    similarity: float
    chunk_id: int
    document_id: int
    filename: str
    path: str
    relative_path: str | None
    absolute_path: str | None
    scan_root: str | None
    page_number: int | None
    chunk_index: int
    start_char: int | None
    end_char: int | None
    document_type: str
    content_category: str | None
    chunk_strategy: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search chunks by cosine similarity over local embeddings."
    )
    add_database_argument(parser)
    parser.add_argument("query", help="semantic search query")
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"number of results to return (default: {DEFAULT_TOP_K})",
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
    return parser.parse_args()


def compact_text(text: str, limit: int = 700) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def search_embeddings(
    database: Path,
    query: str,
    top_k: int,
    model: str,
    ollama_url: str,
) -> list[SemanticResult]:
    query_embedding = embed_text(query, model=model, ollama_url=ollama_url)

    with connect_database(database) as connection:
        initialize_database(connection)
        rows = connection.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.name AS filename,
                d.path AS path,
                d.relative_path,
                d.path AS absolute_path,
                d.scan_root,
                c.page_number,
                c.chunk_index,
                c.start_char,
                c.end_char,
                d.document_type,
                d.content_category,
                c.chunk_strategy,
                c.chunk_text,
                e.embedding,
                e.dimension
            FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE e.model = ?
            """,
            (model,),
        ).fetchall()

    scored_results: list[tuple[float, object]] = []
    for row in rows:
        embedding = blob_to_embedding(row["embedding"], row["dimension"])
        similarity = cosine_similarity(query_embedding, embedding)
        scored_results.append((similarity, row))

    scored_results.sort(key=lambda item: item[0], reverse=True)
    return [
        SemanticResult(
            rank=index,
            similarity=similarity,
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            path=row["path"],
            relative_path=row["relative_path"],
            absolute_path=row["absolute_path"],
            scan_root=row["scan_root"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            document_type=row["document_type"],
            content_category=row["content_category"],
            chunk_strategy=row["chunk_strategy"],
            text=row["chunk_text"],
        )
        for index, (similarity, row) in enumerate(scored_results[:top_k], start=1)
    ]


def print_results(query: str, results: list[SemanticResult], model: str) -> None:
    print(f"Query: {query}")
    print(f"Model: {model}")

    if not results:
        print("\nNo embeddings/results. Run scripts/build_embeddings.py first.")
        return

    for result in results:
        print()
        print(f"{result.rank}. similarity: {result.similarity:.6f}")
        print(f"   file: {result.filename}")
        print(f"   path: {result.relative_path or result.filename}")
        print(f"   location: {format_source(result).location}")
        print(f"   type: {result.document_type}")
        print(f"   category: {result.content_category or 'unknown'}")
        print(f"   strategy: {result.chunk_strategy}")
        print(f"   chunk_id: {result.chunk_id}")
        print(f"   document_id: {result.document_id}")
        print(f"   text: {compact_text(result.text)}")


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)
    try:
        results = search_embeddings(
            database=database,
            query=args.query,
            top_k=args.top_k,
            model=args.model,
            ollama_url=args.ollama_url,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(error)
        return 1

    print_results(args.query, results, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
