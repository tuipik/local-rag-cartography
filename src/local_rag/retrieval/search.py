"""SQLite FTS5 retrieval baseline over chunks.

This stage deliberately does not use embeddings, Ollama, Qdrant, or an LLM.
It provides a transparent keyword/full-text baseline for later comparison.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from local_rag.database import (
    add_database_argument,
    connect_database,
    resolve_database,
)


DEFAULT_TOP_K = 10
TOKEN_PATTERN = re.compile(r"[\wА-Яа-яІіЇїЄєҐґ]+", re.UNICODE)


@dataclass(frozen=True)
class SearchResult:
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
    boost: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search chunks with SQLite FTS5 without LLM or embeddings."
    )
    add_database_argument(parser)
    parser.add_argument("query", help="keyword/full-text query")
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"number of results to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--document-type",
        help="filter by documents.document_type",
    )
    parser.add_argument(
        "--content-category",
        help="filter by documents.content_category",
    )
    parser.add_argument(
        "--chunk-strategy",
        help="filter by chunks.chunk_strategy",
    )
    parser.add_argument(
        "--prefer-reference",
        action="store_true",
        help="soft-rank standard_or_norms/reference chunks above generic material",
    )
    parser.add_argument(
        "--no-rebuild-index",
        action="store_true",
        help="use existing chunks_fts table without rebuilding it first",
    )
    return parser.parse_args()


def initialize_fts(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_text,
            filename,
            path,
            document_type,
            content_category,
            chunk_strategy,
            tokenize = 'unicode61'
        )
        """
    )


def rebuild_fts(connection: sqlite3.Connection) -> int:
    initialize_fts(connection)
    connection.execute("DELETE FROM chunks_fts")
    rows = connection.execute(
        """
        SELECT
            c.id AS chunk_id,
            c.chunk_text,
            d.name AS filename,
            d.relative_path,
            d.path AS absolute_path,
            d.scan_root,
            d.document_type,
            COALESCE(d.content_category, '') AS content_category,
            c.chunk_strategy
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.id
        """
    ).fetchall()
    connection.executemany(
        """
        INSERT INTO chunks_fts (
            rowid,
            chunk_text,
            filename,
            path,
            document_type,
            content_category,
            chunk_strategy
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["chunk_id"],
                row["chunk_text"],
                row["filename"],
                row["relative_path"],
                row["document_type"],
                row["content_category"],
                row["chunk_strategy"],
            )
            for row in rows
        ],
    )
    return len(rows)


def tokenize_query(query: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(query.lower()):
        token = match.group(0)
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def quote_fts_token(token: str) -> str:
    escaped = token.replace('"', '""')
    return f'"{escaped}"'


def build_match_expression(query: str, *, require_all_terms: bool) -> str:
    tokens = tokenize_query(query)
    if not tokens:
        return quote_fts_token(query.strip())
    operator = " AND " if require_all_terms else " OR "
    return operator.join(quote_fts_token(token) for token in tokens)


def fetch_results(
    connection: sqlite3.Connection,
    match_expression: str,
    top_k: int,
    document_type: str | None = None,
    content_category: str | None = None,
    chunk_strategy: str | None = None,
    prefer_reference: bool = False,
) -> list[SearchResult]:
    filters = []
    text_match_expression = f"chunk_text : ({match_expression})"
    params: list[object] = [text_match_expression]
    if document_type:
        filters.append("d.document_type = ?")
        params.append(document_type)
    if content_category:
        filters.append("d.content_category = ?")
        params.append(content_category)
    if chunk_strategy:
        filters.append("c.chunk_strategy = ?")
        params.append(chunk_strategy)

    where_filters = ""
    if filters:
        where_filters = " AND " + " AND ".join(filters)
    params.append(top_k)

    rows = connection.execute(
        f"""
        SELECT
            c.id AS chunk_id,
            c.document_id,
            d.name AS filename,
            d.path AS path,
            d.relative_path,
            d.path AS absolute_path,
            d.scan_root,
            c.page_number,
            d.document_type,
            d.content_category,
            c.chunk_strategy,
            c.chunk_text,
            bm25(chunks_fts, 1.0, 4.0, 3.0, 1.5, 2.0, 0.5) AS bm25_score,
            CASE
                WHEN ? = 1
                 AND d.document_type = 'standard_or_norms'
                 AND d.content_category = 'reference'
                THEN 5.0
                WHEN ? = 1
                 AND d.content_category = 'reference'
                THEN 1.0
                ELSE 0.0
            END AS metadata_boost
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN documents d ON d.id = c.document_id
        WHERE chunks_fts MATCH ?
        {where_filters}
        ORDER BY (bm25_score - metadata_boost) ASC
        LIMIT ?
        """,
        [int(prefer_reference), int(prefer_reference), *params],
    ).fetchall()

    return [
        SearchResult(
            rank=index,
            score=-float(row["bm25_score"]) + float(row["metadata_boost"]),
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            path=row["path"],
            relative_path=row["relative_path"],
            absolute_path=row["absolute_path"],
            scan_root=row["scan_root"],
            page_number=row["page_number"],
            document_type=row["document_type"],
            content_category=row["content_category"],
            chunk_strategy=row["chunk_strategy"],
            text=row["chunk_text"],
            boost=float(row["metadata_boost"]),
        )
        for index, row in enumerate(rows, start=1)
    ]


def search(
    database: Path,
    query: str,
    top_k: int,
    rebuild_index: bool = True,
    document_type: str | None = None,
    content_category: str | None = None,
    chunk_strategy: str | None = None,
    prefer_reference: bool = False,
) -> tuple[list[SearchResult], str, int | None]:
    with connect_database(database) as connection:
        if rebuild_index:
            indexed_chunks = rebuild_fts(connection)
        else:
            initialize_fts(connection)
            indexed_chunks = None

        match_expression = build_match_expression(query, require_all_terms=True)
        results = fetch_results(
            connection,
            match_expression=match_expression,
            top_k=top_k,
            document_type=document_type,
            content_category=content_category,
            chunk_strategy=chunk_strategy,
            prefer_reference=prefer_reference,
        )
        if not results:
            match_expression = build_match_expression(query, require_all_terms=False)
            results = fetch_results(
                connection,
                match_expression=match_expression,
                top_k=top_k,
                document_type=document_type,
                content_category=content_category,
                chunk_strategy=chunk_strategy,
                prefer_reference=prefer_reference,
            )

    return results, match_expression, indexed_chunks


def compact_text(text: str, limit: int = 700) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def print_results(
    query: str,
    results: list[SearchResult],
    match_expression: str,
    indexed_chunks: int | None,
) -> None:
    print(f"Query: {query}")
    print(f"FTS query: {match_expression}")
    if indexed_chunks is not None:
        print(f"Indexed chunks: {indexed_chunks}")

    if not results:
        print("\nNo results.")
        return

    for result in results:
        print()
        print(f"{result.rank}. score: {result.score:.6f}")
        print(f"   file: {result.filename}")
        print(f"   path: {result.relative_path or result.filename}")
        print(f"   page: {result.page_number}")
        print(f"   type: {result.document_type}")
        print(f"   category: {result.content_category or 'unknown'}")
        print(f"   strategy: {result.chunk_strategy}")
        if result.boost:
            print(f"   boost: {result.boost:.1f}")
        print(f"   chunk_id: {result.chunk_id}")
        print(f"   document_id: {result.document_id}")
        print(f"   text: {compact_text(result.text)}")


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)
    try:
        results, match_expression, indexed_chunks = search(
            database=database,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=not args.no_rebuild_index,
            document_type=args.document_type,
            content_category=args.content_category,
            chunk_strategy=args.chunk_strategy,
            prefer_reference=args.prefer_reference,
        )
    except FileNotFoundError as error:
        print(error)
        return 2
    except sqlite3.OperationalError as error:
        print(f"SQLite FTS error: {error}")
        return 1

    print_results(
        query=args.query,
        results=results,
        match_expression=match_expression,
        indexed_chunks=indexed_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
