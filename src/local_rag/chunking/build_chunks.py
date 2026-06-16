"""Rule-based baseline chunk builder.

This stage deliberately does not create embeddings and does not call an LLM.
It creates reproducible text chunks with provenance for later retrieval tests.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from local_rag.database import (
    add_database_argument,
    connect_database,
    resolve_database,
)
from local_rag.reporting import print_counter


PAGE_CHUNK_LIMIT = 3000
TARGET_CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300
MIN_STRUCTURE_MARKERS = 3

POINT_PATTERN = re.compile(
    r"(?m)^(?P<marker>\s*\d+(?:\.\d+){0,2}\.)\s+(?=\S)"
)
TABLE_PATTERN = re.compile(r"\bтаблиц[яі]\b|\btable\b", re.IGNORECASE)


@dataclass(frozen=True)
class Document:
    id: int
    relative_path: str
    document_type: str
    content_category: str | None
    has_text: int | None
    ocr_required: int | None


@dataclass(frozen=True)
class Page:
    id: int
    document_id: int
    page_number: int
    text: str
    text_length: int


@dataclass(frozen=True)
class Chunk:
    document_id: int
    page_id: int | None
    page_number: int | None
    chunk_index: int
    chunk_text: str
    chunk_strategy: str
    source_type: str
    start_char: int | None
    end_char: int | None

    @property
    def chunk_length(self) -> int:
        return len(self.chunk_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rule-based baseline chunks from document_pages."
    )
    add_database_argument(parser)
    parser.add_argument(
        "--page-chunk-limit",
        type=int,
        default=PAGE_CHUNK_LIMIT,
        help=f"page length that can stay as one chunk (default: {PAGE_CHUNK_LIMIT})",
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=TARGET_CHUNK_SIZE,
        help=f"target split size for long pages (default: {TARGET_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"character overlap for long-page splits (default: {CHUNK_OVERLAP})",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            document_id INTEGER NOT NULL,
            page_id INTEGER,
            page_number INTEGER,

            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_length INTEGER NOT NULL,

            chunk_strategy TEXT NOT NULL,
            source_type TEXT,

            start_char INTEGER,
            end_char INTEGER,

            created_at TEXT NOT NULL,

            FOREIGN KEY(document_id) REFERENCES documents(id),
            FOREIGN KEY(page_id) REFERENCES document_pages(id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
        ON chunks(document_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_page_id
        ON chunks(page_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_strategy
        ON chunks(chunk_strategy)
        """
    )


def fetch_documents(connection: sqlite3.Connection) -> list[Document]:
    rows = connection.execute(
        """
        SELECT
            id,
            relative_path,
            document_type,
            content_category,
            has_text,
            ocr_required
        FROM documents
        ORDER BY id
        """
    ).fetchall()
    return [
        Document(
            id=row["id"],
            relative_path=row["relative_path"],
            document_type=row["document_type"] or "unknown",
            content_category=row["content_category"],
            has_text=row["has_text"],
            ocr_required=row["ocr_required"],
        )
        for row in rows
    ]


def fetch_pages_by_document(
    connection: sqlite3.Connection,
) -> dict[int, list[Page]]:
    rows = connection.execute(
        """
        SELECT id, document_id, page_number, text, text_length
        FROM document_pages
        ORDER BY document_id, page_number
        """
    ).fetchall()
    pages_by_document: dict[int, list[Page]] = {}
    for row in rows:
        page = Page(
            id=row["id"],
            document_id=row["document_id"],
            page_number=row["page_number"],
            text=row["text"],
            text_length=row["text_length"],
        )
        pages_by_document.setdefault(page.document_id, []).append(page)
    return pages_by_document


def count_point_markers(text: str) -> int:
    return len(POINT_PATTERN.findall(text))


def is_table_heavy(document: Document, page: Page) -> bool:
    table_count = len(TABLE_PATTERN.findall(page.text))
    return document.content_category == "reference" and table_count >= 2


def is_structure_candidate(document: Document, page: Page) -> bool:
    if document.document_type not in {"order", "procedure", "instruction", "doctrine"}:
        return False
    return count_point_markers(page.text) >= MIN_STRUCTURE_MARKERS


def normalize_chunk_text(text: str) -> str:
    return text.strip()


def split_long_text(
    text: str,
    target_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    if target_size <= overlap:
        raise ValueError("target_size must be greater than overlap")

    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + target_size, text_length)
        if end < text_length:
            paragraph_break = text.rfind("\n\n", start, end)
            line_break = text.rfind("\n", start, end)
            sentence_break = max(
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
            )
            best_break = max(paragraph_break, line_break, sentence_break)
            if best_break > start + target_size // 2:
                end = best_break + 1

        chunk_text = normalize_chunk_text(text[start:end])
        if chunk_text:
            chunks.append((chunk_text, start, end))

        if end >= text_length:
            break
        start = max(0, end - overlap)

    return chunks


def split_by_points(
    text: str,
    target_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    matches = list(POINT_PATTERN.finditer(text))
    if len(matches) < MIN_STRUCTURE_MARKERS:
        return split_long_text(text, target_size, overlap)

    sections: list[tuple[int, int]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append((0, matches[0].start()))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if text[start:end].strip():
            sections.append((start, end))

    chunks: list[tuple[str, int, int]] = []
    buffer_parts: list[tuple[str, int, int]] = []
    buffer_length = 0

    def flush_buffer() -> None:
        nonlocal buffer_parts, buffer_length
        if not buffer_parts:
            return
        start = buffer_parts[0][1]
        end = buffer_parts[-1][2]
        chunk_text = normalize_chunk_text("".join(part[0] for part in buffer_parts))
        if chunk_text:
            chunks.append((chunk_text, start, end))
        buffer_parts = []
        buffer_length = 0

    for start, end in sections:
        section_text = text[start:end]
        section_clean = normalize_chunk_text(section_text)
        if not section_clean:
            continue

        if len(section_clean) > target_size:
            flush_buffer()
            chunks.extend(
                (chunk_text, start + chunk_start, start + chunk_end)
                for chunk_text, chunk_start, chunk_end in split_long_text(
                    section_text,
                    target_size,
                    overlap,
                )
            )
            continue

        if buffer_length and buffer_length + len(section_text) > target_size:
            flush_buffer()
        buffer_parts.append((section_text, start, end))
        buffer_length += len(section_text)

    flush_buffer()
    return chunks


def build_page_chunks(
    document: Document,
    page: Page,
    chunk_index_start: int,
    page_chunk_limit: int,
    target_chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    text = page.text or ""
    clean_text = normalize_chunk_text(text)
    if not clean_text:
        return []

    if is_structure_candidate(document, page):
        strategy = "structure_based_points"
        source_type = "structure"
        pieces = split_by_points(text, target_chunk_size, overlap)
    elif is_table_heavy(document, page) and page.text_length <= page_chunk_limit:
        strategy = "table_or_page_based"
        source_type = "table_page"
        pieces = [(clean_text, 0, len(text))]
    elif page.text_length <= page_chunk_limit:
        strategy = "document_or_page_based" if page.page_number == 1 else "page_based"
        source_type = "page"
        pieces = [(clean_text, 0, len(text))]
    else:
        strategy = "split_long_page"
        source_type = "long_page"
        pieces = split_long_text(text, target_chunk_size, overlap)

    chunks: list[Chunk] = []
    for offset, (chunk_text, start_char, end_char) in enumerate(pieces):
        normalized = normalize_chunk_text(chunk_text)
        if not normalized:
            continue
        chunks.append(
            Chunk(
                document_id=document.id,
                page_id=page.id,
                page_number=page.page_number,
                chunk_index=chunk_index_start + offset,
                chunk_text=normalized,
                chunk_strategy=strategy,
                source_type=source_type,
                start_char=start_char,
                end_char=end_char,
            )
        )
    return chunks


def save_chunks(connection: sqlite3.Connection, chunks: list[Chunk]) -> None:
    created_at = utc_now()
    connection.execute("DELETE FROM chunks")
    connection.executemany(
        """
        INSERT INTO chunks (
            document_id,
            page_id,
            page_number,
            chunk_index,
            chunk_text,
            chunk_length,
            chunk_strategy,
            source_type,
            start_char,
            end_char,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                chunk.document_id,
                chunk.page_id,
                chunk.page_number,
                chunk.chunk_index,
                chunk.chunk_text,
                chunk.chunk_length,
                chunk.chunk_strategy,
                chunk.source_type,
                chunk.start_char,
                chunk.end_char,
                created_at,
            )
            for chunk in chunks
        ],
    )


def describe_lengths(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(mean(lengths), 1),
        "median": round(median(lengths), 1),
    }


def build_chunks(
    database: Path,
    page_chunk_limit: int,
    target_chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], Counter[str], Counter[str]]:
    chunks: list[Chunk] = []
    stats: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    with connect_database(database) as connection:
        initialize_database(connection)
        documents = fetch_documents(connection)
        pages_by_document = fetch_pages_by_document(connection)

        for document in documents:
            if document.ocr_required:
                skipped["ocr_required"] += 1
                continue
            if not document.has_text:
                skipped["no_text"] += 1
                continue

            pages = pages_by_document.get(document.id, [])
            if not pages:
                skipped["no_text"] += 1
                continue

            stats["documents_processed"] += 1
            document_chunk_index = 0
            for page in pages:
                page_chunks = build_page_chunks(
                    document=document,
                    page=page,
                    chunk_index_start=document_chunk_index,
                    page_chunk_limit=page_chunk_limit,
                    target_chunk_size=target_chunk_size,
                    overlap=overlap,
                )
                chunks.extend(page_chunks)
                document_chunk_index += len(page_chunks)

        save_chunks(connection, chunks)

    return chunks, stats, skipped


def print_statistics(
    chunks: list[Chunk],
    stats: Counter[str],
    skipped: Counter[str],
) -> None:
    strategy_counts = Counter(chunk.chunk_strategy for chunk in chunks)
    length_stats = describe_lengths([chunk.chunk_length for chunk in chunks])
    documents_skipped = sum(skipped.values())

    print("\nChunking statistics")
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Documents skipped: {documents_skipped}")
    print(f"Chunks created: {len(chunks)}")

    print_counter("Chunks by strategy", strategy_counts)

    print("\nChunk length:")
    print(f"  min: {length_stats['min']}")
    print(f"  max: {length_stats['max']}")
    print(f"  avg: {length_stats['avg']}")
    print(f"  median: {length_stats['median']}")

    print("\nSkipped:")
    print(f"  ocr_required: {skipped['ocr_required']}")
    print(f"  no_text: {skipped['no_text']}")


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)
    try:
        chunks, stats, skipped = build_chunks(
            database=database,
            page_chunk_limit=args.page_chunk_limit,
            target_chunk_size=args.target_chunk_size,
            overlap=args.overlap,
        )
    except FileNotFoundError as error:
        print(error)
        return 2
    except ValueError as error:
        print(error)
        return 2

    print_statistics(chunks, stats, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
