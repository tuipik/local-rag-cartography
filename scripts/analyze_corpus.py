#!/usr/bin/env python3
"""Analyze extracted document text before designing chunking."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from local_rag.database import (  # noqa: E402
    add_database_argument,
    connect_database,
    resolve_database,
)
from local_rag.reporting import print_counter  # noqa: E402

SHORT_PAGE_THRESHOLD = 100
LONG_PAGE_THRESHOLD = 5000


STRUCTURAL_MARKERS = {
    "section": re.compile(r"\bрозділ\b", re.IGNORECASE),
    "chapter": re.compile(r"\bглава\b", re.IGNORECASE),
    "appendix": re.compile(r"\bдодаток\b", re.IGNORECASE),
    "point_1": re.compile(r"(?m)^\s*\d+\.\s+\S"),
    "point_1_1": re.compile(r"(?m)^\s*\d+\.\d+\.\s+\S"),
    "point_1_1_1": re.compile(r"(?m)^\s*\d+\.\d+\.\d+\.\s+\S"),
    "table": re.compile(r"\bтаблиц[яі]\b|\btable\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class Page:
    document_id: int
    page_number: int
    text: str
    text_length: int


@dataclass(frozen=True)
class Document:
    id: int
    relative_path: str
    extension: str
    document_type: str
    content_category: str | None
    pages_count: int | None
    has_text: int | None
    ocr_required: int | None
    scan_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze document_pages and documents before chunking."
    )
    add_database_argument(parser)
    parser.add_argument(
        "--short-page-threshold",
        type=int,
        default=SHORT_PAGE_THRESHOLD,
        help=f"short page threshold in characters (default: {SHORT_PAGE_THRESHOLD})",
    )
    parser.add_argument(
        "--long-page-threshold",
        type=int,
        default=LONG_PAGE_THRESHOLD,
        help=f"long page threshold in characters (default: {LONG_PAGE_THRESHOLD})",
    )
    parser.add_argument(
        "--limit-examples",
        type=int,
        default=10,
        help="maximum examples to print per list (default: 10)",
    )
    return parser.parse_args()


def fetch_documents(connection) -> list[Document]:
    rows = connection.execute(
        """
        SELECT
            id,
            relative_path,
            extension,
            document_type,
            content_category,
            pages_count,
            has_text,
            ocr_required,
            scan_status
        FROM documents
        ORDER BY id
        """
    ).fetchall()
    return [
        Document(
            id=row["id"],
            relative_path=row["relative_path"],
            extension=row["extension"],
            document_type=row["document_type"] or "unknown",
            content_category=row["content_category"],
            pages_count=row["pages_count"],
            has_text=row["has_text"],
            ocr_required=row["ocr_required"],
            scan_status=row["scan_status"],
        )
        for row in rows
    ]


def fetch_pages(connection) -> list[Page]:
    rows = connection.execute(
        """
        SELECT document_id, page_number, text, text_length
        FROM document_pages
        ORDER BY document_id, page_number
        """
    ).fetchall()
    return [
        Page(
            document_id=row["document_id"],
            page_number=row["page_number"],
            text=row["text"],
            text_length=row["text_length"],
        )
        for row in rows
    ]


def describe_lengths(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(mean(lengths), 1),
        "median": round(median(lengths), 1),
    }


def marker_counts(pages: list[Page]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        for marker, pattern in STRUCTURAL_MARKERS.items():
            matches = pattern.findall(page.text)
            if matches:
                counts[marker] += len(matches)
    return counts


def marker_document_counts(pages: list[Page]) -> dict[str, int]:
    documents_by_marker: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        for marker, pattern in STRUCTURAL_MARKERS.items():
            if pattern.search(page.text):
                documents_by_marker[marker].add(page.document_id)
    return {
        marker: len(document_ids)
        for marker, document_ids in sorted(documents_by_marker.items())
    }


def recommendation_for_type(
    document_type: str,
    content_category: str | None,
    page_count: int,
    avg_page_length: float,
    marker_counts_for_document: Counter[str],
) -> str:
    marker_total = sum(marker_counts_for_document.values())
    has_points = any(
        marker_counts_for_document[marker] > 0
        for marker in ("point_1", "point_1_1", "point_1_1_1")
    )
    has_sections = any(
        marker_counts_for_document[marker] > 0
        for marker in ("section", "chapter", "appendix")
    )
    has_tables = marker_counts_for_document["table"] > 0

    if content_category == "reference" and has_tables:
        return "table_or_page_based"
    if document_type in {"order", "procedure", "instruction", "doctrine"} and has_points:
        return "structure_based_points"
    if document_type in {"training_manual", "law"} and has_sections:
        return "section_or_page_based"
    if document_type in {"list"}:
        return "record_or_page_based"
    if document_type in {"poster", "scheme", "brandbook"}:
        return "page_based"
    if page_count <= 2 and avg_page_length <= LONG_PAGE_THRESHOLD:
        return "document_or_page_based"
    if avg_page_length > LONG_PAGE_THRESHOLD:
        return "split_long_pages"
    if marker_total == 0:
        return "page_based_with_review"
    return "page_based_initial"


def document_marker_counts(pages: list[Page]) -> dict[int, Counter[str]]:
    counts_by_document: dict[int, Counter[str]] = defaultdict(Counter)
    for page in pages:
        for marker, pattern in STRUCTURAL_MARKERS.items():
            matches = pattern.findall(page.text)
            if matches:
                counts_by_document[page.document_id][marker] += len(matches)
    return counts_by_document


def print_length_stats_by_type(
    documents: list[Document],
    pages_by_document: dict[int, list[Page]],
) -> None:
    lengths_by_type: dict[str, list[int]] = defaultdict(list)
    for document in documents:
        lengths_by_type[document.document_type].extend(
            page.text_length for page in pages_by_document.get(document.id, [])
        )

    print("\nText length by document_type:")
    print("  type | pages | min | max | avg | median")
    for document_type in sorted(lengths_by_type):
        lengths = lengths_by_type[document_type]
        stats = describe_lengths(lengths)
        print(
            "  "
            f"{document_type} | {len(lengths)} | {stats['min']} | "
            f"{stats['max']} | {stats['avg']} | {stats['median']}"
        )


def print_examples(title: str, lines: list[str], limit: int) -> None:
    print(f"\n{title}:")
    if not lines:
        print("  none")
        return
    for line in lines[:limit]:
        print(f"  {line}")
    if len(lines) > limit:
        print(f"  ... and {len(lines) - limit} more")


def analyze(
    database: Path,
    short_page_threshold: int,
    long_page_threshold: int,
    limit_examples: int,
) -> int:
    with connect_database(database) as connection:
        documents = fetch_documents(connection)
        pages = fetch_pages(connection)

    documents_by_id = {document.id: document for document in documents}
    pages_by_document: dict[int, list[Page]] = defaultdict(list)
    for page in pages:
        pages_by_document[page.document_id].append(page)

    document_counts = Counter(document.document_type for document in documents)
    page_counts = Counter()
    for document in documents:
        page_counts[document.document_type] += len(pages_by_document.get(document.id, []))

    all_lengths = [page.text_length for page in pages]
    empty_pages = [
        page for page in pages if not page.text.strip()
    ]
    short_pages = [
        page for page in pages if 0 < page.text_length < short_page_threshold
    ]
    long_pages = [
        page for page in pages if page.text_length > long_page_threshold
    ]
    documents_without_text = [
        document
        for document in documents
        if not document.has_text or not pages_by_document.get(document.id)
    ]
    ocr_candidates = [
        document for document in documents if document.ocr_required
    ]

    markers = marker_counts(pages)
    marker_docs = marker_document_counts(pages)
    markers_by_document = document_marker_counts(pages)

    recommendations = Counter()
    recommendation_examples: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        document_pages = pages_by_document.get(document.id, [])
        lengths = [page.text_length for page in document_pages]
        avg_length = mean(lengths) if lengths else 0
        recommendation = recommendation_for_type(
            document.document_type,
            document.content_category,
            len(document_pages),
            avg_length,
            markers_by_document.get(document.id, Counter()),
        )
        recommendations[recommendation] += 1
        recommendation_examples[recommendation].append(document.relative_path)

    print("Corpus analysis report")
    print(f"Documents: {len(documents)}")
    print(f"Pages/fragments: {len(pages)}")

    print_counter("Documents by document_type", document_counts)
    print_counter("Pages by document_type", page_counts)

    stats = describe_lengths(all_lengths)
    print("\nText length overall:")
    print(f"  min: {stats['min']}")
    print(f"  max: {stats['max']}")
    print(f"  avg: {stats['avg']}")
    print(f"  median: {stats['median']}")

    print_length_stats_by_type(documents, pages_by_document)

    print("\nPage quality:")
    print(f"  empty_pages: {len(empty_pages)}")
    print(f"  very_short_pages (<{short_page_threshold} chars): {len(short_pages)}")
    print(f"  very_long_pages (>{long_page_threshold} chars): {len(long_pages)}")
    print(f"  documents_without_text: {len(documents_without_text)}")
    print(f"  ocr_candidates: {len(ocr_candidates)}")

    print("\nStructural markers:")
    for marker in STRUCTURAL_MARKERS:
        print(
            f"  {marker}: {markers[marker]} occurrences "
            f"in {marker_docs.get(marker, 0)} documents"
        )

    short_examples = [
        f"{documents_by_id[page.document_id].relative_path} p.{page.page_number}: {page.text_length} chars"
        for page in short_pages
    ]
    long_examples = [
        f"{documents_by_id[page.document_id].relative_path} p.{page.page_number}: {page.text_length} chars"
        for page in long_pages
    ]
    no_text_examples = [
        f"{document.relative_path} ({document.scan_status})"
        for document in documents_without_text
    ]
    ocr_examples = [
        f"{document.relative_path} ({document.pages_count or 0} pages)"
        for document in ocr_candidates
    ]

    print_examples("Very short page examples", short_examples, limit_examples)
    print_examples("Very long page examples", long_examples, limit_examples)
    print_examples("Documents without text", no_text_examples, limit_examples)
    print_examples("OCR candidates", ocr_examples, limit_examples)

    print_counter("Preliminary chunking recommendation", recommendations)
    print("\nRecommendation examples:")
    for recommendation in sorted(recommendation_examples):
        print(f"  {recommendation}:")
        for path in recommendation_examples[recommendation][:3]:
            print(f"    {path}")
        remaining = len(recommendation_examples[recommendation]) - 3
        if remaining > 0:
            print(f"    ... and {remaining} more")

    print("\nInterpretation:")
    if recommendations["structure_based_points"] > 0:
        print("  Use point/subpoint-aware chunking for normative documents where markers exist.")
    if recommendations["table_or_page_based"] > 0:
        print("  Keep table-heavy reference documents page/table oriented.")
    if recommendations["split_long_pages"] > 0:
        print("  Some pages are too long for raw page-based chunks and need secondary splitting.")
    if ocr_candidates:
        print("  OCR candidates should stay out of chunking until OCR is implemented.")
    print("  Do not choose one global fixed-size strategy without reviewing these groups.")

    return 0


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)

    try:
        return analyze(
            database=database,
            short_page_threshold=args.short_page_threshold,
            long_page_threshold=args.long_page_threshold,
            limit_examples=args.limit_examples,
        )
    except FileNotFoundError as error:
        print(error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
