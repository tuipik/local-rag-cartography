#!/usr/bin/env python3
"""Rule-based document metadata extraction.

Stage 3 deliberately avoids LLM calls. It uses file names, folder names and the
first text characters already stored in document_pages.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from local_rag.database import (  # noqa: E402
    add_database_argument,
    add_missing_columns,
    connect_database,
    resolve_database,
)
from local_rag.reporting import print_counter  # noqa: E402

TEXT_LIMIT = 5000


@dataclass(frozen=True)
class DocumentInput:
    id: int
    name: str
    relative_path: str
    folder_category: str
    text_start: str


@dataclass(frozen=True)
class Metadata:
    document_type: str | None
    content_category: str | None
    document_number: str | None
    document_date: str | None
    organization: str | None
    metadata_status: str
    metadata_notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract document-level metadata with simple rules."
    )
    add_database_argument(parser)
    parser.add_argument(
        "--text-limit",
        type=int,
        default=TEXT_LIMIT,
        help=f"characters from document_pages to inspect (default: {TEXT_LIMIT})",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value.lower()).strip()


def initialize_database(connection) -> None:
    migrations = {
        "document_number": "TEXT",
        "document_date": "TEXT",
        "organization": "TEXT",
        "metadata_status": "TEXT",
        "metadata_notes": "TEXT",
    }
    add_missing_columns(connection, "documents", migrations)


def fetch_documents(connection, text_limit: int) -> list[DocumentInput]:
    rows = connection.execute(
        """
        SELECT
            d.id,
            d.name,
            d.relative_path,
            d.folder_category,
            COALESCE(
                substr(
                    (
                        SELECT group_concat(p.text, ' ')
                        FROM document_pages p
                        WHERE p.document_id = d.id
                        ORDER BY p.page_number
                    ),
                    1,
                    ?
                ),
                ''
            ) AS text_start
        FROM documents d
        ORDER BY d.id
        """,
        (text_limit,),
    ).fetchall()
    return [
        DocumentInput(
            id=row["id"],
            name=row["name"],
            relative_path=row["relative_path"],
            folder_category=row["folder_category"],
            text_start=row["text_start"],
        )
        for row in rows
    ]


def first_match(patterns: list[tuple[str, str]], haystack: str) -> tuple[str | None, str | None]:
    for value, pattern in patterns:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return value, pattern
    return None, None


def classify_document_type(haystack: str) -> tuple[str | None, str | None]:
    patterns = [
        ("order", r"\bнаказ\b"),
        ("law", r"\bзакон\b|верховн[а-яіїєґ]+\s+рад[а-яіїєґ]+"),
        ("doctrine", r"доктрин"),
        ("algorithm", r"алгоритм"),
        ("instruction", r"інструкц|iнструкц|\bінстр\b|\biнстр\b"),
        ("standard_or_norms", r"норм[иа]\b|норми\s+часу|норми\s+витрат|стандарт|standardization|nga\.stnd|nga\.sig|умовн[іi]\s+знаки"),
        ("training_manual", r"методич|методика|посібник|підручник|навчальн|student\s*workbook|workbook|порадник"),
        ("procedure", r"порядок|правила|тимчасовий\s+порядок"),
        ("article", r"статт[яі]|газет"),
        ("report", r"\bзвіт\b"),
        ("list", r"\bсписок\b"),
        ("poster", r"плакат|фотосхема|фотокарта"),
        ("scheme", r"\bсхема\b"),
        ("code", r"\bsub\b|commandbutton|код\b"),
        ("appendix", r"додаток"),
        ("brandbook", r"брендбук|грифування"),
    ]
    return first_match(patterns, haystack)


def classify_content_category(haystack: str) -> tuple[str | None, str | None]:
    patterns = [
        ("reference", r"умовн[іi]\s+знаки|норми\s+часу|норми\s+витрат|грід|grid|довід|стандарт|standard|врізка|скорочена\s+пам"),
        ("cartography_geodesy", r"геопросторов|топограф|картограф|геодез|mgrs|utm|wgs\s*84|схилен|зближен|фотокарт|спеціальн[а-яіїєґ]+\s+карт"),
        ("training", r"навчальн|методич|методика|посібник|підручник|student\s*workbook|workbook|порадник|план\s+конспект"),
        ("operations", r"оперативн|бойов|чергов|повітряна\s+тривога|реагування|кпп|бпла|рхб"),
        ("communications_it", r"starlink|mikrotik|веб\s*сайт|портал|arcgis|coraldraw|photoshop|acrobat|3d\s*друк|3d"),
        ("legal_policy", r"\bнаказ\b|\bзакон\b|правовий\s+захист|порядок"),
        ("branding", r"брендбук|грифування"),
    ]
    return first_match(patterns, haystack)


def extract_document_number(raw_text: str) -> tuple[str | None, str | None]:
    patterns = [
        r"\bнаказ(?:\s+\S+){0,5}?\s+(\d{1,5})\b",
        r"\bзакон\s*№\s*([0-9]+[-–]?[A-ZА-ЯІЇЄҐXIVLC]+)\b",
        r"\b№\s*([0-9]+[-–]?[A-ZА-ЯІЇЄҐXIVLC]*)\b",
        r"\b(ПВП\s*\d+[-–]?\([^)]+\)\d+)\b",
        r"\b(ПДП\s*\d+[-–]?\([^)]+\)\d+)\b",
        r"\b(NGA\.(?:SIG|STND)\.\d+(?:_\d+\.\d+\.\d+)?)\b",
        r"\b(\d{3}[-–]\d{10,}[-–][\d-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip(), pattern
    return None, None


def normalize_year(year: str) -> int:
    value = int(year)
    if value < 100:
        return 2000 + value if value <= 49 else 1900 + value
    return value


def extract_document_date(raw_text: str) -> tuple[str | None, str | None]:
    numeric = re.search(r"\b(\d{1,2})[.](\d{1,2})[.](\d{2,4})\b", raw_text)
    if numeric:
        day, month, year = numeric.groups()
        try:
            return date(normalize_year(year), int(month), int(day)).isoformat(), "numeric_date"
        except ValueError:
            pass

    iso_like = re.search(r"\b(20\d{2})[-_](\d{1,2})[-_](\d{1,2})\b", raw_text)
    if iso_like:
        year, month, day = iso_like.groups()
        try:
            return date(int(year), int(month), int(day)).isoformat(), "iso_like_date"
        except ValueError:
            pass

    return None, None


def extract_organization(haystack: str) -> tuple[str | None, str | None]:
    patterns = [
        ("Міністерство оборони України", r"міністерство\s+оборони\s+україни"),
        ("Генеральний штаб Збройних Сил України", r"генеральн[ийого]+\s+штаб|гш\s+зсу"),
        ("Командування Сил підтримки Збройних Сил України", r"командування\s+сил\s+підтримки"),
        ("Збройні Сили України", r"збройн[іi]\s+сил[иа]\s+україни|\bзсу\b"),
        ("National Geospatial-Intelligence Agency", r"national\s+geospatial|nga\."),
        ("Esri", r"\besri\b|arcgis"),
        ("Верховна Рада України", r"верховн[а-яіїєґ]+\s+рад[а-яіїєґ]+"),
    ]
    return first_match(patterns, haystack)


def extract_metadata(document: DocumentInput) -> Metadata:
    name_haystack = normalize_text(document.name)
    path_haystack = normalize_text(
        " ".join((document.relative_path, document.folder_category))
    )
    raw = " ".join(
        part
        for part in (
            document.name,
            document.relative_path,
            document.folder_category,
            document.text_start,
        )
        if part
    )
    haystack = normalize_text(raw)
    notes: list[str] = []

    document_type, type_rule = classify_document_type(name_haystack)
    type_source = "name"
    if not document_type:
        document_type, type_rule = classify_document_type(path_haystack)
        type_source = "path"
    if not document_type:
        document_type, type_rule = classify_document_type(haystack)
        type_source = "text"
    if type_rule:
        notes.append(f"document_type:{type_source}:{type_rule}")

    content_category, category_rule = classify_content_category(haystack)
    if category_rule:
        notes.append(f"content_category:{category_rule}")

    document_number, number_rule = extract_document_number(raw)
    if number_rule:
        notes.append(f"document_number:{number_rule}")

    document_date, date_rule = extract_document_date(raw)
    if date_rule:
        notes.append(f"document_date:{date_rule}")

    organization, organization_rule = extract_organization(haystack)
    if organization_rule:
        notes.append(f"organization:{organization_rule}")

    extracted_fields = [
        document_type,
        content_category,
        document_number,
        document_date,
        organization,
    ]
    if all(extracted_fields):
        status = "metadata_extracted"
    elif any(extracted_fields):
        status = "metadata_partial"
    else:
        status = "metadata_empty"
        notes.append("no_rules_matched")

    return Metadata(
        document_type=document_type or "unknown",
        content_category=content_category,
        document_number=document_number,
        document_date=document_date,
        organization=organization,
        metadata_status=status,
        metadata_notes="; ".join(notes),
    )


def save_metadata(
    connection,
    document_id: int,
    metadata: Metadata,
) -> None:
    connection.execute(
        """
        UPDATE documents
        SET document_type = ?,
            content_category = ?,
            document_number = ?,
            document_date = ?,
            organization = ?,
            metadata_status = ?,
            metadata_notes = ?
        WHERE id = ?
        """,
        (
            metadata.document_type,
            metadata.content_category,
            metadata.document_number,
            metadata.document_date,
            metadata.organization,
            metadata.metadata_status,
            metadata.metadata_notes,
            document_id,
        ),
    )


def run_metadata_extraction(database: Path, text_limit: int) -> tuple[int, Counter[str], Counter[str], Counter[str]]:
    type_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    with connect_database(database) as connection:
        initialize_database(connection)
        documents = fetch_documents(connection, text_limit=text_limit)
        for document in documents:
            metadata = extract_metadata(document)
            save_metadata(connection, document.id, metadata)
            type_counts[metadata.document_type or "unknown"] += 1
            category_counts[metadata.content_category or "unknown"] += 1
            status_counts[metadata.metadata_status] += 1

    return len(documents), type_counts, category_counts, status_counts


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)

    try:
        total, type_counts, category_counts, status_counts = run_metadata_extraction(
            database,
            text_limit=args.text_limit,
        )
    except FileNotFoundError as error:
        print(error)
        return 2
    print("\nСтатистика метаданих")
    print(f"Documents processed: {total}")
    print_counter("document_type", type_counts)
    print_counter("content_category", category_counts)
    print_counter("metadata_status", status_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
