#!/usr/bin/env python3
"""Scan a document directory and store its catalog in SQLite."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from local_rag.database import (  # noqa: E402
    add_database_argument,
    add_missing_columns,
    connect_database,
    resolve_database,
    table_columns,
)


SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt"}
DOCUMENT_TYPES = {
    ".pdf": "pdf",
    ".doc": "word",
    ".docx": "word",
    ".txt": "text",
}
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Document:
    path: str
    relative_path: str
    name: str
    extension: str
    document_type: str
    parent_dir: str
    relative_dir: str
    size_bytes: int
    modified_at: str
    sha256: str
    folder_category: str
    content_category: str | None
    is_temp_file: bool
    is_supported: bool
    is_duplicate_candidate: bool
    pages_count: int | None
    has_text: bool | None
    ocr_required: bool | None
    scan_status: str
    scan_root: str
    scanned_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Catalog PDF, DOC, DOCX and TXT files in SQLite."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="directory to scan recursively (default: current directory)",
    )
    add_database_argument(parser)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def folder_category_for(relative_path: Path) -> str:
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "uncategorized"


def iter_document_paths(root: Path, database: Path) -> Iterable[Path]:
    database = database.resolve()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.resolve() == database:
            continue
        yield path


def scan_documents(
    root: Path, database: Path
) -> tuple[list[Document], list[tuple[Path, str]]]:
    documents: list[Document] = []
    errors: list[tuple[Path, str]] = []
    scanned_at = datetime.now(timezone.utc).isoformat()

    for path in iter_document_paths(root, database):
        try:
            stat = path.stat()
            relative_path = path.relative_to(root)
            documents.append(
                Document(
                    path=str(path.resolve()),
                    relative_path=str(relative_path),
                    name=path.name,
                    extension=path.suffix.lower().lstrip("."),
                    document_type=DOCUMENT_TYPES[path.suffix.lower()],
                    parent_dir=path.parent.name,
                    relative_dir=(
                        str(relative_path.parent)
                        if relative_path.parent != Path(".")
                        else ""
                    ),
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(),
                    sha256=sha256_file(path),
                    folder_category=folder_category_for(relative_path),
                    content_category=None,
                    is_temp_file=False,
                    is_supported=True,
                    is_duplicate_candidate=False,
                    pages_count=None,
                    has_text=None,
                    ocr_required=None,
                    scan_status="cataloged",
                    scan_root=str(root),
                    scanned_at=scanned_at,
                )
            )
        except (OSError, ValueError) as error:
            errors.append((path, str(error)))

    hash_counts = Counter(document.sha256 for document in documents)
    documents = [
        replace(
            document,
            is_duplicate_candidate=hash_counts[document.sha256] > 1,
        )
        for document in documents
    ]
    return documents, errors


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            relative_path TEXT NOT NULL,
            name TEXT NOT NULL,
            extension TEXT NOT NULL,
            document_type TEXT NOT NULL,
            parent_dir TEXT NOT NULL,
            relative_dir TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            folder_category TEXT NOT NULL,
            content_category TEXT,
            is_temp_file INTEGER NOT NULL DEFAULT 0,
            is_supported INTEGER NOT NULL DEFAULT 1,
            is_duplicate_candidate INTEGER NOT NULL DEFAULT 0,
            pages_count INTEGER,
            has_text INTEGER,
            ocr_required INTEGER,
            scan_status TEXT NOT NULL DEFAULT 'cataloged',
            scan_root TEXT NOT NULL,
            scanned_at TEXT NOT NULL
        )
        """
    )

    columns = table_columns(connection, "documents")
    if "category" in columns and "folder_category" not in columns:
        connection.execute(
            "ALTER TABLE documents RENAME COLUMN category TO folder_category"
        )
        columns.remove("category")
        columns.add("folder_category")

    migrations = {
        "document_type": "TEXT NOT NULL DEFAULT 'unknown'",
        "parent_dir": "TEXT NOT NULL DEFAULT ''",
        "relative_dir": "TEXT NOT NULL DEFAULT ''",
        "content_category": "TEXT",
        "is_temp_file": "INTEGER NOT NULL DEFAULT 0",
        "is_supported": "INTEGER NOT NULL DEFAULT 1",
        "is_duplicate_candidate": "INTEGER NOT NULL DEFAULT 0",
        "pages_count": "INTEGER",
        "has_text": "INTEGER",
        "ocr_required": "INTEGER",
        "scan_status": "TEXT NOT NULL DEFAULT 'cataloged'",
    }
    add_missing_columns(connection, "documents", migrations)

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256)"
    )
    connection.execute("DROP INDEX IF EXISTS idx_documents_category")
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_folder_category
        ON documents(folder_category)
        """
    )


def save_documents(database: Path, root: Path, documents: list[Document]) -> None:
    with connect_database(
        database,
        require_exists=False,
        create_parent=True,
        row_factory=False,
    ) as connection:
        initialize_database(connection)
        connection.executemany(
            """
            INSERT INTO documents (
                path, relative_path, name, extension, document_type, parent_dir,
                relative_dir, size_bytes, modified_at, sha256, folder_category,
                content_category, is_temp_file, is_supported,
                is_duplicate_candidate, pages_count, has_text, ocr_required,
                scan_status, scan_root, scanned_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(path) DO UPDATE SET
                relative_path = excluded.relative_path,
                name = excluded.name,
                extension = excluded.extension,
                document_type = excluded.document_type,
                parent_dir = excluded.parent_dir,
                relative_dir = excluded.relative_dir,
                size_bytes = excluded.size_bytes,
                modified_at = excluded.modified_at,
                sha256 = excluded.sha256,
                folder_category = excluded.folder_category,
                is_temp_file = excluded.is_temp_file,
                is_supported = excluded.is_supported,
                is_duplicate_candidate = excluded.is_duplicate_candidate,
                scan_root = excluded.scan_root,
                scanned_at = excluded.scanned_at
            """,
            [
                (
                    document.path,
                    document.relative_path,
                    document.name,
                    document.extension,
                    document.document_type,
                    document.parent_dir,
                    document.relative_dir,
                    document.size_bytes,
                    document.modified_at,
                    document.sha256,
                    document.folder_category,
                    document.content_category,
                    document.is_temp_file,
                    document.is_supported,
                    document.is_duplicate_candidate,
                    document.pages_count,
                    document.has_text,
                    document.ocr_required,
                    document.scan_status,
                    document.scan_root,
                    document.scanned_at,
                )
                for document in documents
            ],
        )
        connection.execute(
            """
            CREATE TEMP TABLE current_scan_paths (
                path TEXT PRIMARY KEY
            )
            """
        )
        connection.executemany(
            "INSERT INTO current_scan_paths(path) VALUES (?)",
            [(document.path,) for document in documents],
        )
        connection.execute(
            """
            DELETE FROM documents
            WHERE scan_root = ?
              AND path NOT IN (SELECT path FROM current_scan_paths)
            """,
            (str(root),),
        )


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")


def duplicate_groups(documents: list[Document]) -> list[list[Document]]:
    by_hash: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_hash[document.sha256].append(document)
    return [group for group in by_hash.values() if len(group) > 1]


def print_statistics(
    documents: list[Document], errors: list[tuple[Path, str]]
) -> None:
    types = Counter(document.extension for document in documents)
    categories = Counter(document.folder_category for document in documents)
    duplicates = duplicate_groups(documents)

    print("\nСтатистика каталогу")
    print(f"Файлів: {len(documents)}")
    print(f"Загальний розмір: {human_size(sum(d.size_bytes for d in documents))}")

    print("\nТипи:")
    if types:
        for extension, count in sorted(types.items()):
            print(f"  {extension.upper()}: {count}")
    else:
        print("  немає")

    print("\nКатегорії за папками:")
    if categories:
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")
    else:
        print("  немає")

    duplicate_file_count = sum(len(group) for group in duplicates)
    reclaimable_size = sum(
        group[0].size_bytes * (len(group) - 1) for group in duplicates
    )
    print("\nПотенційні дублікати (однаковий SHA-256):")
    print(f"  груп: {len(duplicates)}")
    print(f"  файлів у групах: {duplicate_file_count}")
    print(f"  потенційно зайвий розмір: {human_size(reclaimable_size)}")
    for index, group in enumerate(duplicates, start=1):
        print(f"  Група {index} ({human_size(group[0].size_bytes)}):")
        for document in group:
            print(f"    {document.relative_path}")

    if errors:
        print(f"\nНе вдалося прочитати файлів: {len(errors)}", file=sys.stderr)
        for path, message in errors:
            print(f"  {path}: {message}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    root = args.directory.expanduser().resolve()
    database = resolve_database(args.database)

    if not root.is_dir():
        print(f"Помилка: папку не знайдено: {root}", file=sys.stderr)
        return 2

    documents, errors = scan_documents(root, database)
    try:
        save_documents(database, root, documents)
    except sqlite3.Error as error:
        print(f"Помилка SQLite: {error}", file=sys.stderr)
        return 1

    print(f"Проскановано: {root}")
    print(f"База даних: {database}")
    print_statistics(documents, errors)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
