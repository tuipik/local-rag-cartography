"""Shared SQLite helpers for local RAG scripts."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DATABASE = Path("data/catalog/documents.sqlite")


def add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite database path (default: {DEFAULT_DATABASE})",
    )


def resolve_database(path: Path) -> Path:
    return path.expanduser().resolve()


def connect_database(
    database: Path,
    *,
    require_exists: bool = True,
    create_parent: bool = False,
    row_factory: bool = True,
) -> sqlite3.Connection:
    database = resolve_database(database)
    if require_exists and not database.exists():
        raise FileNotFoundError(f"SQLite database not found: {database}")
    if create_parent:
        database.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database)
    if row_factory:
        connection.row_factory = sqlite3.Row
    return connection


def table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def add_missing_columns(
    connection: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    existing_columns = table_columns(connection, table_name)
    for column, definition in columns.items():
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}"
            )
