"""Build and store chunk embeddings with local Ollama."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import sqlite3
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_rag.database import (
    add_database_argument,
    connect_database,
    resolve_database,
)


DEFAULT_MODEL = "bge-m3"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
PROGRESS_EVERY = 25
DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local Ollama embeddings for chunks and store them in SQLite."
    )
    add_database_argument(parser)
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
        "--limit",
        type=int,
        help="process at most N missing chunks, useful for smoke tests",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"number of chunks per Ollama /api/embed request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help=f"print progress every N chunks (default: {PROGRESS_EVERY})",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id),
            UNIQUE(chunk_id, model)
        )
        """
    )
    migrate_chunk_embeddings_unique_constraint(connection)
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
        ON chunk_embeddings(model)
        """
    )


def migrate_chunk_embeddings_unique_constraint(connection: sqlite3.Connection) -> None:
    indexes = connection.execute("PRAGMA index_list(chunk_embeddings)").fetchall()
    for index in indexes:
        index_name = index["name"]
        is_unique = bool(index["unique"])
        if not is_unique:
            continue
        indexed_columns = [
            row["name"]
            for row in connection.execute(f"PRAGMA index_info({index_name})").fetchall()
        ]
        if indexed_columns == ["chunk_id"]:
            rebuild_chunk_embeddings_table(connection)
            return


def rebuild_chunk_embeddings_table(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE chunk_embeddings RENAME TO chunk_embeddings_old")
    connection.execute(
        """
        CREATE TABLE chunk_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id),
            UNIQUE(chunk_id, model)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO chunk_embeddings (
            id,
            chunk_id,
            model,
            embedding,
            dimension,
            created_at
        )
        SELECT
            id,
            chunk_id,
            model,
            embedding,
            dimension,
            created_at
        FROM chunk_embeddings_old
        """
    )
    connection.execute("DROP TABLE chunk_embeddings_old")


def embedding_to_blob(embedding: list[float]) -> bytes:
    return struct.pack(f"<{len(embedding)}f", *embedding)


def blob_to_embedding(blob: bytes, dimension: int) -> list[float]:
    return list(struct.unpack(f"<{dimension}f", blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def request_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except http.client.RemoteDisconnected as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error


def embed_text(text: str, model: str, ollama_url: str) -> list[float]:
    return embed_texts([text], model=model, ollama_url=ollama_url)[0]


def embed_texts(texts: list[str], model: str, ollama_url: str) -> list[list[float]]:
    base_url = ollama_url.rstrip("/")
    try:
        response = request_json(
            f"{base_url}/api/embed",
            {"model": model, "input": texts},
        )
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(texts):
            return [
                [float(value) for value in embedding]
                for embedding in embeddings
                if isinstance(embedding, list)
            ]
    except RuntimeError:
        # Older Ollama versions used /api/embeddings. Try that before failing.
        pass

    embeddings: list[list[float]] = []
    for text in texts:
        response = request_json(
            f"{base_url}/api/embeddings",
            {"model": model, "prompt": text},
        )
        embedding = response.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama did not return an embedding")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def fetch_missing_chunks(
    connection: sqlite3.Connection,
    model: str,
    limit: int | None,
) -> list[ChunkRecord]:
    limit_clause = ""
    params: list[object] = [model]
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(limit)

    rows = connection.execute(
        f"""
        SELECT c.id, c.chunk_text
        FROM chunks c
        LEFT JOIN chunk_embeddings e
          ON e.chunk_id = c.id
         AND e.model = ?
        WHERE e.id IS NULL
        ORDER BY c.id
        {limit_clause}
        """,
        params,
    ).fetchall()
    return [
        ChunkRecord(id=row["id"], text=row["chunk_text"])
        for row in rows
    ]


def count_existing_embeddings(connection: sqlite3.Connection, model: str) -> int:
    row = connection.execute(
        "SELECT count(*) AS count FROM chunk_embeddings WHERE model = ?",
        (model,),
    ).fetchone()
    return int(row["count"])


def existing_embedding_dimension(connection: sqlite3.Connection, model: str) -> int:
    row = connection.execute(
        """
        SELECT dimension
        FROM chunk_embeddings
        WHERE model = ?
        LIMIT 1
        """,
        (model,),
    ).fetchone()
    return int(row["dimension"]) if row else 0


def save_embedding(
    connection: sqlite3.Connection,
    chunk_id: int,
    model: str,
    embedding: list[float],
) -> None:
    connection.execute(
        """
        INSERT INTO chunk_embeddings (
            chunk_id,
            model,
            embedding,
            dimension,
            created_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id, model) DO UPDATE SET
            embedding = excluded.embedding,
            dimension = excluded.dimension,
            created_at = excluded.created_at
        """,
        (
            chunk_id,
            model,
            embedding_to_blob(embedding),
            len(embedding),
            utc_now(),
        ),
    )


def build_embeddings(
    database: Path,
    model: str,
    ollama_url: str,
    limit: int | None = None,
    progress_every: int = PROGRESS_EVERY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int | float]:
    if batch_size < 1:
        raise RuntimeError("batch_size must be at least 1")

    started_at = time.monotonic()
    with connect_database(database) as connection:
        initialize_database(connection)
        existing_before = count_existing_embeddings(connection, model)
        dimension = existing_embedding_dimension(connection, model)
        chunks = fetch_missing_chunks(connection, model, limit)

        created = 0
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            embeddings = embed_texts(
                [chunk.text for chunk in batch],
                model=model,
                ollama_url=ollama_url,
            )
            if len(embeddings) != len(batch):
                raise RuntimeError("Ollama returned an unexpected embedding count")

            for chunk, embedding in zip(batch, embeddings):
                if not embedding:
                    raise RuntimeError(f"empty embedding for chunk_id={chunk.id}")
                dimension = len(embedding)
                save_embedding(connection, chunk.id, model, embedding)
                created += 1

            if progress_every > 0 and (
                created % progress_every == 0 or created == len(chunks)
            ):
                connection.commit()
                elapsed = time.monotonic() - started_at
                print(
                    f"embedded {created}/{len(chunks)} chunks "
                    f"({created / elapsed:.2f} chunks/sec)"
                )

        existing_after = count_existing_embeddings(connection, model)

    elapsed = time.monotonic() - started_at
    return {
        "existing_before": existing_before,
        "created": created,
        "existing_after": existing_after,
        "skipped_existing": existing_before,
        "dimension": dimension,
        "elapsed_seconds": round(elapsed, 1),
    }


def print_statistics(stats: dict[str, int | float], model: str) -> None:
    print("\nEmbedding build statistics")
    print(f"model: {model}")
    print(f"existing_before: {stats['existing_before']}")
    print(f"created: {stats['created']}")
    print(f"skipped_existing: {stats['skipped_existing']}")
    print(f"existing_after: {stats['existing_after']}")
    print(f"dimension: {stats['dimension']}")
    print(f"elapsed_seconds: {stats['elapsed_seconds']}")


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)
    try:
        stats = build_embeddings(
            database=database,
            model=args.model,
            ollama_url=args.ollama_url,
            limit=args.limit,
            progress_every=args.progress_every,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(error)
        return 1

    print_statistics(stats, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
