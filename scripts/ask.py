#!/usr/bin/env python3
"""Ask a question and generate a grounded answer from local documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from local_rag.database import add_database_argument, resolve_database  # noqa: E402
from local_rag.embeddings.build_embeddings import (  # noqa: E402
    DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL,
)
from local_rag.embeddings.build_embeddings import DEFAULT_OLLAMA_URL  # noqa: E402
from local_rag.rag.answer_generator import generate_answer  # noqa: E402
from local_rag.rag.ollama_client import DEFAULT_LLM_MODEL  # noqa: E402
from local_rag.rag.source_formatter import format_source  # noqa: E402
from local_rag.retrieval.hybrid import (  # noqa: E402
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_FTS_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a grounded local RAG answer with hybrid retrieval and Ollama."
    )
    add_database_argument(parser)
    parser.add_argument("question", help="user question")
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"Ollama LLM model for answer generation (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Ollama embedding model for hybrid retrieval (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"hybrid retrieval top-k (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help=f"candidate pool size per retrieval method (default: {DEFAULT_POOL_SIZE})",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
        help="maximum number of retrieved sources passed to the LLM (default: 5)",
    )
    parser.add_argument(
        "--no-prefer-reference",
        action="store_true",
        help="disable the default FTS reference boost before hybrid fusion",
    )
    parser.add_argument(
        "--no-rebuild-fts",
        action="store_true",
        help="use existing chunks_fts table without rebuilding it first",
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
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=1024,
        help="maximum number of generated tokens (default: 512)",
    )
    return parser.parse_args()


def print_answer(answer: str) -> None:
    print("Answer:")
    print(answer.strip())


def print_sources(sources: list[object]) -> None:
    print()
    print("Sources:")
    if not sources:
        print("(none)")
        return
    for source in sources:
        formatted_source = format_source(source)
        print(f"[{source.index}] {formatted_source.display_path} — {formatted_source.location}")
        if formatted_source.preview:
            print(f"    preview: \"{formatted_source.preview}\"")


def main() -> int:
    args = parse_args()
    database = resolve_database(args.database)
    try:
        result = generate_answer(
            question=args.question,
            database=database,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            ollama_url=args.ollama_url,
            top_k=args.top_k,
            pool_size=args.pool_size,
            max_sources=args.max_sources,
            rebuild_fts=not args.no_rebuild_fts,
            prefer_reference=not args.no_prefer_reference,
            rrf_k=args.rrf_k,
            fts_weight=args.fts_weight,
            embedding_weight=args.embedding_weight,
            temperature=args.temperature,
            num_predict=args.num_predict,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(error)
        return 1

    print_answer(result.answer)
    print_sources(result.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
