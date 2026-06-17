"""End-to-end local RAG answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_rag.embeddings.build_embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from local_rag.embeddings.build_embeddings import DEFAULT_OLLAMA_URL
from local_rag.rag.ollama_client import DEFAULT_LLM_MODEL, ChatMessage, chat
from local_rag.rag.prompt_builder import Source, build_prompt
from local_rag.retrieval.hybrid import (
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_FTS_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    search_hybrid,
)


INSUFFICIENT_INFORMATION = "У наданих документах недостатньо інформації"


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[Source]


def generate_answer(
    *,
    question: str,
    database: Path,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    top_k: int = DEFAULT_TOP_K,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_sources: int = 5,
    rebuild_fts: bool = True,
    prefer_reference: bool = True,
    rrf_k: float = DEFAULT_RRF_K,
    fts_weight: float = DEFAULT_FTS_WEIGHT,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    temperature: float = 0.0,
    num_predict: int = 1024,
) -> AnswerResult:
    hybrid_results, _, _ = search_hybrid(
        database=database,
        query=question,
        top_k=top_k,
        pool_size=max(pool_size, top_k),
        model=embedding_model,
        ollama_url=ollama_url,
        rebuild_fts=rebuild_fts,
        prefer_reference=prefer_reference,
        rrf_k=rrf_k,
        fts_weight=fts_weight,
        embedding_weight=embedding_weight,
    )
    built_prompt = build_prompt(
        question=question,
        results=hybrid_results,
        max_sources=max_sources,
    )
    if not built_prompt.sources:
        return AnswerResult(answer=INSUFFICIENT_INFORMATION, sources=[])

    answer = chat(
        messages=[
            ChatMessage(role="system", content=built_prompt.system_prompt),
            ChatMessage(role="user", content=built_prompt.user_prompt),
        ],
        model=llm_model,
        ollama_url=ollama_url,
        temperature=temperature,
        num_predict=num_predict,
    )
    if not answer:
        answer = INSUFFICIENT_INFORMATION
    return AnswerResult(answer=answer, sources=built_prompt.sources)
