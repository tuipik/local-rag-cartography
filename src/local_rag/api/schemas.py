"""Pydantic schemas for the Local RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from local_rag.embeddings.build_embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from local_rag.rag.ollama_client import DEFAULT_LLM_MODEL
from local_rag.retrieval.hybrid import (
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_FTS_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
)


class HealthResponse(BaseModel):
    status: str = "ok"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50)
    pool_size: int = Field(DEFAULT_POOL_SIZE, ge=1, le=200)
    max_sources: int = Field(5, ge=1, le=20)
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    num_predict: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    rebuild_fts: bool = False
    prefer_reference: bool = True
    rrf_k: float = Field(DEFAULT_RRF_K, gt=0.0)
    fts_weight: float = Field(DEFAULT_FTS_WEIGHT, ge=0.0)
    embedding_weight: float = Field(DEFAULT_EMBEDDING_WEIGHT, ge=0.0)


class SourceResponse(BaseModel):
    document_id: int
    relative_path: str
    location: str
    preview: str
    page_number: int | None
    source_type: str


class AskMeta(BaseModel):
    llm_model: str
    embedding_model: str
    top_k: int
    source_count: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    meta: AskMeta
