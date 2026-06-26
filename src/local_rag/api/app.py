"""FastAPI application for the Local RAG backend MVP."""

from __future__ import annotations

from pathlib import PurePath

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from local_rag.api.schemas import (
    AskMeta,
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceResponse,
)
from local_rag.database import DEFAULT_DATABASE, resolve_database
from local_rag.embeddings.build_embeddings import DEFAULT_OLLAMA_URL
from local_rag.rag.answer_generator import generate_answer
from local_rag.rag.source_formatter import format_source


app = FastAPI(title="Local RAG Cartography API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def source_type(path: str) -> str:
    extension = PurePath(path).suffix.casefold().lstrip(".")
    return extension or "unknown"


def source_response(source: object) -> SourceResponse:
    formatted_source = format_source(source)
    return SourceResponse(
        document_id=source.document_id,
        relative_path=formatted_source.display_path,
        location=formatted_source.location,
        preview=formatted_source.preview,
        page_number=source.page_number,
        source_type=source_type(formatted_source.display_path),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = generate_answer(
            question=request.question,
            database=resolve_database(DEFAULT_DATABASE),
            llm_model=request.llm_model,
            embedding_model=request.embedding_model,
            ollama_url=DEFAULT_OLLAMA_URL,
            top_k=request.top_k,
            pool_size=max(request.pool_size, request.top_k),
            max_sources=request.max_sources,
            rebuild_fts=request.rebuild_fts,
            prefer_reference=request.prefer_reference,
            rrf_k=request.rrf_k,
            fts_weight=request.fts_weight,
            embedding_weight=request.embedding_weight,
            temperature=request.temperature,
            num_predict=request.num_predict,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    sources = [source_response(source) for source in result.sources]
    return AskResponse(
        answer=result.answer,
        sources=sources,
        meta=AskMeta(
            llm_model=request.llm_model,
            embedding_model=request.embedding_model,
            top_k=request.top_k,
            source_count=len(sources),
        ),
    )
