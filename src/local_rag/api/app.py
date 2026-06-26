"""FastAPI application for the Local RAG backend MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from local_rag.api.schemas import (
    AskMeta,
    AskRequest,
    AskResponse,
    DocumentMetadataResponse,
    HealthResponse,
    SourceResponse,
)
from local_rag.database import DEFAULT_DATABASE, connect_database, resolve_database
from local_rag.embeddings.build_embeddings import DEFAULT_OLLAMA_URL
from local_rag.rag.answer_generator import generate_answer
from local_rag.rag.citation_parser import extract_citation_ids
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


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    path: Path
    relative_path: str
    name: str
    extension: str


def source_type(path: str) -> str:
    extension = PurePath(path).suffix.casefold().lstrip(".")
    return extension or "unknown"


def source_response(source: object) -> SourceResponse:
    formatted_source = format_source(source)
    return SourceResponse(
        citation_id=source.index,
        document_id=source.document_id,
        relative_path=formatted_source.display_path,
        location=formatted_source.location,
        preview=formatted_source.preview,
        page_number=source.page_number,
        source_type=source_type(formatted_source.display_path),
        view_url=f"/documents/{source.document_id}/view",
        download_url=f"/documents/{source.document_id}/download",
    )


def get_document_record(document_id: int) -> DocumentRecord:
    database = resolve_database(DEFAULT_DATABASE)
    with connect_database(database) as connection:
        row = connection.execute(
            """
            SELECT
                id,
                path,
                relative_path,
                name,
                extension
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(row["path"]).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Document file not found")

    relative_path = row["relative_path"] or row["name"]
    extension = (row["extension"] or PurePath(relative_path).suffix).lstrip(".").lower()
    return DocumentRecord(
        id=row["id"],
        path=file_path,
        relative_path=relative_path,
        name=row["name"],
        extension=extension,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/documents/{document_id}/metadata", response_model=DocumentMetadataResponse)
def document_metadata(document_id: int) -> DocumentMetadataResponse:
    document = get_document_record(document_id)
    return DocumentMetadataResponse(
        document_id=document.id,
        name=document.name,
        relative_path=document.relative_path,
        extension=document.extension,
        source_type=document.extension or source_type(document.relative_path),
        view_url=f"/documents/{document.id}/view",
        download_url=f"/documents/{document.id}/download",
    )


@app.get("/documents/{document_id}/download")
def download_document(document_id: int) -> FileResponse:
    document = get_document_record(document_id)
    return FileResponse(
        path=document.path,
        filename=document.name,
        media_type="application/octet-stream",
        content_disposition_type="attachment",
    )


@app.get("/documents/{document_id}/view")
def view_document(document_id: int) -> FileResponse:
    document = get_document_record(document_id)
    return FileResponse(
        path=document.path,
        filename=document.name,
        content_disposition_type="inline",
    )


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

    retrieved_sources = [source_response(source) for source in result.sources]
    used_ids = extract_citation_ids(result.answer)
    used_sources = [
        source for source in retrieved_sources if source.citation_id in used_ids
    ]
    return AskResponse(
        answer=result.answer,
        sources=retrieved_sources,
        used_sources=used_sources,
        retrieved_sources=retrieved_sources,
        meta=AskMeta(
            llm_model=request.llm_model,
            embedding_model=request.embedding_model,
            top_k=request.top_k,
            source_count=len(retrieved_sources),
            retrieved_source_count=len(retrieved_sources),
        ),
    )
