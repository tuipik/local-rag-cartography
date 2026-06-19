"""Human-readable source display helpers for RAG citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Protocol


PDF_EXTENSIONS = {".pdf"}
FRAGMENT_EXTENSIONS = {".doc", ".docx", ".txt"}
DEFAULT_PREVIEW_CHARS = 220


class SourceLike(Protocol):
    filename: str
    relative_path: str | None
    page_number: int | None
    chunk_index: int
    start_char: int | None
    end_char: int | None
    text: str


@dataclass(frozen=True)
class FormattedSource:
    display_path: str
    location: str
    preview: str


def compact_preview(text: str, *, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def source_extension(source: SourceLike) -> str:
    display_path = source.relative_path or source.filename
    return PurePath(display_path).suffix.casefold()


def display_path(source: SourceLike) -> str:
    return source.relative_path or source.filename


def char_range(source: SourceLike) -> str | None:
    if source.start_char is None or source.end_char is None:
        return None
    return f"chars {source.start_char}-{source.end_char}"


def source_location(source: SourceLike) -> str:
    extension = source_extension(source)
    if extension in PDF_EXTENSIONS and source.page_number is not None:
        return f"page {source.page_number}"

    range_text = char_range(source)
    fragment = f"fragment {source.chunk_index}"
    if range_text:
        return f"{fragment}, {range_text}"
    return fragment


def format_source(source: SourceLike) -> FormattedSource:
    return FormattedSource(
        display_path=display_path(source),
        location=source_location(source),
        preview=compact_preview(source.text),
    )
