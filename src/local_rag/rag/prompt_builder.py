"""Prompt construction for grounded local RAG answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_rag.retrieval.hybrid import HybridResult


DEFAULT_CONTEXT_CHARS = 1600
DOCUMENT_LOOKUP_MARKERS = (
    "які документи",
    "який документ",
    "який наказ",
    "що регламентує",
    "регламентують",
    "визначають",
)


@dataclass(frozen=True)
class Source:
    index: int
    filename: str
    path: str
    page_number: int | None
    chunk_id: int
    document_id: int
    text: str


@dataclass(frozen=True)
class BuiltPrompt:
    system_prompt: str
    user_prompt: str
    sources: list[Source]


def compact_text(text: str, limit: int = DEFAULT_CONTEXT_CHARS) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def infer_query_type(question: str) -> str:
    normalized = question.casefold()
    if any(marker in normalized for marker in DOCUMENT_LOOKUP_MARKERS):
        return "document_lookup"
    return "conceptual"


def build_sources(results: list[HybridResult], *, max_sources: int) -> list[Source]:
    sources: list[Source] = []
    seen_documents: set[int] = set()
    for result in results:
        if result.document_id in seen_documents:
            continue
        seen_documents.add(result.document_id)
        sources.append(
            Source(
                index=len(sources) + 1,
                filename=result.filename,
                path=result.path,
                page_number=result.page_number,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=compact_text(result.text),
            )
        )
        if len(sources) >= max_sources:
            break
    return sources


def build_prompt(
    *,
    question: str,
    results: list[HybridResult],
    max_sources: int,
) -> BuiltPrompt:
    sources = build_sources(results, max_sources=max_sources)
    query_type = infer_query_type(question)
    system_prompt = "\n".join(
        [
            "Ти локальний RAG-помічник для роботи з документами.",
            "Відповідай українською мовою.",
            "Відповідай тільки на основі наданих джерел.",
            'Якщо в джерелах недостатньо інформації — скажи: "У наданих документах недостатньо інформації".',
            "Не вигадуй накази, номери, дати, сторінки або назви документів.",
            "У відповіді посилайся на джерела у форматі [1], [2].",
            "Кожне твердження про номер, дату або назву документа має мати посилання на те саме джерело.",
            "Якщо питання просить назвати документи, перелічуй тільки релевантні документи з джерел і не переказуй зміст норм.",
            "Не показуй процес міркування.",
        ]
    )

    if not sources:
        context = "Джерела не знайдено."
    else:
        context = "\n\n".join(
            [
                (
                    f"[{source.index}] {source.filename}, "
                    f"page {source.page_number if source.page_number is not None else 'unknown'}, "
                    f"chunk {source.chunk_id}\n{source.text}"
                )
                for source in sources
            ]
        )

    if query_type == "document_lookup":
        task_instruction = "\n".join(
            [
                "Це document lookup запит.",
                "Відповідай рівно про те, які документи знайдено.",
                "Обов'язковий формат відповіді:",
                "Релевантні документи:",
                "- Назва документа [номер джерела].",
                "Не описуй зміст умовних знаків, правил або стандартів.",
                "Не згадуй накази, номери чи дати, якщо питання прямо не просить їх назвати.",
            ]
        )
    else:
        task_instruction = (
            "Сформуй коротку, grounded відповідь із посиланнями на джерела. "
            "Не додавай фактів поза наведеними джерелами."
        )

    user_prompt = "\n\n".join(
        [
            f"Питання: {question}",
            f"Тип запиту: {query_type}",
            "Джерела:",
            context,
            task_instruction,
        ]
    )
    return BuiltPrompt(system_prompt=system_prompt, user_prompt=user_prompt, sources=sources)
