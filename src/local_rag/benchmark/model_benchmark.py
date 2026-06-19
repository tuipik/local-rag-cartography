"""Benchmark local Ollama models over the existing RAG pipeline."""

from __future__ import annotations

import argparse
import html
import json
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from local_rag.database import add_database_argument, resolve_database
from local_rag.embeddings.build_embeddings import (
    DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL,
)
from local_rag.embeddings.build_embeddings import DEFAULT_OLLAMA_URL
from local_rag.rag.prompt_builder import BuiltPrompt, Source, build_prompt
from local_rag.retrieval.hybrid import (
    DEFAULT_EMBEDDING_WEIGHT,
    DEFAULT_FTS_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    search_hybrid,
)


BENCHMARK_QUERIES = [
    # document lookup
    "Які документи регламентують умовні знаки топографічних карт?",
    "що забов'язаний робити черговий військової частини?",
    "які вимоги до оформлення бойових документів",
    "Які документи потрібно використовувати для оформлення оперативних бойових документів",

    # definitions
    "що таке Об'єктно-орієнтована система?",
    "що таке система координат WGS-84?",
    "що таке система координат MGRS",
    "що таке геопросторова підтримка",
    "що таке геобаза даних",

    # comparison
    "порівняти WGS84 та MGRS",
    "Поясни різницю між WGS-84, UTM та MGRS",
    "Як пов'язані між собою UTM та MGRS",
    "Чим відрізняється магнітне схилення від зближення меридіанів",

    # practical cartography
    "як визначити зближення меридіанів",
    "Для чого необхідно враховувати магнітне схилення",
    "Як визначити магнітне схилення на карті",
    "Для чого використовується система MGRS у військовій топографії",
    "У яких випадках використовується система координат WGS-84",

    # standards and norms
    "які норми часу встановлені для редакційно контрольних перевірок?",
    "які норми часу для різних типів робіт",
    "Які види робіт нормуються документом ПВП 11-(30)294",
    "Які документи визначають норми часу для топографо-геодезичних робіт",

    # doctrine and policy
    "Коротко опиши призначення Доктрини геопросторової підтримки ЗСУ",
    "Які основні завдання геопросторової підтримки Збройних Сил України",
    "Які органи відповідають за геопросторову підтримку",

    # GIS / software
    "Які можливості надає ГІС-портал Збройних Сил України",
    "Які програмні продукти використовуються для створення картографічних матеріалів",
    "Яке програмне забезпечення використовується для підготовки карт до друку",
    "Які можливості надає ArcGIS Pro",

    # training materials
    "Які документи описують створення 3D моделей місцевості",
    "Які документи містять навчальні матеріали по UTM та MGRS",
    "Які документи використовуються для навчання геоінформаційних спеціалістів",

    # multi-document synthesis
    "Які системи координат згадуються у документах корпусу",
    "Які документи стосуються навігаційного забезпечення",
    "Які документи стосуються військової топографії",
    "Які документи містять інформацію про геоінформаційні системи",

    # difficult reasoning
    "які документи суперечать один одному",
    "Чи є різні визначення систем координат у різних документах",
    "Які документи дублюють інформацію один одного",

    # source grounding
    "У якому документі описано систему координат WGS-84",
    "У якому документі описано систему MGRS",
    "У якому документі наведені умовні знаки топографічних карт",

    # no-answer tests
    "Які характеристики має танк Leopard 2A7",
    "Які вимоги до польотів F-16 над територією України",
    "Що таке Kubernetes",
    "Які характеристики ракети Taurus",
    "Що таке ChatGPT",
]

THINKING_MARKERS = [
    "Thinking...",
    "...done thinking",
    "Хм,",
    "мені потрібно",
    "мне нужно",
    "спочатку",
    "сначала",
    "let me",
    "okay,",
]

RUSSIAN_MARKERS = [
    "хорошо",
    "мне нужно",
    "источник",
    "пользователь",
    "сначала",
]

DEFAULT_OUTPUT_DIR = Path("data/benchmark")
DEFAULT_TOP_K = 5
DEFAULT_NUM_PREDICT = 1024
DEFAULT_TIMEOUT_SECONDS = 240


@dataclass(frozen=True)
class SourceRecord:
    index: int
    filename: str
    path: str
    relative_path: str | None
    absolute_path: str | None
    scan_root: str | None
    page_number: int | None
    chunk_id: int
    document_id: int
    document_type: str
    content_category: str | None
    text: str


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    query: str
    answer: str
    sources: list[SourceRecord]
    retrieved_context: str
    final_prompt: str
    retrieval_time_seconds: float
    prompt_build_time_seconds: float
    llm_time_seconds: float
    total_time_seconds: float
    prompt_chars: int
    context_chars: int
    answer_chars: int
    source_count: int
    has_answer: bool
    has_sources: bool
    has_citations: bool
    contains_thinking: bool
    wrong_language_suspected: bool
    empty_answer: bool
    too_short_answer: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama models over the RAG pipeline."
    )
    add_database_argument(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        help="only benchmark these Ollama models",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        help="skip a model; can be passed multiple times",
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
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help=f"maximum generated tokens (default: {DEFAULT_NUM_PREDICT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"benchmark output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        default=True,
        help="include final prompt in reports (default: true)",
    )
    parser.add_argument(
        "--no-include-prompt",
        action="store_false",
        dest="include_prompt",
        help="omit final prompt from markdown/html reports",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        default=True,
        help="include retrieved context in reports (default: true)",
    )
    parser.add_argument(
        "--no-include-context",
        action="store_false",
        dest="include_context",
        help="omit retrieved context from markdown/html reports",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Ollama embedding model for retrieval (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Ollama request timeout seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    return parser.parse_args()


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeError(f"Ollama request timed out after {timeout} seconds") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error


def list_ollama_models(ollama_url: str, *, timeout: int) -> list[str]:
    response = request_json(
        f"{ollama_url.rstrip('/')}/api/tags",
        timeout=timeout,
    )
    models = response.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama /api/tags did not return a models list")
    names: list[str] = []
    for model in models:
        if isinstance(model, dict) and isinstance(model.get("name"), str):
            names.append(model["name"])
    return sorted(names)


def generate_raw_answer(
    *,
    ollama_url: str,
    model: str,
    prompt: BuiltPrompt,
    temperature: float,
    num_predict: int,
    timeout: int,
) -> str:
    response = request_json(
        f"{ollama_url.rstrip('/')}/api/chat",
        method="POST",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
    )
    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama did not return a chat message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama did not return message content")
    return content.strip()


def source_to_record(source: Source) -> SourceRecord:
    return SourceRecord(
        index=source.index,
        filename=source.filename,
        path=source.path,
        relative_path=source.relative_path,
        absolute_path=source.absolute_path,
        scan_root=source.scan_root,
        page_number=source.page_number,
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        document_type=source.document_type,
        content_category=source.content_category,
        text=source.text,
    )


def build_retrieved_context(sources: list[Source]) -> str:
    return "\n\n".join(
        [
            (
                f"[{source.index}]\n"
                f"Document: {source.relative_path or source.filename}\n"
                f"Page: {source.page_number if source.page_number is not None else 'unknown'}\n"
                f"Document ID: {source.document_id}\n"
                f"Type: {source.document_type}\n"
                f"Category: {source.content_category or 'unknown'}\n"
                f"Text:\n{source.text}"
            )
            for source in sources
        ]
    )


def build_final_prompt(prompt: BuiltPrompt) -> str:
    return "\n\n".join(
        [
            "SYSTEM:",
            prompt.system_prompt,
            "USER:",
            prompt.user_prompt,
        ]
    )


def contains_any_marker(text: str, markers: list[str]) -> bool:
    normalized = text.casefold()
    return any(marker.casefold() in normalized for marker in markers)


def quality_flags(answer: str, source_count: int) -> dict[str, bool]:
    stripped = answer.strip()
    return {
        "has_answer": bool(stripped),
        "has_sources": source_count > 0,
        "has_citations": bool(re.search(r"\[\d+\]", answer)),
        "contains_thinking": contains_any_marker(answer, THINKING_MARKERS),
        "wrong_language_suspected": contains_any_marker(answer, RUSSIAN_MARKERS),
        "empty_answer": not bool(stripped),
        "too_short_answer": 0 < len(stripped) < 80,
    }


def benchmark_query(
    *,
    database: Path,
    query: str,
    model: str,
    embedding_model: str,
    ollama_url: str,
    top_k: int,
    pool_size: int,
    num_predict: int,
    temperature: float,
    timeout: int,
) -> BenchmarkResult:
    total_start = time.perf_counter()
    error: str | None = None
    answer = ""

    retrieval_start = time.perf_counter()
    hybrid_results, _, _ = search_hybrid(
        database=database,
        query=query,
        top_k=top_k,
        pool_size=max(pool_size, top_k),
        model=embedding_model,
        ollama_url=ollama_url,
        rebuild_fts=False,
        prefer_reference=True,
    )
    retrieval_time = time.perf_counter() - retrieval_start

    prompt_start = time.perf_counter()
    prompt = build_prompt(
        question=query,
        results=hybrid_results,
        max_sources=top_k,
    )
    retrieved_context = build_retrieved_context(prompt.sources)
    final_prompt = build_final_prompt(prompt)
    prompt_build_time = time.perf_counter() - prompt_start

    llm_start = time.perf_counter()
    try:
        answer = generate_raw_answer(
            ollama_url=ollama_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
        )
    except RuntimeError as exception:
        error = str(exception)
    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - total_start

    flags = quality_flags(answer, len(prompt.sources))
    return BenchmarkResult(
        model=model,
        query=query,
        answer=answer,
        sources=[source_to_record(source) for source in prompt.sources],
        retrieved_context=retrieved_context,
        final_prompt=final_prompt,
        retrieval_time_seconds=round(retrieval_time, 4),
        prompt_build_time_seconds=round(prompt_build_time, 4),
        llm_time_seconds=round(llm_time, 4),
        total_time_seconds=round(total_time, 4),
        prompt_chars=len(final_prompt),
        context_chars=len(retrieved_context),
        answer_chars=len(answer),
        source_count=len(prompt.sources),
        error=error,
        **flags,
    )


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def count_flag(results: list[BenchmarkResult], field_name: str) -> int:
    return sum(1 for result in results if bool(getattr(result, field_name)))


def source_lines(sources: list[SourceRecord]) -> list[str]:
    return [
        (
            f"[{source.index}] {source.relative_path or source.filename}, "
            f"page {source.page_number if source.page_number is not None else 'unknown'}, "
            f"document_id={source.document_id}"
        )
        for source in sources
    ]


def write_jsonl(path: Path, results: list[BenchmarkResult]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def write_model_markdown(
    path: Path,
    *,
    model: str,
    results: list[BenchmarkResult],
    include_prompt: bool,
    include_context: bool,
) -> None:
    total = len(results)
    lines = [
        f"# Model Benchmark: {model}",
        "",
        "## Summary",
        "",
        f"- Queries: {total}",
        f"- Avg total time: {average([result.total_time_seconds for result in results]):.2f}s",
        f"- Avg LLM time: {average([result.llm_time_seconds for result in results]):.2f}s",
        f"- Thinking leaks: {count_flag(results, 'contains_thinking')}/{total}",
        f"- Citation pass: {count_flag(results, 'has_citations')}/{total}",
        f"- Language issues: {count_flag(results, 'wrong_language_suspected')}/{total}",
        "",
        "---",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"## Query {index}",
                "",
                "### Question",
                "",
                result.query,
                "",
                "### Timings",
                "",
                f"- Retrieval: {result.retrieval_time_seconds:.2f}s",
                f"- Prompt build: {result.prompt_build_time_seconds:.2f}s",
                f"- LLM: {result.llm_time_seconds:.2f}s",
                f"- Total: {result.total_time_seconds:.2f}s",
                "",
                "### Flags",
                "",
                f"- has_answer: {str(result.has_answer).lower()}",
                f"- has_sources: {str(result.has_sources).lower()}",
                f"- has_citations: {str(result.has_citations).lower()}",
                f"- contains_thinking: {str(result.contains_thinking).lower()}",
                f"- wrong_language_suspected: {str(result.wrong_language_suspected).lower()}",
                f"- empty_answer: {str(result.empty_answer).lower()}",
                f"- too_short_answer: {str(result.too_short_answer).lower()}",
                "",
                "### Retrieved Sources",
                "",
            ]
        )
        lines.extend(source_lines(result.sources) or ["(none)"])
        lines.append("")
        if include_context:
            lines.extend(
                [
                    "### Retrieved Context",
                    "",
                    "```text",
                    result.retrieved_context,
                    "```",
                    "",
                ]
            )
        if include_prompt:
            lines.extend(
                [
                    "### Final Prompt",
                    "",
                    "```text",
                    result.final_prompt,
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "### Model Answer",
                "",
                "```text",
                result.answer,
                "```",
                "",
            ]
        )
        if result.error:
            lines.extend(["### Error", "", result.error, ""])
        lines.extend(["---", ""])

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def html_details(summary: str, content: str) -> str:
    return (
        "<details>"
        f"<summary>{html.escape(summary)}</summary>"
        f"<pre>{html.escape(content)}</pre>"
        "</details>"
    )


def write_model_html(
    path: Path,
    *,
    model: str,
    results: list[BenchmarkResult],
    include_prompt: bool,
    include_context: bool,
) -> None:
    total = len(results)
    rows = []
    for index, result in enumerate(results, start=1):
        flags = ", ".join(
            [
                f"citations={result.has_citations}",
                f"thinking={result.contains_thinking}",
                f"lang_issue={result.wrong_language_suspected}",
            ]
        )
        sources = "\n".join(source_lines(result.sources))
        details = [html_details("Retrieved Sources", sources)]
        if include_context:
            details.append(html_details("Retrieved Context", result.retrieved_context))
        if include_prompt:
            details.append(html_details("Final Prompt", result.final_prompt))
        details.append(html_details("Full Answer", result.answer))
        if result.error:
            details.append(html_details("Error", result.error))
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(result.query)}</td>"
            f"<td>{result.retrieval_time_seconds:.2f}s</td>"
            f"<td>{result.prompt_build_time_seconds:.2f}s</td>"
            f"<td>{result.llm_time_seconds:.2f}s</td>"
            f"<td>{result.total_time_seconds:.2f}s</td>"
            f"<td>{html.escape(flags)}</td>"
            f"<td>{''.join(details)}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>Model Benchmark: {html.escape(model)}</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.4; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem; vertical-align: top; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>Model Benchmark: {html.escape(model)}</h1>
  <h2>Summary</h2>
  <ul>
    <li>Queries: {total}</li>
    <li>Avg total time: {average([result.total_time_seconds for result in results]):.2f}s</li>
    <li>Avg LLM time: {average([result.llm_time_seconds for result in results]):.2f}s</li>
    <li>Thinking leaks: {count_flag(results, 'contains_thinking')}/{total}</li>
    <li>Citation pass: {count_flag(results, 'has_citations')}/{total}</li>
    <li>Language issues: {count_flag(results, 'wrong_language_suspected')}/{total}</li>
  </ul>
  <table>
    <thead>
      <tr><th>#</th><th>Query</th><th>Retrieval</th><th>Prompt</th><th>LLM</th><th>Total</th><th>Flags</th><th>Details</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def model_notes(results: list[BenchmarkResult]) -> str:
    if count_flag(results, "contains_thinking"):
        return "reject: thinking leaks"
    if count_flag(results, "wrong_language_suspected"):
        return "review: language issues"
    if count_flag(results, "has_citations") == len(results):
        return "candidate"
    return "review"


def write_summary_markdown(
    path: Path,
    *,
    grouped_results: dict[str, list[BenchmarkResult]],
) -> None:
    lines = [
        "# Model Benchmark Summary",
        "",
        "| Model | Avg total time | Avg LLM time | Thinking leaks | Citation pass | Language issues | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for model, results in grouped_results.items():
        total = len(results)
        lines.append(
            "| "
            f"{model} | "
            f"{average([result.total_time_seconds for result in results]):.2f}s | "
            f"{average([result.llm_time_seconds for result in results]):.2f}s | "
            f"{count_flag(results, 'contains_thinking')}/{total} | "
            f"{count_flag(results, 'has_citations')}/{total} | "
            f"{count_flag(results, 'wrong_language_suspected')}/{total} | "
            f"{model_notes(results)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_html(
    path: Path,
    *,
    grouped_results: dict[str, list[BenchmarkResult]],
) -> None:
    rows = []
    for model, results in grouped_results.items():
        safe_name = safe_model_name(model)
        total = len(results)
        rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{average([result.total_time_seconds for result in results]):.2f}s</td>"
            f"<td>{average([result.llm_time_seconds for result in results]):.2f}s</td>"
            f"<td>{count_flag(results, 'contains_thinking')}/{total}</td>"
            f"<td>{count_flag(results, 'wrong_language_suspected')}/{total}</td>"
            f"<td>{count_flag(results, 'has_citations')}/{total}</td>"
            f"<td>{html.escape(model_notes(results))}</td>"
            f"<td><a href=\"html/{html.escape(safe_name)}.html\">report</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>Model Benchmark Summary</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.4; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem; }}
  </style>
</head>
<body>
  <h1>Model Benchmark Summary</h1>
  <table>
    <thead>
      <tr><th>Model</th><th>Avg total time</th><th>Avg LLM time</th><th>Thinking leaks</th><th>Language issues</th><th>Citation pass</th><th>Notes</th><th>HTML report</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_comparison_html(
    path: Path,
    *,
    grouped_results: dict[str, list[BenchmarkResult]],
) -> None:
    model_rows = []
    for model, results in grouped_results.items():
        safe_name = safe_model_name(model)
        total = len(results)
        model_rows.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{average([result.total_time_seconds for result in results]):.2f}s</td>"
            f"<td>{average([result.llm_time_seconds for result in results]):.2f}s</td>"
            f"<td>{count_flag(results, 'contains_thinking')}/{total}</td>"
            f"<td>{count_flag(results, 'wrong_language_suspected')}/{total}</td>"
            f"<td>{count_flag(results, 'has_citations')}/{total}</td>"
            f"<td>{count_flag(results, 'empty_answer')}/{total}</td>"
            f"<td>{count_flag(results, 'too_short_answer')}/{total}</td>"
            f"<td>{html.escape(model_notes(results))}</td>"
            f"<td><a href=\"html/{html.escape(safe_name)}.html\">model report</a></td>"
            "</tr>"
        )

    query_sections = []
    for query_index, query in enumerate(BENCHMARK_QUERIES, start=1):
        rows = []
        for model, results in grouped_results.items():
            if query_index > len(results):
                continue
            result = results[query_index - 1]
            flags = []
            if result.has_citations:
                flags.append("citations")
            if result.contains_thinking:
                flags.append("thinking leak")
            if result.wrong_language_suspected:
                flags.append("language issue")
            if result.empty_answer:
                flags.append("empty")
            if result.too_short_answer:
                flags.append("too short")
            if result.error:
                flags.append("error")
            flag_text = ", ".join(flags) if flags else "none"
            details = [
                html_details("Answer", result.answer),
                html_details("Sources", "\n".join(source_lines(result.sources))),
            ]
            if result.error:
                details.append(html_details("Error", result.error))
            rows.append(
                "<tr>"
                f"<td>{html.escape(model)}</td>"
                f"<td>{result.total_time_seconds:.2f}s</td>"
                f"<td>{result.llm_time_seconds:.2f}s</td>"
                f"<td>{result.answer_chars}</td>"
                f"<td>{result.source_count}</td>"
                f"<td>{html.escape(flag_text)}</td>"
                f"<td>{''.join(details)}</td>"
                "</tr>"
            )
        query_sections.append(
            f"""
  <section>
    <h2>Query {query_index}</h2>
    <p><strong>{html.escape(query)}</strong></p>
    <table>
      <thead>
        <tr><th>Model</th><th>Total</th><th>LLM</th><th>Answer chars</th><th>Sources</th><th>Flags</th><th>Details</th></tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
"""
        )

    document = f"""<!doctype html>
<html lang="uk">
<head>
  <meta charset="utf-8">
  <title>Model Benchmark Comparison</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.4; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem; vertical-align: top; }}
    th {{ background: #f2f2f2; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 1rem; }}
    section {{ margin-top: 2rem; }}
  </style>
</head>
<body>
  <h1>Model Benchmark Comparison</h1>
  <p>Formal benchmark comparison across all models included in this run. Content correctness is intended for manual review.</p>

  <h2>Model Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Avg total time</th>
        <th>Avg LLM time</th>
        <th>Thinking leaks</th>
        <th>Language issues</th>
        <th>Citation pass</th>
        <th>Empty answers</th>
        <th>Too short</th>
        <th>Notes</th>
        <th>Report</th>
      </tr>
    </thead>
    <tbody>{''.join(model_rows)}</tbody>
  </table>

  <h2>Per-query Comparison</h2>
  {''.join(query_sections)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def select_models(
    *,
    available_models: list[str],
    requested_models: list[str] | None,
    skip_models: list[str],
) -> list[str]:
    selected = requested_models if requested_models else available_models
    skipped = set(skip_models)
    return [model for model in selected if model not in skipped]


def prepare_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    models_dir = output_dir / "models"
    html_dir = output_dir / "html"
    models_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, html_dir


def ensure_existing_fts_table(database: Path) -> None:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'chunks_fts'
            """
        ).fetchone()
    if row is None:
        raise RuntimeError(
            "chunks_fts table does not exist. Run scripts/search_hybrid.py once "
            "outside benchmark to create the retrieval index."
        )


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    database = resolve_database(args.database)
    ensure_existing_fts_table(database)
    output_dir = args.output_dir.expanduser().resolve()
    models_dir, html_dir = prepare_output_dirs(output_dir)
    available_models = list_ollama_models(args.ollama_url, timeout=args.timeout)
    models = select_models(
        available_models=available_models,
        requested_models=args.models,
        skip_models=args.skip_model,
    )
    if not models:
        raise RuntimeError("No models selected for benchmark")

    print(f"Available models: {', '.join(available_models)}")
    print(f"Benchmark models: {', '.join(models)}")

    all_results: list[BenchmarkResult] = []
    grouped_results: dict[str, list[BenchmarkResult]] = {}
    for model in models:
        print()
        print(f"Model: {model}")
        model_results: list[BenchmarkResult] = []
        for index, query in enumerate(BENCHMARK_QUERIES, start=1):
            print(f"  Query {index}/{len(BENCHMARK_QUERIES)}: {query}")
            result = benchmark_query(
                database=database,
                query=query,
                model=model,
                embedding_model=args.embedding_model,
                ollama_url=args.ollama_url,
                top_k=args.top_k,
                pool_size=args.pool_size,
                num_predict=args.num_predict,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            print(f"    total={result.total_time_seconds:.2f}s llm={result.llm_time_seconds:.2f}s citations={result.has_citations}")
            model_results.append(result)
            all_results.append(result)

        grouped_results[model] = model_results
        safe_name = safe_model_name(model)
        write_model_markdown(
            models_dir / f"{safe_name}.md",
            model=model,
            results=model_results,
            include_prompt=args.include_prompt,
            include_context=args.include_context,
        )
        write_model_html(
            html_dir / f"{safe_name}.html",
            model=model,
            results=model_results,
            include_prompt=args.include_prompt,
            include_context=args.include_context,
        )

    write_jsonl(output_dir / "model_benchmark_results.jsonl", all_results)
    write_summary_markdown(
        output_dir / "model_benchmark_summary.md",
        grouped_results=grouped_results,
    )
    write_summary_html(
        output_dir / "model_benchmark_summary.html",
        grouped_results=grouped_results,
    )
    write_comparison_html(
        output_dir / "model_benchmark_comparison.html",
        grouped_results=grouped_results,
    )
    return all_results


def main() -> int:
    args = parse_args()
    try:
        results = run_benchmark(args)
    except (FileNotFoundError, RuntimeError) as error:
        print(error)
        return 1
    print()
    print(f"Benchmark complete. Results: {len(results)}")
    print(f"Output directory: {args.output_dir.expanduser().resolve()}")
    return 0
