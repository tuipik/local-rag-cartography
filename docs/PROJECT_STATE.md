# Local RAG Cartography

## Мета

Побудувати локальну RAG-систему для роботи з документами з картографії, геодезії, топографії та суміжних дисциплін.

Система повинна:

* знаходити релевантні документи;
* відповідати на питання на основі документів;
* надавати посилання на джерела;
* показувати сторінки та фрагменти;
* працювати локально;
* використовувати Ollama.

---

## Типи документів

* PDF
* DOC
* DOCX
* TXT

Майбутні:

* JPG
* PNG
* CDR
* PPT

---

## Поточний обсяг

Тестова вибірка:

* ~150 файлів
* ~2 GB

---

## Обраний стек

Embeddings:

* BGE-M3

Vector storage:

* SQLite `chunk_embeddings` for current MVP
* Qdrant deferred until corpus size or retrieval requirements justify it

LLM:

* Primary: `gemma4:e2b`
* Fallback: `gemma3:4b`
* Reference baseline: `qwen3:8b`

Backend:

* Python
* FastAPI

---

## Поточний етап

Stage 12. Modern UI / UX.

---

## Поточний фокус

Modern UI / UX:

* Покращити usability після завершення Core RAG MVP.
* Оновити дизайн, layout, answer viewer і source display.
* Підготувати UX до майбутнього deployment.
* Production readiness переноситься на Stage 13.

---

## Current status

Completed:

* Stage 11.5. Evidence-based Sources and Document Links.

Next milestone:

Stage 12. Modern UI / UX.

Reason:

Core RAG stack is considered stable. The next priority is improving usability before deployment.

---

## Виконано

* Етап 0. Проєктування.
* Етап 1. Інвентаризація документів.
* Етап 2. Витяг тексту.
* Етап 3. Метадані.
* Етап 3.5. Аналіз корпусу документів.
* Етап 4. Chunking.
* Етап 5. Retrieval без LLM.
* Етап 6. Embeddings.
* Етап 6.5. Оцінка FTS vs Embeddings.
* Етап 7. Hybrid Search.
* Етап 7.5. Retrieval error analysis.
* Stage 8.5. Model benchmark.
* Stage 8.6. Final model selection.
* Stage 9. Source Traceability / Citation Improvements.
* Stage 10. FastAPI Backend MVP.
* Stage 11. Simple UI MVP.
* Stage 11.5. Evidence-based Sources and Document Links.

---

## Current API status

Implemented:

* `GET /health`
* `POST /ask`
* `GET /documents/{document_id}/metadata`
* `GET /documents/{document_id}/download`
* `GET /documents/{document_id}/view`

---

## Поточна модель MVP

Primary LLM: `gemma4:e2b`  
Fallback LLM: `gemma3:4b`  
Reference baseline: `qwen3:8b`

---

## Прийняті рішення

### 2026-06-15

* Використовувати BGE-M3.
* Не використовувати Agno на першому етапі.
* Спочатку реалізувати retrieval без LLM.
* Використовувати Git як джерело правди для проєкту.
