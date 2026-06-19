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

---

## Поточний етап

Stage 9. Source Traceability / Citation Improvements.

---

## Поточний фокус

Source traceability MVP:

* PDF джерела показують `relative_path` і page number.
* DOC/DOCX/TXT джерела показують `relative_path`, fragment/chunk location, character range і preview замість оманливого `page 1`.
* OCR, Word layout parsing і UI file opening не входять у поточну ітерацію.

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
