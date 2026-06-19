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

Stage 11. Simple UI MVP.

---

## Поточний фокус

Simple UI MVP:

* Надати базовий React/Vite інтерфейс для роботи з RAG.
* Дозволити користувачу ввести питання, надіслати його у FastAPI і побачити відповідь.
* Показувати структуровані sources з `relative_path`, `location` і `preview`.
* Auth, streaming, chat history, document opening і admin UI не входять у поточну ітерацію.

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

---

## Current API status

Implemented:

* `GET /health`
* `POST /ask`

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
