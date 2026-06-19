# ROADMAP

## Етап 0. Проєктування

* [x] Визначити цілі проєкту
* [x] Вибрати підхід до роботи
* [x] Створити репозиторій
* [x] Створити базову документацію

---

## Етап 1. Аудит документів

Мета:

Зрозуміти склад бібліотеки документів.

Результат:

* [x] маніфест документів;
* [x] типи документів;
* [x] категорії документів;
* [x] статистика по документах;
* [x] оцінка якості PDF.

---

## Етап 2. Витяг тексту

Мета:

Отримати текст із документів.

Результат:

* [x] парсер PDF;
* [x] парсер DOCX;
* [x] парсер DOC;
* [x] парсер TXT;
* [x] збереження тексту та метаданих.

---

## Етап 3. Метадані

Мета:

Автоматично витягати інформацію про документи.

Результат:

* [ ] назва;
* [x] дата;
* [x] номер документа;
* [x] тип документа;
* [x] категорія документа.

---

## Етап 3.5. Аналіз корпусу документів

Мета:

Дослідити структуру витягнутого тексту перед проєктуванням chunking.

Причина:

Різні типи документів потребують різних стратегій chunking. Для нормативних документів важливі пункти та підпункти, для навчальних матеріалів — розділи, для довідників — сторінки або таблиці.

Результат:

* [x] статистика довжини сторінок;
* [x] список документів без тексту;
* [x] список OCR-кандидатів;
* [x] виявлення структурних маркерів;
* [x] оцінка, чи підходить page-based chunking;
* [x] попередня рекомендація щодо chunking strategy.

Критерій приймання:

* [x] створено `scripts/analyze_corpus.py`;
* [x] скрипт читає SQLite-каталог;
* [x] скрипт формує зрозумілий звіт;
* [x] на основі звіту прийнято рішення для Етапу 4.

---

## Етап 4. Chunking

Мета:

Створити chunks для майбутнього retrieval із привʼязкою до документа, сторінки, типу chunking strategy та позиції в документі.

Результат:

* [x] таблиця `chunks`;
* [x] скрипт `scripts/build_chunks.py`;
* [x] rule-based chunking baseline;
* [x] виключення OCR-кандидатів;
* [x] обробка дуже довгих сторінок;
* [x] збереження звʼязку chunk → document → page.

Критерій приймання:

* [x] chunks створюються для документів із текстом;
* [x] OCR-кандидати не потрапляють у chunks;
* [x] дуже довгі сторінки розбиті;
* [x] кожен chunk має provenance: document_id, page_number, filename/path;
* [x] є статистика по кількості chunks, середній довжині та strategy.

---

## Етап 5. Retrieval без LLM

Мета:

Отримати якісний пошук документів.

Результат:

* [x] SQLite FTS5 baseline по `chunks`;
* [x] скрипт `scripts/search_chunks.py`;
* [x] top-k результати пошуку;
* [x] provenance для кожного результату: файл, шлях, сторінка, document_id, chunk_id;
* [x] metadata filters: document_type, content_category, chunk_strategy;
* [x] baseline-тестові запити для оцінки якості пошуку.

Критерій приймання:

* [x] пошук повертає релевантні chunks;
* [x] видно файл і сторінку;
* [x] можна пояснити, чому результат знайдено;
* [x] є top-k результати;
* [x] OCR-документи не ламають пошук;
* [x] код не залежить від LLM, embeddings або Qdrant;
* [x] перевірені тестові запити:
  * [x] `умовні знаки топографічних карт`;
  * [x] `норми часу редакційно контрольні перевірки`;
  * [x] `MGRS UTM координати`;
  * [x] `порядок оформлення оперативних бойових документів`;
  * [x] `геопросторова підтримка Збройних Сил України`.

---

## Етап 6. Embeddings

Мета:

Порівняти класичний пошук та semantic search.

Результат:

* [x] Додати локальні embeddings через Ollama.
* [x] Використати модель BGE-M3.
* [x] Зберігати embeddings у SQLite.
* [x] Реалізувати повторний запуск без дублювання.
* [x] Реалізувати semantic search.
* [x] Показувати provenance для результатів.

Критерій приймання:

* [x] embeddings побудовані для всіх chunks;
* [x] повторний запуск не дублює embeddings;
* [x] semantic search повертає top-k з provenance;
* [x] результати можна порівняти з FTS baseline;
* [x] немає залежності від Qdrant, LLM або Agno;
* [x] перевірені ті самі тестові запити, що й для FTS baseline.

---

## Етап 6.5. Оцінка FTS vs Embeddings

Мета:

Порівняти якість пошуку SQLite FTS5 та semantic search на однакових тестових запитах.

Результат:

* [x] набір тестових запитів;
* [x] результати FTS;
* [x] результати embeddings;
* [x] ручна оцінка релевантності;
* [x] висновок, чи embeddings дають приріст;
* [x] рішення, чи переходити до hybrid search.

Критерій приймання:

* [x] створено evaluation-звіт;
* [x] порівняно мінімум 5 тестових запитів;
* [x] для кожного запиту визначено, який пошук кращий;
* [x] прийнято рішення щодо Етапу 7.

---

## Етап 7. Hybrid Search

Мета:

Об'єднати результати SQLite FTS5 та semantic search через embeddings.

Причина:

Оцінка Етапу 6.5 показала, що embeddings мають кращий Hit@1, але FTS має кращий Hit@10. Це означає, що методи доповнюють один одного.

Результат:

* [x] реалізовано hybrid search;
* [x] використано простий rank fusion;
* [x] порівняно FTS, embeddings та hybrid на одному evaluation наборі;
* [x] сформовано звіт;
* [x] прийнято рішення щодо reranking.

Критерій приймання:

* [x] hybrid search повертає top-k з provenance;
* [x] результати FTS і embeddings об'єднуються без LLM;
* [x] є evaluation-звіт FTS vs embeddings vs hybrid;
* [x] hybrid не погіршує якість на ключових запитах;
* [x] рішення про reranking зафіксовано в DECISIONS.md.

---

## Етап 7.5. Retrieval error analysis

Мета:

Проаналізувати запити, які погано знаходяться навіть після hybrid search.

Проблемні запити:

* q008: схилення та зближення меридіанів
* q009: система координат WGS-84

Результат:

* [x] визначити причину помилки;
* [x] вирішити, чи проблема в query, metadata, chunking, aliases або OCR;
* [x] запропонувати мінімальні виправлення без додавання reranker.

---

## Stage 8 - Local LLM Integration

Status: Done.

Goal:

Generate grounded answers from retrieved chunks.

Scope:

* [x] Ollama integration;
* [x] Prompt builder;
* [x] Source citations;
* [x] No agents;
* [x] No Agno;
* [x] No reranker.

Acceptance criteria:

* [x] User question;
* [x] Hybrid retrieval;
* [x] LLM answer;
* [x] Source references.

---

## Stage 8.5 - Model Benchmark

Status: Done.

Goal:

Обрати основну локальну LLM-модель для MVP на основі якості відповіді, швидкості, стабільності, цитування, української мови та відсутності reasoning leaks.

Result:

* [x] benchmark виконано на фінальних моделях;
* [x] збережено markdown/html/jsonl звіти;
* [x] порівняно швидкість, thinking leaks, language issues, citation behavior;
* [x] визначено фінальних кандидатів.

Acceptance criteria:

* [x] Benchmark script discovers Ollama models;
* [x] Benchmark reports are generated for selected models;
* [x] Summary markdown and HTML reports are generated;
* [x] JSONL with timings, prompts, contexts, answers, sources and quality flags is generated;
* [x] Benchmark reports show source `relative_path`, not only filename.

---

## Stage 8.6. Final model selection

Status: Done.

Decision:

* Primary model: `gemma4:e2b`
* Fallback model: `gemma3:4b`
* Reference baseline: `qwen3:8b`

Reason:

`gemma4:e2b` показала найкращий баланс швидкості, стабільності, української мови та no-answer поведінки.

---

## Stage 9. Source Traceability / Citation Improvements

Мета:

Покращити джерела так, щоб користувач міг реально знайти фрагмент у документі.

Проблема:

Зараз для DOC/DOCX часто показується `page 1`, бо документ витягується як один великий текстовий блок.

Потрібно підтримати:

* для PDF залишити сторінки;
* для DOC/DOCX додати більш людську локалізацію: розділ, пункт, абзац або фрагмент;
* таблиці: сторінка + таблиця / рядок, якщо можливо;
* у відповідях показувати `relative_path`;
* у майбутньому UI відкривати файл через backend endpoint;
* не використовувати `chunk_id` як основне посилання для кінцевого користувача.

Критерій приймання:

* [x] sources містять `relative_path`;
* [x] PDF sources містять page number;
* [x] DOC/DOCX sources не вводять в оману фальшивою сторінкою;
* [x] sources мають location;
* [x] sources мають preview або character range;
* [x] `ask.py` показує новий формат;
* [x] benchmark reports показують новий формат;
* [x] немає OCR/layout ускладнення;
* [x] є реалізація human-readable location для DOC/DOCX.

---

## Stage 10. FastAPI Backend MVP

Мета:

Надати HTTP API для майбутнього UI.

Scope:

* [x] FastAPI app
* [x] `GET /health`
* [x] `POST /ask`
* [x] Pydantic request/response schemas
* [x] reuse existing RAG-core
* [x] structured sources in API response

Out of scope:

* auth
* users
* chat history
* streaming
* document opening
* UI

Acceptance criteria:

* [x] API starts with uvicorn
* [x] `/health` returns ok
* [x] `/ask` returns answer, sources, meta
* [x] sources include `relative_path`, `location`, `preview`
* [x] API does not duplicate RAG logic from CLI

---

## Stage 11. OCR

Мета:

Підтримка сканованих документів.

---

## Stage 12. UI

Мета:

Інтерфейс користувача.

---

## Stage 13. Agent Layer (опціонально)

Мета:

Експерименти з Agno та агентами.
