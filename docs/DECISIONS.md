# DECISIONS

## 2026-06-15

### Embedding model

Рішення:

Використовувати BGE-M3.

Причини:

* multilingual;
* добре працює з нормативними документами;
* підтримує hybrid retrieval;
* підтримує sparse retrieval.

Альтернативи:

* nomic-embed-text-v2-moe

Причина відмови:

Менше переваг для нормативної документації.

---

### Порядок реалізації

Рішення:

Спочатку retrieval, потім LLM.

Причини:

Якість retrieval визначає якість всієї системи.

---

### Agent Framework

Рішення:

Не використовувати Agno на початкових етапах.

Причини:

Потрібно спочатку довести якість retrieval.


## 2026-06-15

### Metadata storage

Рішення:

На перших етапах використовувати SQLite як основне сховище метаданих.

Причини:

- простота;
- локальність;
- достатньо для 150 документів;
- зручно для аудиту, витягу тексту, chunking та тестування retrieval.

Qdrant буде додано пізніше, після появи стабільних chunks.

---

## 2026-06-16

### Text extraction stage

Рішення:

Етап 2 виконує тільки витяг машинно-читаного тексту з PDF, DOCX, DOC та TXT і записує результат у SQLite.

Межі:

- не робити OCR;
- не робити embeddings;
- не підключати LLM.

Підхід:

- PDF: PyMuPDF;
- DOCX: python-docx;
- TXT: читання plain text із кількома кодуваннями;
- DOC: конвертація через LibreOffice у TXT.

Для DOC використовується timeout, щоб один проблемний файл не блокував увесь етап.

---

### Metadata extraction stage

Рішення:

Етап 3 починається з rule-based витягу метаданих без LLM.

Поля:

- document_type;
- content_category;
- document_number;
- document_date;
- organization;
- metadata_status;
- metadata_notes.

Джерела правил:

- назва файлу;
- папка та відносний шлях;
- перші символи тексту з document_pages.

Пріоритет:

Спочатку правила за назвою файлу, потім за шляхом/папкою, потім за текстом документа.

Причина:

Для першого проходу потрібна прозора класифікація, яку можна рев'ювати через metadata_notes і поступово покращувати.

---

### Corpus analysis before chunking

Рішення:

Перед реалізацією chunking додати проміжний етап аналізу корпусу документів.

Причини:

- документи мають різну структуру;
- page-based chunking може бути недостатнім;
- fixed-size chunking може зруйнувати логіку нормативних пунктів;
- для якісного retrieval потрібно зберегти структуру документа.

Альтернативи:

- одразу перейти до fixed-size chunking;
- використовувати однакову chunking strategy для всіх документів.

Причина відмови:

Це може погіршити якість retrieval і цитування.

---

### Chunking strategy

Рішення:

Не використовувати одну глобальну fixed-size chunking strategy для всіх документів.

Для Етапу 4 використати гібридну стратегію:

- structure-aware chunking для документів із пунктами та підпунктами;
- page-based chunking для коротких сторінок і документів із природною посторінковою структурою;
- secondary splitting для дуже довгих сторінок або DOC/DOCX-фрагментів;
- table/page-oriented chunking для довідкових і табличних матеріалів;
- OCR-кандидати тимчасово виключити з chunking.

Причини:

- корпус містить різні типи документів;
- DOC/DOCX часто витягуються як один великий фрагмент;
- нормативні документи мають пункти та підпункти;
- таблиці й довідкові матеріали вимагають збереження локального контексту;
- fixed-size chunking може зруйнувати логіку нормативних пунктів.

Компроміс:

Перша версія chunking буде rule-based baseline, а не ідеальний семантичний парсер структури документа.

---

### SQLite FTS5 retrieval baseline

Рішення:

Для Етапу 5 використати SQLite FTS5 як baseline retrieval без LLM, embeddings, Ollama або Qdrant.

Підхід:

- індексувати `chunks` у `chunks_fts`;
- шукати по `chunk_text`;
- повертати top-k chunks;
- показувати provenance: файл, шлях, сторінка, document_id, chunk_id, document_type, content_category, chunk_strategy;
- підтримати metadata filters;
- додати `--prefer-reference` для м'якого підняття довідкових і нормативних матеріалів.

Причини:

- потрібен контрольний keyword/full-text baseline перед embeddings;
- результати можна пояснити через збіг термінів у chunk_text;
- provenance дозволяє оцінювати якість пошуку та цитування;
- embeddings надалі треба порівнювати з baseline, а не оцінювати на око.

Обмеження:

SQLite FTS5 не виконує українську морфологічну нормалізацію, тому цей baseline не замінює майбутній semantic або hybrid search.

---

### Local embeddings in SQLite

Рішення:

Для Етапу 6 додати semantic search через локальну Ollama-модель `bge-m3`, але не замінювати SQLite FTS5 baseline.

Підхід:

- embeddings будуються для `chunks`;
- вектори зберігаються в SQLite у таблиці `chunk_embeddings`;
- embedding зберігається як float32 BLOB;
- пошук виконується через cosine similarity у Python;
- результати semantic search показують той самий provenance, що й FTS baseline;
- hybrid search поки не реалізується.

Причини:

- корпус має лише кілька тисяч chunks, тому SQLite достатньо для першої оцінки;
- простіше дебажити і перевіряти повноту embeddings;
- semantic search треба порівняти з FTS baseline до додавання складнішої інфраструктури;
- Qdrant варто додавати тільки після підтвердження користі embeddings.

Обмеження:

Це окремий semantic baseline, а не hybrid retrieval. Порівняння FTS vs embeddings буде виконано окремим етапом перед об'єднанням сигналів.

---

### Embeddings storage

Рішення:

На першому етапі embeddings зберігати в SQLite у таблиці `chunk_embeddings`.

Схема підтримує кілька моделей через унікальність:

`UNIQUE(chunk_id, model)`

Причини:

- корпус невеликий;
- не потрібна додаткова інфраструктура;
- простіше дебажити;
- можна порівнювати різні embedding-моделі;
- Qdrant поки передчасний.

Компроміс:

SQLite-пошук по embeddings виконується повним перебором cosine similarity і не масштабується на великі корпуси.

---

### Hybrid search after evaluation

Рішення:

Після оцінки FTS vs embeddings перейти до Hybrid Search.

Причини:

- embeddings показали кращий Hit@1;
- FTS показав кращий Hit@10;
- проблемні кейси різні для FTS та embeddings;
- отже, методи доповнюють один одного.

Підхід:

На першому етапі використати простий Reciprocal Rank Fusion (RRF), без reranker, без LLM і без Qdrant.

Початкові параметри baseline:

- `rrf_k = 60`;
- `fts_weight = 1.5`;
- `embedding_weight = 1.0`.

Причина:

На evaluation-наборі така вага зберігає сильний Hit@1 від embeddings і не погіршує Hit@10 від FTS.

Альтернативи:

- залишити тільки FTS;
- залишити тільки embeddings;
- одразу додати reranker;
- одразу перейти на Qdrant hybrid search.

Причина відмови:

Reranker і Qdrant поки передчасні. Спочатку треба перевірити простий hybrid baseline.

---

### Current retrieval baseline

Рішення:

Поточним retrieval baseline вважати RRF Hybrid Search.

Підхід:

- SQLite FTS5 дає keyword/full-text сигнал;
- BGE-M3 embeddings дають semantic сигнал;
- результати об'єднуються через Reciprocal Rank Fusion;
- baseline не використовує LLM, reranker або Qdrant.

Параметри:

- `rrf_k = 60`;
- `fts_weight = 1.5`;
- `embedding_weight = 1.0`.

Причини:

- hybrid зберігає Hit@1 на рівні embeddings;
- hybrid зберігає Hit@10 на рівні FTS;
- hybrid покращує Hit@3 на evaluation-наборі;
- підхід простий, прозорий і відтворюваний.

---

### Retrieval error analysis before reranking

Рішення:

Не додавати reranker одразу після hybrid search.

Причини:

- проблемні кейси q008 і q009 не є чистими retrieval failures;
- q008 повертає релевантний навчальний матеріал;
- q009 є неоднозначним широким запитом;
- частина проблем пов'язана з benchmark design, а не з пошуковим алгоритмом.

Наступний крок:

Розширити evaluation dataset і додати типи запитів.
