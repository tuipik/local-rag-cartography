# Run Commands

Команди запуску локального RAG-проєкту.

## Передумови

З кореня репозиторію:

```bash
uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -e . --python .venv/bin/python
```

Frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

Ollama має бути запущена, а моделі мають бути доступні:

```bash
ollama list
```

## Backend API

Запуск FastAPI:

```bash
.venv/bin/uvicorn local_rag.api.app:app --reload
```

API буде доступне за адресою:

```text
http://127.0.0.1:8000
```

Перевірка:

```bash
curl http://127.0.0.1:8000/health
```

Тестовий запит:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "що таке система координат WGS-84?",
    "top_k": 5,
    "llm_model": "gemma4:e2b",
    "num_predict": 1024
  }'
```

## Frontend UI

В окремому терміналі:

```bash
cd frontend
npm run dev
```

UI буде доступний за адресою:

```text
http://127.0.0.1:5173
```

Якщо backend запущений не на `http://127.0.0.1:8000`, передай URL:

```bash
cd frontend
VITE_RAG_API_URL=http://127.0.0.1:8000 npm run dev
```

Production build:

```bash
cd frontend
npm run build
```

## CLI Smoke Tests

Hybrid search:

```bash
.venv/bin/python scripts/search_hybrid.py \
  "норми часу редакційно контрольні перевірки" \
  --top-k 5 \
  --no-rebuild-fts
```

Ask через CLI:

```bash
.venv/bin/python scripts/ask.py \
  "які норми часу встановлені для редакційно контрольних перевірок?" \
  --top-k 5 \
  --no-rebuild-fts
```

## Document Links

Metadata:

```bash
curl http://127.0.0.1:8000/documents/1/metadata
```

View:

```bash
curl -I http://127.0.0.1:8000/documents/1/view
```

Download:

```bash
curl -I http://127.0.0.1:8000/documents/1/download
```
