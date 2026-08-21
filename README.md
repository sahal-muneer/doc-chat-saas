# doc-chat

An offline, on-premise document-chat app. Upload a PDF or Word document,
log in, and ask it questions in plain English — answers stream back in
real time, grounded strictly in the document's own content, with page
citations. **Every part of this — including the AI itself — runs entirely
on your own machine. Nothing is ever sent to an external API.**

## Architecture at a glance

- **Frontend**: Next.js (React + TypeScript + Tailwind)
- **Backend**: FastAPI
- **Database**: PostgreSQL + pgvector (vector search)
- **Background jobs**: Redis + Celery (document processing runs off the request path)
- **AI**: BAAI/bge-m3 (embeddings) + Qwen2.5 7B via Ollama (generation) — both run locally

Three long-running processes make up the app: the FastAPI backend, a
Celery worker, and the Next.js dev server. Postgres and Redis run in
Docker.

## Prerequisites

| Tool | Notes |
|---|---|
| **Python 3.11+** | Developed against 3.14 |
| **Node.js 18+** | Developed against v26.7 |
| **Docker Desktop** | Runs Postgres + Redis |
| **Ollama** | [ollama.com](https://ollama.com) — runs the LLM locally |
| **~10GB free disk** | Embedding model (~4.3GB) + Qwen2.5:7B (~4.7GB) |
| **16GB+ RAM recommended** | For running Qwen2.5:7B comfortably alongside everything else |

## Setup

### 1. Clone and start Postgres + Redis

```bash
git clone <this-repo-url>
cd doc-chat-saas
docker compose up -d
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The backend reads a few settings from environment variables, all with
working local defaults — you don't need to set anything to run this
locally:

- `DATABASE_URL` (defaults to the Postgres container started above)
- `REDIS_URL` (defaults to the Redis container started above)
- `JWT_SECRET` (defaults to a dev-only value — **not for production use**)

### 3. Pull the LLM

```bash
ollama pull qwen2.5:7b
```

This downloads ~4.7GB, one time. Make sure `ollama serve` is running
(Homebrew installs usually start it automatically as a background
service — check with `ollama list`).

### 4. Start the backend + worker (two separate terminals, `backend/` with the venv active in both)

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
celery -A app.worker.celery_app worker --loglevel=info
```

The database tables are created automatically the first time the
backend starts — no separate migration step needed.

### 5. Frontend (a third terminal)

```bash
cd frontend
npm install
npm run dev
```

## Using it

1. Open **http://localhost:3000**
2. Sign up with any email/password (this is a fresh database — there's
   no pre-existing account or seed data)
3. Upload a PDF or Word document from the sidebar
4. Once its status flips from `pending` to `ready` (a few seconds,
   automatic), click it and ask it a question

Backend API docs (Swagger UI): **http://localhost:8000/docs**

## Verifying it's really offline

Everything above works with your internet connection fully disabled,
once the one-time downloads (Ollama's model, the embedding model, the
Docker images, and `npm`/`pip` packages) have completed. There's no
step at runtime that calls out to any external service.

## Troubleshooting

- **"Ollama not found" / connection refused on port 11434** — Ollama
  isn't running. Start it with `ollama serve`, or check
  `brew services list` if installed via Homebrew.
- **Port already in use** — something else is already using 8000, 3000,
  5432, or 6379. Stop the conflicting process or adjust the relevant
  port in `docker-compose.yml` / the `uvicorn`/`npm run dev` commands.
- **A document stays stuck on `pending`** — the Celery worker (step 4,
  second terminal) isn't running. Uploaded files are only processed
  while it's alive.
- **CORS errors in the browser console** — the backend only allows
  requests from `http://localhost:3000` by default (see `main.py`). If
  you're running the frontend on a different port, that'll need
  updating.
