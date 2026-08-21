# Project context for Claude Code

## What this is
An offline, on-premise document-chat SaaS. Users upload PDF/Word documents and
chat with them via RAG (Retrieval-Augmented Generation). **100% offline — zero
external API calls at runtime.** This constraint drives every architecture
decision. If a library/tool needs internet access at runtime (not just at
`pip install` time), flag it — it's a real problem, not a nitpick. (We already
hit this once with `tiktoken` downloading its encoding table on first use.)

Full architecture, tech stack rationale, data model, and phased roadmap live in
`ARCHITECTURE.md` in this repo — read it if you need the bigger picture before
making a design decision.

## Why this project exists (important — read before writing code)
The primary goal is **learning**: AI/ML/LLM/RAG/vector DB/chunking concepts, plus
backend/frontend/full-stack integration, for resume-building and interview
readiness. Working code is secondary to working, *understood* code.

### How to work with me because of this
- **Write the code yourself, but explain every concept as you go.** Don't just
  hand over a diff — explain *why* this approach, what the alternatives were,
  and what trade-off we're making. Assume I want to be able to explain any
  design decision in an interview.
- **Go deepest on RAG/LLM/vector concepts** (chunking, embeddings, retrieval,
  prompt construction, generation). Backend plumbing (FastAPI routes, DB
  models) and frontend can be explained more briefly — still correct, just
  less ceremony.
- Match the teaching style already in `backend/app/ingestion/chunker.py` —
  long docstring at the top explaining WHY the file exists and WHY each design
  decision was made, not just WHAT the code does. New files, especially in
  `ingestion/` and `rag/`, should follow this pattern.
- If you make a real-world mistake while building (like the tiktoken network
  dependency), don't just quietly fix it — call it out. Those are some of the
  most valuable lessons in this project.
- When there's a genuine trade-off (e.g. pgvector vs Qdrant, chunk size vs
  overlap), briefly state it rather than silently picking one.

## Tech stack (see ARCHITECTURE.md for full rationale)
- **Frontend:** React + Next.js + Tailwind + react-i18next
- **Backend:** FastAPI + Celery + Redis + Nginx
- **Database:** PostgreSQL + pgvector (MVP) → Qdrant later if needed
- **Embeddings:** BAAI/bge-m3 (multilingual — chosen now so Arabic in Phase 2
  doesn't require re-embedding the whole corpus)
- **LLM:** Qwen2.5 7B/14B quantized via Ollama (MVP) → vLLM at scale
- **Document parsing:** PyMuPDF (PDF) + python-docx (Word)
- **Auth:** Custom JWT + Argon2 (MVP) → Keycloak later for enterprise SSO

## Current phase: MVP (English only)
Building toward the MVP epics in `ARCHITECTURE.md` section 8:
1. Auth & access foundation
2. Document ingestion pipeline ← **in progress**
3. Core RAG & chat
4. Ops basics
5. Frontend shell

Don't build Phase 2 (Arabic) or Phase 3 (multi-tenant scale) features unless
explicitly asked — but keep the code extensible toward them per the
"configuration layer" pattern in `ARCHITECTURE.md` (e.g. language/model choices
belong in config, not hardcoded).

## Progress so far
- `backend/app/ingestion/chunker.py` — done. Recursive text splitting with
  token-aware sizing and overlap. Read this file first for the established
  code style before writing new ingestion code.
- `backend/app/ingestion/parser.py` — next up. PDF/DOCX text extraction
  feeding into the chunker, preserving page numbers for citations.

## House rules
- No external API calls at runtime, ever — this breaks the core product
  requirement, not just a style preference.
- Prefer explaining trade-offs over silently picking the "obviously right"
  answer — there usually isn't just one.
- Keep MVP scope narrow — resist adding Phase 2/3 features "while we're here."