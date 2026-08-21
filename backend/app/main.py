"""
main.py
=======

WHY THIS FILE EXISTS:

Every FastAPI project needs exactly one place where the `FastAPI()`
application object actually gets created and all the individual routers
(documents.py, and eventually auth, chat, etc.) get registered onto it.
This is that place — the actual thing `uvicorn` runs to start the server.
Individual route files (api/documents.py) define WHAT endpoints exist;
this file is what actually assembles them into one running app.
"""

import logging

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.logging import setup_logging
from app.db.database import engine
from app.ingestion.store import create_tables

# Called FIRST, before anything else in this file — every other file's
# `logging.getLogger(__name__)` calls rely on setup_logging() having
# already configured the format/level, so this has to run before any
# route, or even create_tables() below, has a chance to log anything.
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="doc-chat-saas")

# Browsers block cross-origin requests by default — a page served from
# localhost:3000 (the Next.js frontend) trying to call localhost:8000 (this
# API) gets silently rejected by the BROWSER itself, before the request
# even reaches this server, unless this server explicitly says "requests
# from that origin are allowed." This is a browser-only protection: curl,
# Swagger's "Try it out", and our earlier tests never hit it, because
# CORS is enforced by the browser, not the server or the HTTP protocol
# itself. Listing the exact frontend origin (not "*", allow-everything)
# means only this specific frontend can call the API from a browser — a
# meaningful difference even in local dev, and the right habit to have
# before this ever runs somewhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure the database schema exists before the app starts accepting
# requests. This is the same create_tables() we already tested manually —
# calling it here means a fresh checkout of this project, with Postgres
# running, "just works" the first time you start the server, with no
# separate manual setup step to remember.
create_tables()

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> JSONResponse:
    """
    A REAL health check — actually verifies the two things this app
    cannot function without, instead of just confirming the FastAPI
    process itself is alive.

    WHY THE OLD VERSION ("just return {'status': 'ok'}") WAS MISLEADING:
    that version would report "ok" even if Postgres had crashed, or Ollama
    had been stopped — because it never actually asked either of them
    anything. A monitoring system (or a human, at 3am) trusting that
    health check would see "ok" while /chat was completely broken for
    every user. A health check that can't actually detect the failures
    that matter isn't worth having.

    WHY 503, NOT 200, WHEN A DEPENDENCY IS DOWN: 503 ("Service
    Unavailable") is the standard HTTP status specifically meaning "I'm
    not able to serve requests properly right now" — this is what lets
    automated tools (a load balancer, a container orchestrator like
    Kubernetes) detect "this instance is unhealthy, stop sending it
    traffic" purely from the status code, without needing to parse the
    response body at all.
    """
    checks = {}

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.error(f"health check: database unreachable: {exc}")
        checks["database"] = "unreachable"

    try:
        # /api/tags is a lightweight Ollama endpoint that just lists
        # locally-installed models — enough to prove Ollama itself is up
        # and responding, without the cost of an actual generation call.
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        checks["ollama"] = "ok"
    except Exception as exc:
        logger.error(f"health check: ollama unreachable: {exc}")
        checks["ollama"] = "unreachable"

    healthy = all(status == "ok" for status in checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
