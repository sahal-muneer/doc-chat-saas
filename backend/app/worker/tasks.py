"""
tasks.py
========

WHY THIS FILE EXISTS:

The actual background task definition — the function that runs inside the
SEPARATE Celery worker process, not inside the FastAPI request handler
anymore. Deliberately reuses ingest_document() / save_chunks() /
update_document_status() UNCHANGED from api/documents.py's old inline
version — none of the real ingestion logic moved or duplicated, only WHERE
it gets called from changed. This is exactly the payoff of having kept
that logic in reusable functions (pipeline.py, store.py) from the very
start, instead of writing it directly inside the route handler.
"""

import logging
from pathlib import Path

from app.ingestion.pipeline import ingest_document
from app.ingestion.store import save_chunks, update_document_status
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="process_document")
def process_document(document_id: str, file_path: str) -> None:
    """
    Run the full parse -> chunk -> embed -> store pipeline for one already-
    uploaded, already-saved-to-disk file. Called via
    `process_document.delay(document_id, file_path)` from api/documents.py
    — .delay() doesn't run this function directly; it serializes these
    arguments into a message, sends it to Redis, and returns immediately.
    THIS function only actually executes later, inside the worker process,
    whenever it picks that message up.

    file_path IS A STRING, NOT A Path OBJECT: Celery has to serialize task
    arguments to send them through Redis (by default, as JSON) — a Path
    object isn't JSON-serializable, a plain string is. Converting back to
    Path(file_path) below is a one-line cost for that constraint.

    WHY THIS HAS NO try/except AROUND update_document_status("failed",...)
    LIKE THE OLD INLINE VERSION HANDLED IT DIFFERENTLY: actually, it still
    needs that — see below. The difference worth noting is WHO finds out
    about a failure now: previously, an exception here would become an
    HTTPException the browser saw immediately. Now, this code runs with no
    browser connection waiting on it at all — a crash here would be
    completely invisible to the user unless we explicitly log it AND
    update the document's status, which is exactly why both still happen
    below.
    """
    try:
        chunks = ingest_document(Path(file_path), document_id=document_id)
        save_chunks(chunks)
        update_document_status(document_id, "ready")
        logger.info(f"processing succeeded: document_id={document_id} chunk_count={len(chunks)}")
    except Exception:
        update_document_status(document_id, "failed")
        logger.exception(f"processing failed: document_id={document_id}")
