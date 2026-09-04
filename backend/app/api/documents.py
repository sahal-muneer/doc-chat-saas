"""
documents.py
============

WHY THIS FILE EXISTS:

Every other file so far has been called directly from Python — a script,
or the test we ran manually. This file is the first thing in the project
that's reachable over HTTP: a real endpoint a browser, a frontend, or
`curl` can hit. Its job: accept an uploaded PDF/DOCX file, run it through
the full ingestion pipeline (parser -> pipeline -> chunker -> embedder ->
store), and report back what happened.

NO LONGER SYNCHRONOUS — INGESTION NOW RUNS IN A BACKGROUND WORKER: this
route used to do ALL the work — parsing, chunking, embedding, saving —
before sending any HTTP response back, which meant a large PDF could leave
the browser waiting many seconds. That's fixed now: upload_document()
below only saves the file and enqueues a Celery task
(process_document.delay(...)) — see app/worker/tasks.py for what actually
runs the pipeline, in a completely separate process. This route now
returns almost immediately, with status "pending", regardless of how big
the file is — the real work happens afterward, off this request entirely.
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.auth.dependencies import get_current_user_id
from app.core.rate_limit import limiter
from app.ingestion.store import (
    create_document,
    delete_document,
    get_document_owner,
    list_documents,
)
from app.worker.tasks import process_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Where uploaded files actually live on disk. parse_document() (parser.py)
# takes a file PATH, not raw bytes — so an uploaded file has to be saved
# somewhere real before ingest_document() can be called on it. This is a
# plain local folder for now; a real multi-server deployment would use
# shared/object storage instead (so any server instance can read a file
# regardless of which one handled the original upload) — another "fine for
# MVP, a real decision point later" simplification, same shape as several
# others we've flagged.
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_SUFFIXES = {".pdf", ".docx"}


@router.get("")
def get_documents(user_id: str = Depends(get_current_user_id)) -> list[dict]:
    """
    List every document belonging to the LOGGED-IN caller only — id,
    filename, status. `user_id` here comes entirely from
    get_current_user_id (dependencies.py): a request with no token, or an
    invalid one, never even reaches this function's body — FastAPI runs
    the dependency first and rejects with 401 before this code executes at
    all. This is the endpoint a frontend calls first, before it can call
    /chat at all: it's what turns "some UUID nobody knows" into an actual
    clickable list like "refund_policy.pdf, invoice_template.docx, ..."
    for a user to choose from — now correctly scoped to just their own
    documents. See store.py's list_documents() for why this returns plain
    dicts rather than raw database rows.
    """
    return list_documents(user_id)


@router.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request, file: UploadFile, user_id: str = Depends(get_current_user_id)
) -> dict:
    """
    Accept an uploaded PDF/DOCX file, save it, and QUEUE it for background
    processing — returns almost immediately with status "pending"; the
    actual parsing/chunking/embedding happens afterward, in the Celery
    worker (app/worker/tasks.py). Requires login — user_id (from the
    verified JWT) is stamped onto the new document immediately, at
    creation, so it has a real owner from the very first row written.

    RATE-LIMITED, 10/minute PER USER: each upload writes a file to disk and
    queues a real Celery job — cheap for one document, but nothing
    currently stops a user (or a bug in a script hitting this endpoint) from
    queueing hundreds of jobs in seconds and backing up the worker for
    everyone. `request: Request` here is unused directly in this function's
    body — it exists purely so the @limiter.limit decorator above can find
    the raw HTTP request it needs to identify the caller (see
    rate_limit.py).
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        # Same "fail fast, with a message that actually explains the
        # problem" instinct as parse_document()'s ValueError in parser.py —
        # just expressed as an HTTP error instead of a Python exception,
        # since the caller here is a client over the network, not code in
        # the same process.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .docx are supported.",
        )

    # Create the `documents` row FIRST, before touching the file at all.
    # This gives us document_id immediately, which we then use to name the
    # saved file on disk (see below) — so the on-disk filename and the
    # database row are linked by construction, not by a naming convention
    # someone has to remember to keep in sync.
    document_id = create_document(filename=file.filename or "unnamed", user_id=user_id)
    logger.info(f"upload started: document_id={document_id} filename={file.filename!r} user_id={user_id}")

    # Name the saved file after the document_id, not the original filename.
    # Two different users could both upload a file called "report.pdf" —
    # using the original name would mean the second upload silently
    # overwrites the first one's file on disk. document_id is guaranteed
    # unique (it's the primary key), so this can't collide.
    saved_path = UPLOAD_DIR / f"{document_id}{suffix}"
    with saved_path.open("wb") as buffer:
        # UploadFile wraps a temporary file-like object (file.file) that
        # FastAPI already streamed the upload into — shutil.copyfileobj
        # copies it to our real destination in chunks, rather than reading
        # the whole upload into memory as one giant bytes object first.
        # For a large PDF, that difference matters.
        shutil.copyfileobj(file.file, buffer)

    # .delay(...) is shorthand for "send this task to Redis and return
    # immediately" — it does NOT run process_document() here, in this
    # process. By the time this line finishes, the task message has been
    # handed to Redis; whether a worker is even running to pick it up yet
    # is a completely separate question this line doesn't wait to find out.
    process_document.delay(document_id, str(saved_path))
    logger.info(f"upload queued: document_id={document_id} filename={file.filename!r}")

    return {
        "document_id": document_id,
        "filename": file.filename,
        # "pending", not "ready" — this is the real, honest behavior
        # change from before: the response now reflects that processing
        # HASN'T happened yet, not that it definitely succeeded.
        "status": "pending",
    }


@router.delete("/{document_id}")
def delete_document_route(
    document_id: str, user_id: str = Depends(get_current_user_id)
) -> dict:
    """
    Delete a document — its database row, every chunk belonging to it
    (cascaded automatically, see store.py's delete_document()), and its
    saved file on disk. Requires login, and the document must actually
    belong to the caller — same 404-not-403 ownership check as
    api/chat.py's _ensure_owner, and the same anti-enumeration reasoning:
    trying to delete someone ELSE's document should look identical to
    trying to delete a document_id that was never real.
    """
    owner_id = get_document_owner(document_id)
    if owner_id is None or owner_id != user_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    delete_document(document_id)

    # The database doesn't know the file's suffix (.pdf vs .docx) — only
    # its own id-based naming convention from upload time does. Globbing
    # for "<document_id>.*" finds it without needing to store the suffix
    # anywhere, and naturally does nothing if the file was somehow already
    # missing, rather than raising an error over a file that's not there
    # to delete anyway.
    for path in UPLOAD_DIR.glob(f"{document_id}.*"):
        path.unlink()

    logger.info(f"document deleted: document_id={document_id} user_id={user_id}")
    return {"document_id": document_id, "deleted": True}
