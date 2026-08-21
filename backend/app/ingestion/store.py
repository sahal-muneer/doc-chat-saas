"""
store.py
========

WHY THIS FILE EXISTS:

pipeline.py produces a list of fully-embedded, in-memory chunker.py `Chunk`
objects — but per models.py's docstring, a `Chunk` is a value that only
exists for the duration of one Python process. This file is what actually
makes ingestion durable: it takes those in-memory Chunks and writes them
into Postgres as `DocumentChunk` rows (models.py), so they survive the
process ending and can be searched later, potentially by an entirely
different process (e.g. a chat API handling a question days after the
document was uploaded).

This file is intentionally the ONLY place in the ingestion pipeline that
imports database code (database.py, models.py). parser.py, chunker.py, and
embedder.py all stay free of any SQLAlchemy import — none of them need to
know a database exists at all. Keeping that boundary sharp means each file
can be tested, understood, and changed in isolation, without a database
even needing to be running (see the parser.py/chunker.py/embedder.py demos
we've already run — none of them touched Postgres).
"""

from sqlalchemy import select

from app.db.database import Base, engine, get_session
from app.db.models import Document, DocumentChunk
from app.ingestion.chunker import Chunk


def create_tables() -> None:
    """
    Create every table defined in models.py, if it doesn't already exist.

    WHY THIS EXISTS, AND WHY IT'S THE RIGHT TOOL FOR RIGHT NOW, NOT
    FOREVER: `Base.metadata.create_all(engine)` looks at every model class
    that inherits from Base (Document, DocumentChunk) and issues the
    matching CREATE TABLE statements, skipping any table that already
    exists. This is the simplest possible way to get a schema into an
    empty database, which is exactly what an MVP needs. What it CANNOT do
    is evolve a schema that already has data in it — if you later add a
    column to DocumentChunk, calling this function again won't add that
    column to the existing table; it'll just see the table already exists
    and do nothing. Real production systems use a migration tool (Alembic
    is the standard one for SQLAlchemy) that tracks a versioned history of
    schema changes and can upgrade an existing database step by step. This
    project doesn't have any real data to protect yet, so create_all() is
    the right amount of machinery for right now — Alembic is a "when the
    schema needs to change without losing data" problem, not a "day one"
    problem, and CLAUDE.md's instruction to keep MVP scope narrow argues
    directly against reaching for it before it's actually needed.
    """
    Base.metadata.create_all(engine)


def create_document(filename: str, user_id: str) -> str:
    """
    Insert a new `documents` row for a file that's about to be ingested,
    and return its id — which the caller then passes to
    pipeline.ingest_document(document_id=...) so every Chunk produced
    stamps itself with the correct document it belongs to.

    user_id STAMPS OWNERSHIP AT THE MOMENT OF CREATION: this is what makes
    "only YOUR documents" enforceable later — list_documents() and the
    ownership check in get_document_owner() both rely on this column having
    been set correctly from the very first insert, not patched on after
    the fact.

    A SQLALCHEMY GOTCHA THIS FUNCTION DELIBERATELY AVOIDS: by default,
    after `session.commit()`, SQLAlchemy marks every object touched in
    that session as "expired" — its attribute values get discarded, and
    the NEXT time you read one of them, SQLAlchemy silently runs a fresh
    query to reload it, using that same session's connection. That's fine
    as long as the session is still open. But once the `with get_session()`
    block below exits, the session is closed — so if you tried to read
    `document.id` AFTER this function returned, there'd be no open
    connection left to refresh it from, and you'd hit a
    `DetachedInstanceError` (a genuinely common real-world SQLAlchemy bug).
    The fix here: `return document.id` is the very last thing that happens
    INSIDE the `with` block, while the session is still open — so the
    (re)fetch, if one even happens, succeeds safely, and what actually
    leaves this function is a plain Python string, completely independent
    of the database session that produced it.
    """
    with get_session() as session:
        document = Document(filename=filename, user_id=user_id)
        session.add(document)
        session.commit()
        return document.id


def save_chunks(chunks: list[Chunk]) -> None:
    """
    Persist a list of chunker.py's in-memory Chunk objects as DocumentChunk
    rows — the actual chunker.py Chunk -> database DocumentChunk translation
    that models.py's docstring describes.

    WHY WE BUILD THE WHOLE LIST OF ROWS FIRST, THEN save ONCE with
    add_all() + ONE commit() — NOT one add()+commit() per chunk in a loop:
    this is the exact same "batch it, don't do it one at a time" idea
    we've already seen twice — chunk_text() packs many small text pieces
    together before finalizing a Chunk, and embed_chunks() embeds many
    texts in one model.encode() call rather than one at a time. Here,
    every commit() is a round trip that asks Postgres to durably write
    data to disk (this is what makes a commit safe against a crash, but
    also what makes it comparatively slow). Committing once for 50 chunks
    is far faster than committing 50 times for one chunk each — same
    underlying reason as embedder.py's batching, applied to database
    writes instead of model inference.
    """
    if not chunks:
        return

    rows = [
        DocumentChunk(
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            text=chunk.text,
            token_count=chunk.token_count,
            embedding=chunk.embedding,
        )
        for chunk in chunks
    ]

    with get_session() as session:
        session.add_all(rows)
        session.commit()


def list_documents(user_id: str) -> list[dict]:
    """
    Return every document belonging to user_id, newest first — the thing a
    frontend calls BEFORE it can call anything else in api/chat.py, since
    /chat needs a document_id and, until now, the only way to learn one
    was to have been the person who just uploaded that exact file.

    NOW SCOPED TO user_id, NOT EVERY DOCUMENT IN THE SYSTEM: this is the
    actual authorization change — before this, this function had no
    concept of "whose" documents it was returning at all, which meant
    every logged-in user could see every other user's document list. The
    `.where(Document.user_id == user_id)` below is the entire fix.

    RETURNS PLAIN DICTS, NOT Document OBJECTS — same DetachedInstanceError
    reasoning as create_document() and search_chunks() in retriever.py:
    once the `with get_session()` block below closes, reading attributes
    off a Document object becomes unsafe. Converting to plain dicts INSIDE
    the session block sidesteps that entirely.
    """
    with get_session() as session:
        documents = session.scalars(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        ).all()

        return [
            {
                "document_id": document.id,
                "filename": document.filename,
                "status": document.status,
                "created_at": document.created_at.isoformat(),
            }
            for document in documents
        ]


def get_document_owner(document_id: str) -> str | None:
    """
    Return the user_id that owns document_id, or None if no such document
    exists. This exists specifically for api/chat.py to check BEFORE
    running retrieval: "does the document being asked about actually
    belong to whoever is asking?" Kept separate from search_chunks()
    (retriever.py) on purpose — retriever.py's job is purely "find similar
    chunks," a general-purpose search utility with no concept of who's
    allowed to ask; ownership is an authorization question, which belongs
    at the API layer that already knows who the current user is, not
    buried inside a search function.
    """
    with get_session() as session:
        document = session.get(Document, document_id)
        return document.user_id if document is not None else None


def delete_document(document_id: str) -> None:
    """
    Delete a document row — and, because of cascade="all, delete-orphan"
    on Document.chunks (models.py), every DocumentChunk belonging to it
    gets deleted right along with it, automatically. Without that cascade
    setting, deleting a Document while its chunks still pointed at it via
    a foreign key would either fail outright (Postgres refusing to leave a
    dangling reference) or leave orphaned chunk rows behind forever.

    DOES NOT TOUCH THE FILE ON DISK: this function only knows about the
    database — it has no idea documents/uploads even exists as a folder.
    Deleting the actual saved file is api/documents.py's job (see its
    delete route), the same boundary that's existed in this file since it
    was first written: store.py is the ONLY place that imports database
    code, and stays that way even for deletion.
    """
    with get_session() as session:
        document = session.get(Document, document_id)
        if document is not None:
            session.delete(document)
            session.commit()


def update_document_status(document_id: str, status: str) -> None:
    """
    Change a document's status (e.g. "pending" -> "ready" once ingestion
    finishes, or -> "failed" if it errors out). This is a small, separate
    function rather than folding a status update into create_document() or
    save_chunks(), because it happens at a DIFFERENT time than either of
    those: create_document() runs BEFORE ingestion starts (status stays
    "pending"), save_chunks() runs partway through, and the final "ready"
    or "failed" update only makes sense once the whole pipeline — parsing,
    chunking, embedding, AND saving — has either fully succeeded or thrown
    partway through. The upload endpoint (api/documents.py) is what
    actually knows which of those outcomes happened, so it's the one that
    calls this, at the end.
    """
    with get_session() as session:
        document = session.get(Document, document_id)
        document.status = status
        session.commit()
