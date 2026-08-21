"""
models.py
=========

WHY THIS FILE EXISTS:

These are the ORM ("Object-Relational Mapper") models — Python classes that
SQLAlchemy translates into real Postgres tables, and translates real
Postgres rows back into. Writing `Document(filename="report.pdf")` and
saving it creates an actual row in an actual `documents` table; querying
later gives you back a `Document` Python object, not raw SQL results.

TWO DIFFERENT "Chunk" CONCEPTS NOW EXIST IN THIS CODEBASE — READ THIS
BEFORE CONFUSING THEM: chunker.py's `Chunk` (a plain dataclass) is an
IN-MEMORY value — it exists only for the duration of one ingestion run, and
it doesn't know or care whether it will ever be saved anywhere. This file's
`DocumentChunk` (a database model) is a PERSISTED row — it exists as long
as the row exists in Postgres, survives the Python process ending, and can
be queried back out later by anything with database access. They hold
mostly the same fields, but they are NOT the same object, and one doesn't
inherit from or extend the other. store.py (next file) is the bridge: it
reads chunker.py's `Chunk` objects and creates `DocumentChunk` rows from
them. This split — a plain "domain" object separate from its "persistence"
representation — is a common, deliberate pattern in real backend systems:
it means chunker.py never has to import SQLAlchemy or know anything about
databases at all, and this file never has to know anything about how
chunking works. Each file only knows what it actually needs to.
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.ingestion.embedder import EMBEDDING_DIMENSIONS


class User(Base):
    """
    A registered account. Deliberately holds ONLY what authentication
    itself needs — email and a password HASH, never the real password (see
    app/auth/security.py for why: hashing, not encryption, and why Argon2
    specifically).

    WHY email IS unique=True: this is what makes "email" a valid way to
    look someone up at login time, and what stops two accounts from ever
    silently sharing one email — enforced by Postgres itself, not just
    application code remembering to check first (a real gap: two
    simultaneous signup requests for the same email could both pass an
    application-level check before either commits — the database
    constraint is what actually closes that race condition).
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        Text, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(Text, unique=True)
    hashed_password: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )


class Document(Base):
    """One uploaded file (PDF or DOCX). Chunks (below) belong to a Document
    via a foreign key — deleting a Document should take its chunks with it,
    which is what cascade="all, delete-orphan" on the relationship enforces
    at the ORM level (see the `chunks` relationship below)."""

    __tablename__ = "documents"

    # A plain string UUID, generated in Python (not by Postgres), so it's
    # already known and usable the moment the object is created — before
    # it's even been saved to the database. This matters: pipeline.py's
    # ingest_document() needs a document_id to stamp onto every Chunk it
    # creates, and it needs that id BEFORE any database write happens (see
    # store.py). A database-generated ID (e.g. an auto-incrementing integer)
    # wouldn't exist yet at that point — you'd have to save the Document
    # first, then go back and re-stamp every chunk with the ID Postgres
    # handed back. Generating the ID in Python up front avoids that
    # ordering problem entirely.
    id: Mapped[str] = mapped_column(
        Text, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(Text)

    # WHO uploaded this document — the actual "authorization" piece.
    # Without this column, every document belongs to no one in particular,
    # which is exactly the current state: any logged-in (or NOT logged in)
    # user can list, chat with, or upload documents regardless of who
    # actually owns them. Adding this FK is what makes "only YOUR
    # documents" an enforceable database-level fact, not just something a
    # route happens to check.
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))

    # A plain string status field rather than a database ENUM type — ENUMs
    # are more rigid (changing the allowed values later means a schema
    # migration) and for an MVP with only a handful of states, the
    # simplicity is worth more than the extra type safety right now.
    # Expected values: "pending", "processing", "ready", "failed".
    status: Mapped[str] = mapped_column(Text, default="pending")

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )

    # This lets you write `document.chunks` in Python and get back every
    # DocumentChunk row belonging to it, without writing a SELECT yourself.
    # cascade="all, delete-orphan" means deleting a Document also deletes
    # its chunks — you never want an orphaned chunk pointing at a document
    # that no longer exists.
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """
    The persisted counterpart of chunker.py's Chunk dataclass — see the
    module docstring above for why these are deliberately two separate
    things rather than one shared class.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    chunk_index: Mapped[int]
    page_number: Mapped[int | None]
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int]

    # Vector(EMBEDDING_DIMENSIONS) — imported from embedder.py rather than
    # hardcoding 1024 again here. This is exactly the payoff of defining
    # EMBEDDING_DIMENSIONS as a named constant back in embedder.py: if the
    # embedding model ever changes to one with a different vector size,
    # this schema definition updates automatically along with it, instead
    # of being a second place someone has to remember to edit by hand.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
