"""
database.py
============

WHY THIS FILE EXISTS:

Every other file we've written so far (parser.py, chunker.py, embedder.py,
pipeline.py) works entirely in memory — call a function, get Python objects
back, nothing is saved anywhere. This file is the first thing in the
project that actually talks to a real, persistent database. Its only job
is to set up HOW we connect to Postgres, so every other file that needs a
database connection (models.py, store.py, and eventually the FastAPI
routes) can import a ready-made connection from here instead of each
inventing its own.

WHY THE CONNECTION STRING COMES FROM AN ENVIRONMENT VARIABLE, NOT A
HARDCODED STRING: this is the exact "configuration layer" principle
CLAUDE.md calls out explicitly — a database password is a secret, and
secrets don't belong hardcoded into source code that gets checked into git.
The DEFAULT value below matches docker-compose.yml's local dev credentials
purely for convenience (so a fresh clone of this repo "just works" against
the local Docker Postgres with zero setup) — but a real deployment would
set DATABASE_URL to point at its actual production database, with real
credentials, entirely outside of any file that lives in version control.

WHY SQLALCHEMY, RATHER THAN WRITING RAW SQL BY HAND (a genuine trade-off,
worth stating explicitly per CLAUDE.md's house rules): chunker.py's
chunking algorithm was written from scratch on purpose, specifically so the
RAG-relevant mechanics were visible to learn from — that's the CORE, novel
part of this whole project. Talking to a SQL database, by contrast, is
comparatively standard, well-trodden backend plumbing (CLAUDE.md itself
draws this distinction: go deep on RAG/LLM concepts, but backend plumbing
can lean on existing tools with less ceremony). SQLAlchemy is the
overwhelmingly standard choice for Python + Postgres, it's what you'll see
in real production FastAPI codebases, and it comes with a well-maintained
`pgvector` integration (see models.py) that handles converting Python
lists into Postgres's vector type correctly — reimplementing that by hand
would be extra work that teaches SQL escaping mechanics, not RAG concepts.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://docchat:docchat_dev_password@localhost:5432/docchat",
)

# The engine manages a pool of actual TCP connections to Postgres. Created
# ONCE, at import time, and reused for the lifetime of the process — same
# "expensive resource, load once, reuse many times" idea we've seen twice
# already (tiktoken's encoding in chunker.py, bge-m3's weights in
# embedder.py). Opening a brand new database connection for every single
# query would be needlessly slow; the engine's connection pool avoids that.
engine = create_engine(DATABASE_URL)

# A "session" is a temporary workspace for one unit of database work — you
# open one, do some reads/writes through it, then close it. SessionLocal
# isn't a session itself; it's a FACTORY that produces new Session objects
# on demand (calling SessionLocal() creates one). We don't want ONE shared
# session for the whole app: sessions aren't safe to use from multiple
# places at once, so the standard pattern is "make a fresh one for each
# unit of work, then close it" — store.py below does exactly that.
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    """
    The base class every ORM model (see models.py) inherits from. This is
    SQLAlchemy's mechanism for knowing which Python classes correspond to
    database tables at all — models.py's Document and DocumentChunk classes
    both inherit from this Base, and that's what lets SQLAlchemy generate
    the actual CREATE TABLE statements for them later.
    """

    pass


def get_session() -> Session:
    """Create a new database session. Caller is responsible for closing it
    (a `with` block, since Session supports the context manager protocol —
    same pattern we already used for fitz.Document in parser.py)."""
    return SessionLocal()
