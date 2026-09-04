"""
conftest.py
===========

WHY THIS FILE EXISTS: pytest automatically discovers a file literally named
conftest.py and treats everything in it as shared setup available to every
test file in this directory — this is where FIXTURES (reusable setup/
teardown, injected into a test just by naming them as a parameter) live, so
individual test files don't each need their own copy of "spin up a test
database" or "build an authenticated client."

WHY A SEPARATE TEST DATABASE (docchat_test), NOT THE REAL DEV DATABASE
(docchat): tests need to freely create and delete rows without any risk of
wiping out real data you're looking at in the app while developing — running
the test suite should never be able to delete a document you just uploaded
to manually test something. A completely separate database, created and torn
down by the test run itself, makes that structurally impossible rather than
a rule someone has to remember to follow.

WHY A SEPARATE REDIS LOGICAL DATABASE TOO (index 1, not 0): rate_limit.py's
Limiter stores its request counters in Redis. If tests used the same Redis
database as real dev usage (index 0), two things would go wrong: test runs
would accumulate counters across sessions (a test suite run twice in a row
could start already partially rate-limited, a classically "flaky test"
cause), and hammering /auth/login in a test could burn through YOUR real
rate-limit quota while you're using the app normally. Redis supports 16
separate numbered logical databases out of the box, entirely isolated from
each other, on the same running server — index 1 costs nothing extra to use
and fixes both problems.

WHY THE ENV VAR OVERRIDES HAPPEN BEFORE ANY app.* IMPORT, AT THE VERY TOP OF
THIS FILE: database.py and rate_limit.py both read their URLs from the
environment and build a module-level engine/Limiter ONCE, at import time
(see database.py's own docstring: "created ONCE, reused for the lifetime of
the process"). If any app module were imported before these env vars are
set, that module would already be holding a connection to the real dev
database/Redis DB, and setting the env var afterward would change nothing.
"""

import os

os.environ["DATABASE_URL"] = (
    "postgresql+psycopg://docchat:docchat_dev_password@localhost:5432/docchat_test"
)
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["JWT_SECRET"] = "test-only-secret-not-used-anywhere-real"

import psycopg
import pytest
import redis
from sqlalchemy import text

from app.db.database import Base, engine


def _ensure_test_database_exists() -> None:
    """
    Postgres has no "CREATE DATABASE IF NOT EXISTS" syntax, and a database
    can't be created from within a connection that's already connected TO
    it — so this connects to the "postgres" admin database (which always
    exists on any Postgres server) first, checks the catalog for whether
    docchat_test is already there, and creates it if not.
    """
    admin_conn = psycopg.connect(
        "postgresql://docchat:docchat_dev_password@localhost:5432/postgres",
        autocommit=True,
    )
    try:
        exists = admin_conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", ("docchat_test",)
        ).fetchone()
        if not exists:
            admin_conn.execute("CREATE DATABASE docchat_test")
    finally:
        admin_conn.close()


@pytest.fixture(scope="session", autouse=True)
def test_database():
    """
    Runs ONCE per test session, automatically (autouse=True — no test needs
    to name this explicitly to get it). Makes sure docchat_test exists,
    enables the pgvector extension inside it, and creates every table from
    models.py fresh.

    WHY pgvector NEEDS ENABLING HERE, EVEN THOUGH THE REAL docchat DATABASE
    ALREADY HAS IT: the pgvector/pgvector Docker image auto-enables the
    extension only in the ONE database named by POSTGRES_DB at the
    container's first-ever startup (docchat, per docker-compose.yml) — a
    database created LATER, after that, like docchat_test here, does not
    get it for free. `CREATE EXTENSION IF NOT EXISTS vector` is the same
    one-time step, just run explicitly instead of by the image's init
    script.
    """
    _ensure_test_database_exists()
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="session", autouse=True)
def clean_rate_limit_redis():
    """
    Flushes the TEST Redis database (index 1 — see module docstring) once
    at the start of the session, so leftover counters from a previous test
    run can never cause a test to start already partially (or fully)
    rate-limited. flushdb() only touches index 1, never the real dev
    database at index 0 — that isolation is exactly what makes this safe to
    do unconditionally.
    """
    client = redis.Redis.from_url(os.environ["REDIS_URL"])
    client.flushdb()


@pytest.fixture
def db_session():
    """
    A single database session for one test, closed automatically when the
    test finishes — the same get_session() context-manager pattern used
    throughout the actual app code (store.py, history.py), so tests read
    the database the same way the app itself does.
    """
    from app.db.database import get_session

    with get_session() as session:
        yield session


@pytest.fixture
def client():
    """
    FastAPI's TestClient — sends real requests through the actual app
    (routing, dependencies, middleware, the rate limiter, all of it)
    without needing a real `uvicorn` process running. Built on httpx
    under the hood (see requirements-dev.txt).
    """
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
