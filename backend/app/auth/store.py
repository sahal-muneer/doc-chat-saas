"""
store.py (auth)
================

WHY THIS FILE EXISTS:

Same separation-of-concerns idea as ingestion/store.py: this is the ONLY
place in the auth system that talks to the database directly. api/auth.py
calls these functions but never imports SQLAlchemy or the User model
itself — it just asks for "create a user" or "find a user by email" and
gets back plain data, the same DetachedInstanceError-avoiding pattern
used everywhere else in this project (retriever.py's RetrievedChunk,
store.py's list_documents()).
"""

from dataclasses import dataclass

from sqlalchemy import select

from app.auth.security import hash_password
from app.db.database import get_session
from app.db.models import User


@dataclass
class UserRecord:
    """A plain, database-independent snapshot of one user row."""

    id: str
    email: str
    hashed_password: str


def create_user(email: str, password: str) -> str:
    """
    Hash the password (security.py — the real password never gets stored),
    create the User row, and return the new user's id. If `email` already
    belongs to another account, Postgres's unique constraint on
    User.email rejects this with an IntegrityError — the caller
    (api/auth.py) is what turns that into a clean 400 response instead of
    a raw database error leaking out over HTTP.
    """
    with get_session() as session:
        user = User(email=email, hashed_password=hash_password(password))
        session.add(user)
        session.commit()
        return user.id


def get_user_by_email(email: str) -> UserRecord | None:
    """
    Look up a user by email — the first step of login. Returns None (not
    an exception) when no such user exists, since "no account with this
    email" is an entirely normal, expected outcome here, not an error
    condition — the caller decides what to do with a None (reject the
    login attempt), same as any other lookup that might reasonably find
    nothing.
    """
    with get_session() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None:
            return None
        return UserRecord(id=user.id, email=user.email, hashed_password=user.hashed_password)
