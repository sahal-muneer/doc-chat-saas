"""
history.py
==========

WHY THIS FILE EXISTS:

Every /chat request has, until now, been treated as if it's the very first
message ever sent — retriever.py and generator.py both take just
(document_id, query) with no notion of what was asked a moment ago. This
file is what makes a conversation actually a CONVERSATION: it persists each
turn (both the user's question and the model's answer) as a `Message` row
(models.py), and reads them back so api/chat.py can hand prior turns to the
model on the next request.

WHY A PLAIN messages TABLE, NOT A SEPARATE conversations ENTITY: see
models.py's Message docstring — this product only has one chat thread per
(user, document) right now, so document_id + user_id is already a complete,
unique key for "this conversation." Inventing a conversation_id on top of
that would be structure with no current use.

WHY get_messages() TAKES A limit, RETURNING ONLY THE MOST RECENT N ROWS —
NOT THE ENTIRE HISTORY, EVERY TIME: two different callers need two different
amounts of history. The frontend, loading a document you've chatted with
before, wants to show the WHOLE conversation (limit=None). But every prompt
sent to Qwen2.5 has a real, finite context window — and unlike the frontend
display, that context also has to fit the retrieved excerpts and the
question itself. Sending fifty turns of history into every single request,
forever, would eventually crowd out the excerpts entirely (the actual
grounding data) or just silently blow past what the model can attend to.
api/chat.py caps this to a handful of recent turns for anything that
actually goes into a prompt — recent context, not the full transcript,
which is exactly what a human re-joining a long conversation does too:
skims the last few messages, not the entire history, to pick up the thread.
"""

from sqlalchemy import select

from app.db.database import get_session
from app.db.models import Message


def save_message(document_id: str, user_id: str, role: str, content: str) -> None:
    """
    Persist one turn. Called twice per chat request — once for the user's
    question (before generation starts), once for the assistant's answer
    (after it finishes) — so a request that fails partway through generation
    still leaves the user's question in history rather than silently
    dropping it.
    """
    with get_session() as session:
        session.add(
            Message(document_id=document_id, user_id=user_id, role=role, content=content)
        )
        session.commit()


def get_messages(document_id: str, user_id: str, limit: int | None = None) -> list[dict]:
    """
    Return this (document, user)'s conversation, OLDEST FIRST — the order a
    prompt (and a chat UI) actually wants to read it in. When `limit` is
    given, this returns the most recent `limit` rows, not the first `limit`
    — "the last few things said," not "the first few things said years
    ago." That means the query has to sort newest-first to apply the LIMIT
    correctly, then reverse the Python list back into oldest-first order
    before returning it.

    SCOPED TO user_id, NOT JUST document_id: same authorization reasoning
    as ingestion/store.py's list_documents() — without this, one user could
    read a completely different user's chat history for a document, if
    document-level ownership were ever the only check performed. In
    practice api/chat.py's _ensure_owner() already blocks a non-owner from
    reaching this function at all, but scoping the query itself is a second,
    independent layer of the same guarantee — a mistake in one place
    doesn't automatically become a data leak.
    """
    with get_session() as session:
        query = (
            select(Message)
            .where(Message.document_id == document_id, Message.user_id == user_id)
            .order_by(Message.created_at.desc())
        )
        if limit is not None:
            query = query.limit(limit)

        rows = session.scalars(query).all()

        return [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in reversed(rows)
        ]
