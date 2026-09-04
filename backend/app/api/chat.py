"""
chat.py
=======

WHY THIS FILE EXISTS:

retriever.py's search_chunks() is just a Python function — nothing outside
this codebase can call it directly. This file exposes retrieval over real
HTTP, the same way api/documents.py exposed ingestion, so it's actually
reachable from Swagger UI, a frontend, or curl.

WHY /chat/search STILL EXISTS NOW THAT /chat (BELOW) DOES REAL GENERATION:
/chat/search does ONLY retrieval — no LLM call, no waiting on Qwen2.5. It
stays useful precisely BECAUSE /chat exists now: it's a fast, direct way to
inspect what retrieval alone found, separate from whatever the LLM did with
it — genuinely useful for debugging "why did the chatbot answer that way?"
by checking whether the problem was bad retrieval (wrong chunks found) or
bad generation (right chunks, but a bad answer built from them).
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth.dependencies import get_current_user_id
from app.core.rate_limit import limiter
from app.ingestion.store import get_document_owner
from app.rag.condenser import condense_question
from app.rag.generator import generate_answer, stream_answer
from app.rag.history import get_messages, save_message
from app.rag.retriever import search_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# How many PRIOR turns get sent to the model as conversational context, on
# every request. Capped, not unbounded — see history.py's get_messages()
# docstring for why: excerpts + history + the question all have to fit in
# one finite context window, and an ever-growing, never-trimmed history
# would eventually crowd out the actual grounding data. 6 messages = 3
# user/assistant pairs — enough for "what about the deadlines mentioned
# there?" to resolve correctly, without letting a long-running conversation
# quietly bloat every single request forever.
HISTORY_LIMIT = 6


def _ensure_owner(document_id: str, user_id: str) -> None:
    """
    Reject the request if document_id doesn't belong to user_id — including
    when document_id doesn't exist at all. This is the actual authorization
    enforcement for every route below: get_current_user_id (a dependency)
    only proves WHO is asking; it says nothing about whether they're
    allowed to ask about THIS specific document. Without this check, any
    logged-in user who somehow learned another user's document_id (e.g.
    from a leaked URL) could still chat with a document that isn't theirs.

    WHY 404, NOT 403 ("Forbidden"), WHEN THE DOCUMENT BELONGS TO SOMEONE
    ELSE: a 403 would confirm "this document_id exists, you're just not
    allowed to see it" — leaking that the ID is real. 404 ("not found") is
    honestly indistinguishable from "this document_id was never real to
    begin with," which is exactly the same anti-enumeration reasoning
    behind login's identical error message for "wrong password" vs
    "no such email" in api/auth.py.
    """
    owner_id = get_document_owner(document_id)
    if owner_id is None or owner_id != user_id:
        # WARNING, not INFO: a request for a document_id that's either
        # fake or belongs to someone else is exactly the kind of thing
        # worth being able to spot in logs later — a burst of these from
        # one user_id could mean someone's actively probing for other
        # people's document IDs, not just an honest typo.
        logger.warning(f"denied: user_id={user_id} tried document_id={document_id} (not owner)")
        raise HTTPException(status_code=404, detail="Document not found.")


@router.get("/{document_id}/history")
def get_history(document_id: str, user_id: str = Depends(get_current_user_id)) -> list[dict]:
    """
    Return this (document, user)'s ENTIRE past conversation, oldest first —
    what the frontend calls when a document is selected, so reopening or
    refreshing the page shows the conversation that was already there
    instead of a blank chat window. Deliberately no `limit` here (unlike
    HISTORY_LIMIT's use inside chat()/chat_stream() below): what a HUMAN
    wants to see scrolling back through a chat is the real, full
    transcript; what get sent into a PROMPT is a separate, much smaller
    concern driven by the model's context window, not by what's useful to
    display.
    """
    _ensure_owner(document_id, user_id)
    return get_messages(document_id, user_id)


class SearchRequest(BaseModel):
    """
    FastAPI + Pydantic: declaring the expected shape of the request body as
    a class, instead of manually pulling fields out of raw JSON. FastAPI
    reads this class's type hints and does three things automatically: (1)
    validates incoming requests match this shape, rejecting anything that
    doesn't with a clear 422 error, (2) converts the validated JSON into a
    real Python object your function can use directly (request.query, not
    some_dict["query"]), and (3) uses this class to generate the "Try it
    out" form fields you see in Swagger UI — this is WHY Swagger already
    knows exactly what fields to show you, with no separate docs to write.
    """

    document_id: str
    query: str
    top_k: int = 5


@router.post("/search")
@limiter.limit("30/minute")
def search(request: Request, body: SearchRequest, user_id: str = Depends(get_current_user_id)) -> list[dict]:
    """
    Retrieval only — no LLM involved. Returns the raw chunks that would be
    handed to Qwen2.5 once generation exists, so you can inspect retrieval
    quality on its own. Requires login, and the document must belong to
    the caller (see _ensure_owner above).

    `request: Request` (the raw HTTP request, required for the
    @limiter.limit decorator above to identify the caller) and `body:
    SearchRequest` (the parsed JSON body this route actually reads) used to
    share the name "request" — that collision would have silently broken
    slowapi's lookup for the real Request object, so the body param is
    named `body` here and in every other route below.
    """
    _ensure_owner(body.document_id, user_id)
    results = search_chunks(body.document_id, body.query, body.top_k)
    return [
        {
            "chunk_index": r.chunk_index,
            "page_number": r.page_number,
            "text": r.text,
            "token_count": r.token_count,
            "distance": r.distance,
        }
        for r in results
    ]


@router.post("")
@limiter.limit("15/minute")
def chat(request: Request, body: SearchRequest, user_id: str = Depends(get_current_user_id)) -> dict:
    """
    The real thing: retrieval AND generation, in one call. Runs
    search_chunks() first, then hands its output straight into
    generate_answer() — the exact "retriever, then generator" order
    described earlier, just now happening inside one HTTP request instead
    of two separate manual steps. Requires login, and the document must
    belong to the caller (see _ensure_owner above).

    WHY THE RESPONSE INCLUDES "sources", NOT JUST "answer": returning the
    plain text answer alone would mean trusting the model with no way to
    check its work. Including the chunks that were actually fed into the
    prompt lets a caller (a frontend, or you testing in Swagger) see
    exactly what the model was working from — e.g. "page 2 of the
    document" — which is the beginning of real citations, and also the
    fastest way to debug a wrong answer: was retrieval wrong, or did the
    model misread correct chunks?

    RATE-LIMITED LOWER THAN /chat/search (15/minute vs 30/minute): this
    route, unlike /chat/search, ends in a real Qwen2.5 generation call —
    genuinely slow and resource-heavy on this hardware, and the actual
    thing rate limiting on this project exists to protect in the first
    place. See rate_limit.py's module docstring.
    """
    _ensure_owner(body.document_id, user_id)
    # Logging query_len (a number), not body.query itself (the actual
    # text): a user's questions could reasonably contain sensitive
    # information about the document they're asking about — logging isn't
    # a place to casually accumulate that. This still answers the useful
    # operational question ("is anyone sending huge queries?") without
    # capturing content that wasn't meant to be stored twice.
    logger.info(f"chat: user_id={user_id} document_id={body.document_id} query_len={len(body.query)}")

    # Fetched BEFORE this turn's question is saved below — otherwise the
    # current question would show up twice: once as "the question," once
    # again inside its own history.
    history = get_messages(body.document_id, user_id, limit=HISTORY_LIMIT)
    retrieval_query = condense_question(history, body.query)
    chunks = search_chunks(body.document_id, retrieval_query, body.top_k)

    save_message(body.document_id, user_id, "user", body.query)
    # Generation still sees the user's ORIGINAL wording, not the condensed
    # retrieval query — condense_question()'s rewrite exists purely to make
    # vector search work; the answer should read like a response to what
    # was actually typed.
    answer = generate_answer(chunks, body.query, history)
    save_message(body.document_id, user_id, "assistant", answer)

    return {
        "answer": answer,
        "sources": [
            {
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "text": c.text,
            }
            for c in chunks
        ],
    }


@router.post("/stream")
@limiter.limit("15/minute")
def chat_stream(
    request: Request, body: SearchRequest, user_id: str = Depends(get_current_user_id)
) -> StreamingResponse:
    """
    Same retrieval + generation as POST /chat, but the response starts
    arriving the moment the FIRST token is generated, instead of only
    after the ENTIRE answer is ready. This is the endpoint the real
    frontend uses; POST /chat above stays useful for quick testing in
    Swagger, since Swagger's "Try it out" just shows a stream's raw
    concatenated bytes anyway, not a live-updating view.

    WHY THIS RETURNS StreamingResponse(event_generator(), ...) INSTEAD OF
    A NORMAL dict: a normal FastAPI route returns one complete value,
    which FastAPI serializes to JSON and sends as a single response body.
    StreamingResponse instead takes a Python generator and sends each
    piece it yields to the browser AS SOON AS it's yielded — the request
    stays open, and data trickles out over time, rather than all at once
    at the end.

    THE WIRE FORMAT — Server-Sent Events (SSE): each event is plain text
    shaped like `data: <json>\\n\\n` (a "data:" prefix, then a blank line
    to mark the end of that event). This is a standard, simple format
    browsers know how to consume, even though — as flagged when this was
    planned — we're not using the browser's built-in EventSource here,
    because EventSource only supports GET requests and this needs a POST
    body (document_id, query). The frontend instead reads this same format
    manually.

    WHY chunks (RETRIEVAL) HAPPEN BEFORE THE STREAM STARTS, NOT DURING:
    search_chunks() has to finish first regardless — generation can't start
    without knowing what to put in the prompt. So the "sources" part of the
    response is already known before any streaming happens; it gets sent
    as one final event, after every token, purely so the frontend can
    render the answer text as it arrives without waiting on sources too.
    Requires login, and the document must belong to the caller (see
    _ensure_owner above) — checked BEFORE the StreamingResponse starts,
    so an unauthorized request gets a clean 404 instead of a stream that
    opens and then has nothing to send.
    """
    _ensure_owner(body.document_id, user_id)
    logger.info(f"chat/stream: user_id={user_id} document_id={body.document_id} query_len={len(body.query)}")

    # Same ordering reasoning as POST /chat above: fetch history, condense,
    # retrieve, THEN save this turn's question — so it isn't double-counted
    # inside its own history.
    history = get_messages(body.document_id, user_id, limit=HISTORY_LIMIT)
    retrieval_query = condense_question(history, body.query)
    chunks = search_chunks(body.document_id, retrieval_query, body.top_k)
    save_message(body.document_id, user_id, "user", body.query)

    def event_stream():
        # Streaming means the full answer never exists as one string here —
        # only as a sequence of small pieces yielded to the browser. To
        # save the complete answer to history once the stream finishes,
        # every piece has to be collected as it goes by, then joined at the
        # end — the same "accumulate, don't discard" idea as everywhere
        # else in this codebase, just applied to an HTTP response instead
        # of a database write.
        answer_parts: list[str] = []
        for token in stream_answer(chunks, body.query, history):
            answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        save_message(body.document_id, user_id, "assistant", "".join(answer_parts))

        sources = [
            {
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "text": c.text,
            }
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
