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

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth.dependencies import get_current_user_id
from app.ingestion.store import get_document_owner
from app.rag.generator import generate_answer, stream_answer
from app.rag.retriever import search_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


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
def search(request: SearchRequest, user_id: str = Depends(get_current_user_id)) -> list[dict]:
    """
    Retrieval only — no LLM involved. Returns the raw chunks that would be
    handed to Qwen2.5 once generation exists, so you can inspect retrieval
    quality on its own. Requires login, and the document must belong to
    the caller (see _ensure_owner above).
    """
    _ensure_owner(request.document_id, user_id)
    results = search_chunks(request.document_id, request.query, request.top_k)
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
def chat(request: SearchRequest, user_id: str = Depends(get_current_user_id)) -> dict:
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
    """
    _ensure_owner(request.document_id, user_id)
    # Logging query_len (a number), not request.query itself (the actual
    # text): a user's questions could reasonably contain sensitive
    # information about the document they're asking about — logging isn't
    # a place to casually accumulate that. This still answers the useful
    # operational question ("is anyone sending huge queries?") without
    # capturing content that wasn't meant to be stored twice.
    logger.info(f"chat: user_id={user_id} document_id={request.document_id} query_len={len(request.query)}")
    chunks = search_chunks(request.document_id, request.query, request.top_k)
    answer = generate_answer(chunks, request.query)

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
def chat_stream(
    request: SearchRequest, user_id: str = Depends(get_current_user_id)
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
    _ensure_owner(request.document_id, user_id)
    logger.info(f"chat/stream: user_id={user_id} document_id={request.document_id} query_len={len(request.query)}")
    chunks = search_chunks(request.document_id, request.query, request.top_k)

    def event_stream():
        for token in stream_answer(chunks, request.query):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

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
