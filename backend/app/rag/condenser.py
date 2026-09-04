"""
condenser.py
============

WHY THIS FILE EXISTS — THE PROBLEM GENERATION MEMORY DOESN'T SOLVE:

history.py + generator.py's updated build_messages() give the MODEL memory
of prior turns, which fixes things like "what did I just ask you?" But
retriever.py's search_chunks() still only ever sees the CURRENT question,
embedded on its own, with zero awareness of anything said before it. Ask
"what are the termination clauses?" and get an answer, then follow up with
"what about the deadlines mentioned there?" — that second question, embedded
in isolation, has no "there" to point to. Vector search has no way to know
"there" means "the termination clauses section" unless something makes that
explicit first. This is the actual, well-known gap in naive conversational
RAG: the LLM understands context, but the retriever is a completely
separate system that doesn't see any of it by default.

THE FIX — QUERY CONDENSATION (a.k.a. "history-aware retrieval"): before
retrieval runs, make one extra, fast LLM call whose only job is to rewrite
the latest question into a fully standalone one, using the conversation
history as context. "What about the deadlines mentioned there?" becomes
something like "What are the deadlines mentioned in the termination
clauses?" — a question that means the same thing with zero prior context
required. THAT rewritten question is what gets embedded and searched;
generation still receives the user's ORIGINAL wording (so the answer sounds
like it's responding to what was actually typed, not a robotic rewrite).

THE REAL COST, STATED PLAINLY: this is a second LLM round-trip on every
follow-up message — roughly doubling time-to-first-token on a 7B model
running locally, which isn't instant to begin with. That's a genuine
latency/quality trade-off, not a free improvement, which is exactly why
this only runs when there IS history (see condense_question() below) — the
very first question in any conversation has nothing to condense against,
so it skips this call entirely and goes straight to retrieval, same as
before conversational memory existed at all.
"""

import logging

import requests

from app.rag.generator import GENERATION_MODEL_NAME, OLLAMA_URL

logger = logging.getLogger(__name__)

_CONDENSE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question into a standalone question, using the "
    "conversation history for context, so it can be understood with no "
    "prior context at all. Preserve the original meaning and intent "
    "exactly — do not answer the question, do not add information that "
    "wasn't implied by the conversation, and do not explain your rewrite. "
    "If the follow-up question is already standalone, return it completely "
    "unchanged. Output ONLY the rewritten question, nothing else."
)


def condense_question(history: list[dict], question: str) -> str:
    """
    Rewrite `question` into a standalone version using `history` (the same
    oldest-first [{"role", "content"}, ...] shape history.get_messages()
    returns), or return it completely unchanged if there's no history to
    condense against — see the module docstring for why that early return
    matters, not just as an optimization but as the reason this never adds
    latency to the first message of a conversation.
    """
    if not history:
        return question

    messages = [{"role": "system", "content": _CONDENSE_SYSTEM_PROMPT}]
    messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    messages.append(
        {
            "role": "user",
            "content": f"Follow-up question: {question}\n\nStandalone question:",
        }
    )

    response = requests.post(
        OLLAMA_URL,
        json={"model": GENERATION_MODEL_NAME, "messages": messages, "stream": False},
    )
    response.raise_for_status()
    rewritten = response.json()["message"]["content"].strip()

    # Lengths only, not the actual text — same reasoning as api/chat.py's
    # query_len logging: a question (and its rewrite) can reasonably
    # contain sensitive information about the document being asked about,
    # so logs stay useful for "is this working at all?" without becoming a
    # second place that content quietly accumulates.
    logger.info(f"condensed question: {len(question)} chars -> {len(rewritten)} chars")
    return rewritten
