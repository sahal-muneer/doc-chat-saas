"""
rate_limit.py
=============

WHY THIS FILE EXISTS:

Nothing in this project so far stops one client from calling any endpoint
as fast as it possibly can. That's a real gap in two different ways: /chat
and /chat/stream each trigger a full LLM generation call to Qwen2.5 —
genuinely expensive on this hardware, not a cheap database read — so one
caller hammering them can starve everyone else. And /auth/login has zero
protection against someone scripting thousands of password guesses per
minute against it (a "brute-force" or "credential-stuffing" attack). Rate
limiting closes both gaps: it caps how many requests a given caller can
make in a time window, rejecting anything past that with 429 ("Too Many
Requests") instead of serving it.

WHY slowapi (A LIBRARY), NOT HAND-ROLLED REDIS COUNTERS: per CLAUDE.md's
"backend plumbing can lean on existing tools" principle — the same
reasoning database.py gives for using SQLAlchemy instead of writing raw SQL
by hand — rate limiting has real, easy-to-get-subtly-wrong subtlety (race
conditions in a naive "read count, check, increment" sequence; which
time-window algorithm to use) that a well-tested library handles correctly
out of the box. slowapi is the standard FastAPI-oriented wrapper around the
`limits` package.

WHY REDIS-BACKED, NOT slowapi's DEFAULT IN-MEMORY STORAGE: in-memory
counters live inside ONE Python process's memory — fine as long as this API
only ever runs as a single `uvicorn` process (true right now), but silently
WRONG the moment it ever runs as multiple worker processes
(`uvicorn --workers 4`, or multiple containers behind a load balancer):
each process would count independently, so a "20 requests/minute" limit
would actually allow 20 * (number of processes), with no error or warning
anywhere. Redis is already running in this project as Celery's broker (see
worker/celery_app.py) — pointing the limiter at that same instance means
every process shares ONE real, correct count, with no new infrastructure
to stand up.

WHY TWO DIFFERENT "WHO IS THIS REQUEST FROM" STRATEGIES (key functions):
most routes require login, so the fairest and most accurate way to identify
"who's calling" is the authenticated user_id — one heavy user shouldn't be
able to use up another user's quota just because they happen to share a
network/IP (e.g. two people on the same office wifi). But the two auth
routes (signup, login) are the ONE place a caller has no token yet by
definition — those fall back to the caller's IP address instead, which is
also exactly the property that matters most for blocking a brute-force
login attack in the first place.
"""

import os

import jwt
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth.security import decode_access_token

# Same "configuration layer" pattern as celery_app.py's own REDIS_URL —
# each file that needs Redis reads its own env var independently rather
# than importing a constant from an unrelated domain's file, matching how
# DATABASE_URL and JWT_SECRET are each defined locally where used.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def user_id_or_ip(request: Request) -> str:
    """
    The default key function for authenticated routes: identify the caller
    by their user_id — extracted straight from the JWT in the Authorization
    header — falling back to their IP address if no token is present or it
    doesn't verify. Reads the header directly off the raw Request rather
    than going through get_current_user_id (auth/dependencies.py): slowapi
    calls this function itself, outside FastAPI's normal dependency
    injection flow, so there's no Depends() machinery available to reuse
    here.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ")
        try:
            return decode_access_token(token)
        except jwt.PyJWTError:
            pass
    return get_remote_address(request)


# One shared Limiter instance — imported by main.py to register it on the
# app, and by every route file that applies an @limiter.limit(...)
# decorator to a specific endpoint.
limiter = Limiter(key_func=user_id_or_ip, storage_uri=REDIS_URL)
