"""
dependencies.py
================

WHY THIS FILE EXISTS:

security.py knows how to verify a JWT's signature and expiry, in plain
Python — it has no idea FastAPI, HTTP, or requests exist. This file is the
bridge: a FastAPI "dependency," which is FastAPI's mechanism for reusable
per-request logic. Any route that adds
`user_id: str = Depends(get_current_user_id)` to its function signature
automatically gets this function run FIRST — extracting and verifying the
token from the request's Authorization header, and either handing the
route a real user_id, or rejecting the request with 401 before the route's
own code ever runs at all. This is how "must be logged in" becomes a
one-line addition to any route, instead of copy-pasted token-checking code
in every protected endpoint.

WHY HTTPBearer, NOT FastAPI's OAuth2PasswordBearer (a more commonly-seen
example in FastAPI tutorials): OAuth2PasswordBearer models a specific flow
where Swagger's "Authorize" button posts a username/password to a
form-encoded token endpoint. Our actual login endpoint (api/auth.py) takes
JSON, not a form, and already returns a token directly — HTTPBearer matches
what's actually happening here: the client just sends
`Authorization: Bearer <token>`, and this dependency's only job is reading
and verifying that token. Swagger UI still gets a working "Authorize"
button either way, just one where you paste in a token you already
obtained from POST /auth/login, rather than typing a password into Swagger
itself.
"""

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.security import decode_access_token

_bearer_scheme = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """
    Extract and verify the JWT from this request's Authorization header,
    return the user_id it belongs to. Any route depending on this function
    can trust that, by the time its own code runs, `user_id` is a real,
    currently-logged-in user — invalid or expired tokens never reach that
    far, they get rejected right here with a 401 instead.
    """
    try:
        return decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        # Deliberately one generic message for every failure reason
        # (expired, tampered, malformed) — same "don't leak more detail
        # than necessary" instinct as login's shared error message in
        # api/auth.py. Telling a caller EXACTLY why their token failed
        # (e.g. "signature invalid" vs "expired") gives an attacker
        # probing the system more information than they need.
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
