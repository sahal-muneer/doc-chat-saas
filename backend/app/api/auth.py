"""
auth.py
=======

WHY THIS FILE EXISTS:

Exposes signup and login over real HTTP, and includes one small test route
(GET /auth/me) purely to prove get_current_user_id (dependencies.py) really
works — reading the token from Authorization, verifying it, and reporting
back which user it belongs to — before wiring that same dependency onto
the real routes (documents/chat) in a later step.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from slowapi.util import get_remote_address
from sqlalchemy.exc import IntegrityError

from app.auth.dependencies import get_current_user_id
from app.auth.security import create_access_token, verify_password
from app.auth.store import create_user, get_user_by_email
from app.core.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
@limiter.limit("5/minute", key_func=get_remote_address)
def signup(request: Request, body: SignupRequest) -> dict:
    """
    Create a new account and immediately log them in — return a real,
    usable access token in the same response, rather than making a brand
    new user make a second request to log in right after signing up.

    RATE-LIMITED BY IP, NOT user_id: there's no logged-in user yet at the
    moment of signing up — user_id doesn't exist until AFTER this succeeds
    — so IP is the only identity available to limit by here. See
    rate_limit.py's module docstring for the full reasoning. `request:
    Request` (the raw HTTP request slowapi needs to inspect) is a
    DIFFERENT object from `body: SignupRequest` (the parsed JSON body this
    route actually uses) — they used to share the name `request`, which
    would have silently broken slowapi's lookup for the real Request
    object; renaming the body param avoids that collision.
    """
    try:
        user_id = create_user(body.email, body.password)
    except IntegrityError:
        # Postgres's unique constraint on User.email is what actually
        # caught this — see store.py's create_user() docstring for why
        # that database-level guarantee matters more than an
        # application-level "check first" would.
        logger.warning(f"signup rejected: email already exists ({body.email})")
        raise HTTPException(
            status_code=400, detail="An account with this email already exists."
        )

    # Logging the EMAIL is fine — it's how a real admin identifies which
    # account this is. body.password and the returned token are never
    # logged, anywhere in this file — see logging.py's module docstring
    # for why that specific line can't be crossed even accidentally.
    logger.info(f"signup: new account created for {body.email} (user_id={user_id})")
    token = create_access_token(user_id)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login")
@limiter.limit("5/minute", key_func=get_remote_address)
def login(request: Request, body: LoginRequest) -> dict:
    """
    Verify email + password, return a fresh access token.

    WHY THE ERROR MESSAGE IS THE SAME WHETHER THE EMAIL DOESN'T EXIST OR
    THE PASSWORD IS WRONG: this is a deliberate security choice, not a
    missed detail. If "no account with that email" and "wrong password"
    gave different error messages, an attacker could use that difference
    to silently check which email addresses have accounts on this system
    at all (called "user enumeration") — a genuine, commonly-exploited
    information leak. One identical message for both cases closes that
    off entirely.

    RATE-LIMITED BY IP, 5/minute — THE MOST IMPORTANT LIMIT IN THIS WHOLE
    FILE: this is the one endpoint an unauthenticated attacker can call
    directly, and password-guessing (brute force / credential stuffing) is
    exactly the attack this specific limit exists to slow down to
    uselessness.
    """
    user = get_user_by_email(body.email)
    if user is None or not verify_password(body.password, user.hashed_password):
        # Same email logged for both failure reasons (unknown email, wrong
        # password) — logging can afford to be MORE detailed internally
        # than the error message shown to the caller (which deliberately
        # can't distinguish the two, per the docstring above); an admin
        # reading logs later isn't the enumeration risk, an external
        # attacker probing the API response is.
        logger.warning(f"login failed: {body.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    logger.info(f"login: {body.email} (user_id={user.id})")
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me")
def me(user_id: str = Depends(get_current_user_id)) -> dict:
    """
    A minimal proof that get_current_user_id actually works end to end:
    call this WITHOUT a token and get 401; call it WITH a valid token from
    /auth/login and get back the exact user_id that token belongs to. This
    route does nothing else useful on its own — it exists purely to verify
    the dependency before it gets relied on by real routes.
    """
    return {"user_id": user_id}
