"""
security.py
============

WHY THIS FILE EXISTS:

Two genuinely different jobs live here, both foundational to authentication:
(1) turning a plain password into something safe to store, and checking a
later login attempt against it, and (2) creating and verifying JWTs — the
"proof you're logged in" a client presents on every request after login.
Both are security-critical and easy to get subtly wrong, so they're kept in
one small, carefully-explained file rather than scattered across the auth
endpoints that use them.

PART 1 — PASSWORD HASHING, WHY ARGON2 (per CLAUDE.md's stack):

We NEVER store a user's real password anywhere — not encrypted, not
"protected," never. Instead, at signup, the password gets run through
Argon2, a ONE-WAY hashing function: easy to compute forward (password ->
hash), practically impossible to reverse (hash -> password). At login, we
don't "decrypt and compare" — we hash whatever was just typed and check if
THAT matches the stored hash. This means even if the entire `users` table
were ever leaked, an attacker still doesn't have anyone's actual password.

Argon2 specifically (not a faster hash like SHA-256) is deliberately SLOW
and MEMORY-HUNGRY, on purpose: it won the Password Hashing Competition
specifically for resisting large-scale cracking attempts. A fast hash lets
an attacker who steals the database try billions of guesses per second on
cheap hardware (a GPU); Argon2's cost makes that same attack impractically
slow, even though it adds a small, unnoticeable delay to a normal login.

PART 2 — JWTs, WHY THEY MAKE LOGIN "STATELESS":

HTTP requests don't inherently remember anything between them. After a
successful login, the server needs to hand back something the client can
present on every future request as proof of who they are. A JWT
(JSON Web Token) is a string with three dot-separated parts:
header.payload.signature. The payload is a JSON object — e.g.
{"sub": "<user_id>", "exp": <expiry timestamp>} — and it is NOT encrypted,
just base64-encoded, so anyone holding the token can read it. The actual
security is the SIGNATURE: the server signs the payload with a secret key
only it knows, and if anyone tampers with the payload afterward (e.g.
changing "sub" to someone else's user_id), the signature no longer matches
and the server rejects the token outright.

WHY THIS AVOIDS A DATABASE LOOKUP ON EVERY REQUEST: verifying a JWT's
signature is pure math (recompute the expected signature, compare) — no
query needed to confirm "is this session still valid," unlike a classic
server-side session-cookie approach, which has to check a sessions table
or Redis on every single request. The trade-off, worth knowing honestly:
a JWT can't be force-revoked early — once issued, it's valid until it
naturally expires, since the server never tracked it anywhere in the first
place. This is a real limitation, not a mistake; it's what "stateless"
actually costs.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Turn a plain password into an Argon2 hash, safe to store."""
    return _password_hasher.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Check a plain password against a stored hash. Returns True/False rather
    than letting the exception escape — callers (login) shouldn't need to
    know or care that argon2 signals "wrong password" via an exception
    internally; "did it match" is a plain yes/no question from their side.
    """
    try:
        _password_hasher.verify(hashed_password, password)
        return True
    except VerifyMismatchError:
        return False


# The secret key used to SIGN every JWT — this is what makes forging a
# valid token computationally infeasible without it. Reading it from an
# environment variable (with a dev-only fallback) is the same
# "configuration layer" principle as DATABASE_URL in database.py — except
# here, the stakes of hardcoding it are much higher: anyone who read this
# source file would be able to forge a valid login token for ANY user if
# the real secret were hardcoded. A real deployment must set JWT_SECRET to
# a long, random, actually-secret value — the fallback below exists purely
# so local development works out of the box.
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-secret-do-not-use-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = 60 * 24  # 24 hours


def create_access_token(user_id: str) -> str:
    """
    Build a signed JWT proving "this is user_id, issued by us, valid until
    exp." "sub" (short for "subject") and "exp" are STANDARD JWT claim
    names, not something made up here — jwt.decode() below specifically
    knows to check "exp" against the current time automatically and raise
    if it's passed, precisely because "exp" is the standardized name for it.
    """
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    """
    Verify a token's signature and expiry, and return the user_id it
    belongs to. Deliberately lets jwt's own exceptions (ExpiredSignatureError,
    InvalidSignatureError, etc.) propagate rather than swallowing them here
    — the caller (a FastAPI dependency, built next) is what actually needs
    to decide "an invalid token means reject this request with 401," and
    catching the specific exception type there produces a clearer error
    than converting everything into one generic failure this deep in the
    stack.
    """
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return payload["sub"]


if __name__ == "__main__":
    # Manual test, same "prove it with real output" pattern as every other
    # file: hash a password, confirm the right password verifies and the
    # wrong one doesn't, then create and decode a real token.
    hashed = hash_password("correct-horse-battery-staple")
    print(f"hash: {hashed[:50]}...")
    print(f"correct password verifies: {verify_password('correct-horse-battery-staple', hashed)}")
    print(f"wrong password verifies:   {verify_password('wrong-password', hashed)}")

    token = create_access_token(user_id="demo-user-123")
    print(f"\ntoken: {token}")
    print(f"decoded user_id: {decode_access_token(token)}")

    print("\nexpired token check:")
    try:
        expired_payload = {
            "sub": "demo-user-123",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
        }
        expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        decode_access_token(expired_token)
    except jwt.ExpiredSignatureError:
        print("  correctly rejected an expired token")
