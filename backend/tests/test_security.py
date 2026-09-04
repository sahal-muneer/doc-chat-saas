"""
test_security.py
=================

WHY THIS FILE EXISTS, AND WHY THESE TESTS NEED NO MOCKING AT ALL:
hash_password/verify_password (Argon2, via argon2-cffi) and
create_access_token/decode_access_token (JWT, via pyjwt) are both PURE —
same input always produces a checkable, deterministic outcome, and neither
one talks to a database, the network, or anything outside this process.
That makes them the simplest, most valuable kind of test to have: fast,
100% reliable, and they test our OWN logic directly rather than some
external service's behavior.
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.auth.security import (
    JWT_ALGORITHM,
    JWT_SECRET,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_does_not_store_the_plain_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    # Argon2's own hash format always starts with this — a real assertion
    # that Argon2 specifically ran, not just "some string came back."
    assert hashed.startswith("$argon2")


def test_verify_password_accepts_the_correct_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", hashed) is True


def test_verify_password_rejects_the_wrong_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("wrong-password", hashed) is False


def test_access_token_roundtrip_returns_the_same_user_id():
    token = create_access_token(user_id="user-123")
    assert decode_access_token(token) == "user-123"


def test_decode_access_token_rejects_an_expired_token():
    # Hand-crafted with an already-past "exp" claim — the same technique as
    # security.py's own __main__ demo — to prove decode_access_token()
    # actually enforces expiry, rather than just trusting the signature.
    expired_payload = {
        "sub": "user-123",
        "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
    }
    expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired_token)


def test_decode_access_token_rejects_a_tampered_signature():
    token = create_access_token(user_id="user-123")
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")

    with pytest.raises(jwt.PyJWTError):
        decode_access_token(tampered)
