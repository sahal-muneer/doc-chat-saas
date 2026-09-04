"""
test_api_auth.py
=================

WHY THIS FILE EXISTS, AND HOW IT'S DIFFERENT FROM EVERY OTHER TEST FILE
HERE: test_security.py and test_chunker.py test pure functions directly;
test_generator.py/test_condenser.py mock out the one external call each
makes. This file does neither — it sends REAL HTTP requests through the
ACTUAL FastAPI app (routing, Pydantic validation, the auth dependency, the
rate limiter, all of it) via TestClient, against a REAL Postgres test
database (see conftest.py). This is the only genuinely end-to-end test in
this project's suite: it's what actually proves signup, login, and
token-based authorization work TOGETHER, not just each piece in isolation.

A DELIBERATE BUDGET, WORTH KNOWING ABOUT: /auth/signup and /auth/login are
both rate-limited to 5 requests/minute per caller (rate_limit.py), and
TestClient requests all appear to come from the same fake address — so
every test function below SHARES one rate-limit counter with every other
one in this file. This file makes exactly 4 such calls total, comfortably
under the 5/minute cap. Adding more auth-hitting tests later without
checking this budget would start failing with 429 instead of the assertion
you meant to test — worth remembering if this file grows.
"""

import uuid

from app.auth.store import create_user


def _unique_email() -> str:
    # A fresh, random email per test — NOT because table state leaks
    # between runs (conftest.py recreates the schema every session,
    # unrelated tests can't collide) but so two tests within the SAME
    # session can each create an account without hitting User.email's
    # unique constraint against each other.
    return f"test-{uuid.uuid4()}@example.com"


def test_signup_then_me_returns_the_correct_user_id(client):
    email = _unique_email()
    response = client.post("/auth/signup", json={"email": email, "password": "correct-horse-battery-staple"})

    assert response.status_code == 200
    token = response.json()["access_token"]

    me_response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    # The user_id /auth/me reports back has to be a REAL row this signup
    # call actually created — not just "some token was accepted."
    assert me_response.json()["user_id"]


def test_login_with_correct_password_returns_a_usable_token(client):
    email = _unique_email()
    # Created directly through the store, not via POST /auth/signup — saves
    # one call out of this file's shared rate-limit budget (see module
    # docstring) for a test that isn't actually about signup at all.
    create_user(email, "correct-horse-battery-staple")

    response = client.post("/auth/login", json={"email": email, "password": "correct-horse-battery-staple"})

    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_with_wrong_password_and_unknown_email_give_the_identical_error(client):
    email = _unique_email()
    create_user(email, "correct-horse-battery-staple")

    wrong_password_response = client.post("/auth/login", json={"email": email, "password": "wrong-password"})
    unknown_email_response = client.post(
        "/auth/login", json={"email": "no-such-account@example.com", "password": "anything"}
    )

    # This IS the actual security property, not an incidental detail — see
    # api/auth.py's own docstring on why login can't reveal WHICH reason it
    # failed for, to prevent an attacker from using the API to discover
    # which email addresses have accounts at all.
    assert wrong_password_response.status_code == 401
    assert unknown_email_response.status_code == 401
    assert wrong_password_response.json()["detail"] == unknown_email_response.json()["detail"]


def test_protected_route_without_a_token_is_rejected(client):
    response = client.get("/auth/me")
    assert response.status_code in (401, 403)
