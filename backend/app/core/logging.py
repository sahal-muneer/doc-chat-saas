"""
logging.py
==========

WHY THIS FILE EXISTS:

Every route so far has had exactly one way to tell what happened: whatever
happened to print to the terminal while uvicorn was running, live, in
front of you. That's fine while you're the one testing everything by hand
— it's completely unworkable once real users exist, because nobody's
watching the terminal at 3am when a document fails to ingest. LOGGING is
the fix: deliberately writing out "what happened, when, to whom" as
structured, searchable text, so a failure can be investigated AFTER the
fact, not just observed live.

WHY PYTHON'S BUILT-IN `logging` MODULE, NOT A THIRD-PARTY LIBRARY: this is
the same "don't reach for more machinery than the MVP needs" instinct as
create_tables() over Alembic — Python's standard library logging already
does everything this project actually needs (leveled messages, consistent
formatting, one place to configure it), and it's already installed
everywhere Python is. Structured JSON logging (where each log line is a
parseable JSON object, common in real production systems so log-analysis
tools can query fields like "show me every failed upload") is a genuine
upgrade path later — flagging it as a real trade-off, not silently
deciding it doesn't matter.

WHY LOGS GO TO STDOUT (THE CONSOLE), NOT A FILE ON DISK: this follows a
widely-used principle (from the "12-Factor App" methodology) — an
application shouldn't manage its own log FILES (rotating them, deciding
where they live, cleaning up old ones). It should just write logs to
stdout, and let WHATEVER IS RUNNING the app (a terminal now, but later
Docker, a cloud platform, etc.) decide what to do with that stream —
capture it, forward it to a log-aggregation service, store it. This is
exactly why `docker compose logs` already works for the Postgres
container without anything special configured for it: Postgres's own
process does this same thing.

A REAL SECURITY RULE THIS FILE'S CALLERS MUST FOLLOW: never log a
password, a full JWT, or anything else secret. It's easy to write
`logger.info(f"login attempt: {request}")` without thinking, and
accidentally write every user's password straight into a log file
forever. Every logging call added to auth.py below logs an EMAIL, never a
password or token — worth calling out explicitly, the same way security.py
flags Argon2's purpose, because "don't log secrets" is a rule easy to
violate by accident, not on purpose.
"""

import logging
import sys


def setup_logging() -> None:
    """
    Configure logging ONCE, at application startup (called from main.py,
    before anything else runs). Every other file just does
    `logger = logging.getLogger(__name__)` and calls logger.info(...) /
    logger.error(...) — they don't need to know or care how logging is
    configured, only that it IS, by the time they run.

    WHY __name__ AS THE LOGGER NAME, EVERYWHERE THIS PATTERN IS USED:
    `getLogger(__name__)` gives each file its own logger, automatically
    named after that file's module path (e.g. "app.api.chat"). This means
    every log line already says WHICH FILE produced it, for free, without
    manually typing a label into every message.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
