"""
celery_app.py
=============

WHY THIS FILE EXISTS:

This is where the Celery application object gets created — the equivalent
of main.py's `app = FastAPI(...)`, just for the background-worker side of
this project instead of the web-server side. Nothing about ingestion logic
lives here; this file's only job is CONFIGURING how Celery connects to
Redis.

THE FULL PICTURE — WHY A SECOND PROCESS AT ALL: up to now, this whole
project has run as ONE Python process: `uvicorn app.main:app` handles every
HTTP request AND does all the work for it, including the slow parts
(parsing a PDF, embedding chunks). That's exactly why a large upload makes
the browser wait — the same process that's supposed to be free to accept
the NEXT request is instead busy finishing the CURRENT one.

Celery's answer: run a SEPARATE, independent Python process — a "worker"
— whose only job is picking up units of work and executing them, with no
concept of HTTP requests at all. The web process and the worker process
don't call each other directly; they both talk to REDIS, which sits
between them purely as a message queue:

    FastAPI process                Redis                Celery worker
    ----------------                -----                -------------
    "process_document.delay(...)"    |                         |
         ---------------------> [message queued] ------------->|
    (returns immediately,                                 (picks up the
     doesn't wait for this)                                 message, runs
                                                             the real work,
                                                             whenever it's
                                                             free)

This is precisely the Redis/Celery architecture discussed conceptually
back when the ingestion pipeline was first built — this file is where it
actually becomes real, rather than a diagram.

WHY REDIS SPECIFICALLY, NOT JUST A DATABASE TABLE OF "PENDING JOBS": a
database table CAN work as a crude queue, but it requires something to
keep polling it ("has a new row appeared yet?"), which wastes effort and
adds latency. Redis is built for exactly this: a worker can efficiently
BLOCK and wait, woken up the instant a new message arrives, with
essentially no polling overhead and very low latency between "message
sent" and "worker picks it up."
"""

import os

from celery import Celery

from app.core.logging import setup_logging

# The Celery worker is a COMPLETELY SEPARATE process from
# `uvicorn app.main:app` — main.py's setup_logging() call never runs in
# this process, since this process never imports main.py at all. Without
# calling it again here, every logger.info()/logger.exception() call in
# tasks.py would use Python's unconfigured default logging behavior
# instead of the consistent timestamp/level/module format the rest of
# this project relies on.
setup_logging()

# Same "configuration layer" pattern as DATABASE_URL and JWT_SECRET: read
# from an environment variable with a sensible local-dev default, so a
# real deployment can point this at a different Redis instance (or a
# managed Redis service) without editing code.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# `celery_app.py` (this file's own name) is passed as the app name — used
# internally by Celery mostly for logging/naming, not something callers
# need to think about. broker=where task MESSAGES get sent; backend=where
# task RESULTS (return values, success/failure state) get stored — both
# point at the same Redis instance here since there's no reason to run two
# separate infrastructure pieces for an MVP.
celery_app = Celery(
    "doc_chat_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Without this, `celery_app` above exists, but Celery has no idea what
# TASKS (actual functions it's allowed to run) exist — this tells it to
# look inside app/worker/tasks.py and register everything decorated with
# @celery_app.task there.
celery_app.autodiscover_tasks(["app.worker"])
