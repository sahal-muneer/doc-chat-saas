"""
embedder.py
===========

WHY THIS FILE EXISTS:

pipeline.py turns a document into a list of Chunks — but a Chunk right now
is just text plus some bookkeeping (which document, which page, which
position). To actually SEARCH chunks later ("find the ones relevant to this
question"), we need to convert each chunk's text into an EMBEDDING: a list
of numbers (a "vector") that captures what the text MEANS, positioned in
space such that texts with similar meaning end up as vectors that are close
together. That conversion is this file's only job.

WHAT AN EMBEDDING ACTUALLY IS, IN PLAIN TERMS:

Imagine every possible sentence could be plotted as a single point in some
enormous space — not 2D or 3D like a graph you'd draw by hand, but with
hundreds or thousands of dimensions. A sentence about "refund policy" and
another sentence about "how to return a product" would land as two points
close together in that space, even though they don't share many of the same
words — because an embedding model is trained specifically to understand
meaning, not just match words. A sentence about "refund policy" and one
about "server maintenance schedule" would land far apart. Once every chunk
in your database has a point in that space, answering "which chunks are
relevant to this question" becomes a geometry problem: embed the question
into that same space, and find whichever chunk-points are closest to it.
That's the entire mechanism behind vector search — no separate AI reasoning
step happens at search time, just distance calculations, which is why it's
fast even across huge documents.

WHY BAAI/bge-m3 SPECIFICALLY (per CLAUDE.md's tech stack):

- It's MULTILINGUAL. The MVP is English-only (per CLAUDE.md's current
  phase), but Arabic is planned for Phase 2. This matters more than it
  might seem: swapping embedding models later isn't like swapping out a
  function — every chunk already stored would need to be RE-EMBEDDED with
  the new model, because two different models' vector spaces aren't
  compatible with each other (a vector from model A means nothing when
  compared against a vector from model B). Picking a model that already
  supports the future requirement avoids that expensive rework later. This
  is the "configuration layer" principle from CLAUDE.md in action: decide
  the thing that's expensive to change, early.
- It's runnable fully locally — no API calls, which is non-negotiable per
  CLAUDE.md's "100% offline" requirement.

WHY THIS IS A SEPARATE MODEL FROM Qwen2.5 (the chat model, not used here):
bge-m3's only job is "text in, vector out" — it doesn't generate sentences,
answer questions, or "think." It's a much smaller, much faster, much more
specialized model than a chat LLM, purpose-built for exactly one thing:
producing good embeddings. Every chunk gets embedded once, at upload time;
Qwen2.5 only gets involved later, at question-answering time, and never
sees the embedding step at all.

REAL-WORLD LESSON, same shape as the tiktoken issue in chunker.py — worth
catching BEFORE it surprises you in production, not after: the first time
you ever load "BAAI/bge-m3" on a machine, the embedding library downloads
the actual model weights (~2GB) from Hugging Face's servers over the
internet. That part is a one-time cost, and was always understood.

A SECOND, MORE SUBTLE VERSION OF THE SAME LESSON, DISCOVERED LATER, LIVE,
WHILE TESTING THE CELERY WORKER: this docstring used to claim that once
the weights are cached, "every future load reads from disk, no network
needed." That was WRONG, and worth correcting honestly rather than quietly
fixing. By DEFAULT, huggingface_hub (which sentence-transformers uses
under the hood) makes real network requests to huggingface.co on EVERY
model load — a handful of HEAD requests checking "is there a newer version
of this file?" — even when the weights are already fully cached locally.
With no internet, these calls fail and it silently falls back to the
cache, which is WHY the earlier full offline test (Wi-Fi off, everything
worked) didn't catch this — "degrades gracefully when offline" quietly
masked "still tries to phone home when online." That's a real gap against
CLAUDE.md's "100% offline, zero external API calls at runtime" rule: with
Wi-Fi on, this app was making outbound requests to a third party on every
single model load, without that being a deliberate decision anyone made.

THE FIX, BELOW: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE, set BEFORE
sentence_transformers is even imported. This doesn't just make the network
call fail gracefully — it stops huggingface_hub from ever ATTEMPTING one,
which is both genuinely offline (not just offline-tolerant) and faster
(no wasted round-trip / timeout before falling back to cache).

The remaining, separate, still-true point from before: for a real
"100% offline, on-premise" deployment, the deployment process needs to
either (a) pre-download the model weights into the deployment
image/environment during setup — before the product ever needs to serve a
real request — or (b) ship the weight files directly as build artifacts.
What must NOT happen is a customer's air-gapped, offline server trying to
download 2GB from the internet the first time someone uploads a document,
with no cache to fall back to at all.

WHY MODEL LOADING IS LAZY HERE, UNLIKE chunker.py's TIKTOKEN LOADING:
chunker.py loads its tiktoken encoding EAGERLY — the moment chunker.py is
imported, before any function in it is even called. That's fine there
because tiktoken's encoding table is tiny (a few megabytes). bge-m3's
weights are roughly 2GB and can take real, noticeable time to load into
memory. If THIS file loaded the model eagerly at import time, then simply
writing `import embedder` anywhere in the codebase — even in a test file
that has nothing to do with embeddings — would pay that enormous cost
immediately and unconditionally. Instead, _get_model() below loads the
model on the FIRST actual call to embed_chunks(), and keeps it cached in
memory for every call after that. Same underlying idea as chunker.py's
"load once, reuse many times" pattern — just triggered lazily, at first
use, rather than eagerly, at import time — because the cost/benefit
calculus changes when the resource being loaded is 1000x heavier.
"""

import os
from typing import Optional

# MUST run before `from sentence_transformers import SentenceTransformer`
# below — huggingface_hub reads these environment variables once, at
# import time, to decide whether it's allowed to make network calls at
# all. Setting them after the import would be too late; the library would
# already have made its "am I online?" decision. See the module docstring
# above for the full story of why this exists.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import Chunk

# Named constant instead of a magic string buried inside a function — and,
# per CLAUDE.md's "configuration layer" principle, this is exactly the kind
# of value a real production build would pull from an environment variable
# or config file rather than hardcoding, so it can be swapped (e.g. for
# testing with a smaller/faster model) without editing code. A single
# module-level constant is the reasonable middle ground for an MVP.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# bge-m3 produces 1024-dimensional vectors — i.e. each embedding is a list
# of exactly 1024 numbers. This isn't used anywhere in THIS file, but it's
# recorded here because the next piece of the pipeline (the pgvector
# database schema, coming up when we build storage) needs to declare a
# vector column with a fixed size — e.g. `VECTOR(1024)` — and it's much
# safer for that number to live in one named place than to be a "magic 1024"
# copy-pasted into a schema file with no explanation of where it came from.
EMBEDDING_DIMENSIONS = 1024

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Load bge-m3 on first use, then reuse the same loaded model for every
    later call. See the module docstring for why this is lazy rather than
    eager, unlike chunker.py's tiktoken loading."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Compute an embedding for every Chunk in the list, filling in each
    Chunk's `.embedding` field in place. Also returns the same list, purely
    for convenience (`chunks = embed_chunks(chunks)` reads nicely at a call
    site, even though it's the same objects being mutated, not new ones).

    WHY WE EMBED ALL CHUNKS IN ONE CALL, NOT ONE CHUNK AT A TIME: the model
    can process many pieces of text at once far more efficiently than it can
    process them one-by-one in a loop, because the underlying computation
    (matrix operations on a GPU or CPU) is built to work on batches of
    inputs simultaneously. Calling `model.encode()` once with a list of 50
    chunk texts is dramatically faster than calling it 50 separate times
    with one chunk text each — the same total amount of "work" happens far
    more efficiently when it's handed over all at once. This matters in
    real usage: a single uploaded document can easily produce dozens of
    chunks, and this function will get called once per document upload.
    """
    if not chunks:
        # Guard against an empty list BEFORE calling _get_model(). If we
        # didn't check this first, embedding an empty document would still
        # trigger the (potentially slow, first-time-only) model load for no
        # reason — wasted work for an input that has nothing to embed.
        return chunks

    model = _get_model()
    texts = [chunk.text for chunk in chunks]

    # normalize_embeddings=True scales every output vector to length 1
    # (a "unit vector"). Why this matters: there are two common ways to
    # measure how similar two vectors are — cosine similarity (which cares
    # only about the ANGLE between two vectors, ignoring their length) and
    # dot product (which is cheaper to compute, but is affected by both
    # angle AND length). If every vector is already normalized to the same
    # length, dot product and cosine similarity become mathematically
    # equivalent — so normalizing here lets the database use the faster
    # dot-product comparison later (in pgvector) while still getting
    # cosine-similarity-style results. Doing it once here, at embedding
    # time, is simpler than re-normalizing every vector on every search.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    for chunk, vector in zip(chunks, vectors):
        # model.encode() returns a numpy array, not a plain Python list.
        # .tolist() converts it — numpy arrays don't serialize cleanly to
        # JSON or into a database driver's expected input type, while a
        # plain list[float] just works everywhere without any numpy-
        # specific handling needed downstream (in storage code, in an API
        # response, anywhere). Doing the conversion once, here, means
        # nothing downstream ever has to think about numpy at all.
        chunk.embedding = vector.tolist()

    return chunks


def embed_query(text: str) -> list[float]:
    """
    Embed a single piece of text — in practice, a user's chat question, at
    RETRIEVAL time rather than ingestion time — and return the raw vector.

    WHY THIS IS SEPARATE FROM embed_chunks(): embed_chunks() takes a list of
    Chunk objects and mutates them in place, because at ingestion time we
    always have many chunks to embed together (see the batching comment on
    embed_chunks() above). At query time, there's exactly one input — the
    question the user just typed — and no Chunk object to attach it to; the
    caller (retriever.py) just needs the raw vector to search with. Reusing
    embed_chunks() here would mean wrapping a plain string in a fake Chunk
    just to satisfy its signature, which would be more confusing, not less.

    CRITICALLY, THIS MUST USE THE SAME MODEL (bge-m3) AS embed_chunks(): a
    question's vector and a chunk's vector are only comparable if they were
    both produced by the same model, in the same vector space. This is why
    _get_model()'s cached singleton matters here too — the query embedding
    and every chunk embedding it gets compared against were computed by the
    literal same loaded model instance.
    """
    model = _get_model()
    vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()


if __name__ == "__main__":
    # Manual test: embed a couple of short, deliberately-related-in-meaning
    # sentences and confirm the resulting vectors actually behave the way
    # embeddings are supposed to — similar meaning should mean small
    # "distance" between the vectors, even when the sentences don't share
    # many of the same words.
    demo_chunks = [
        Chunk(text="The refund policy allows returns within 30 days.", document_id="demo", chunk_index=0),
        Chunk(text="You can send a product back for a full reimbursement within a month.", document_id="demo", chunk_index=1),
        Chunk(text="The server maintenance window is scheduled for Sunday night.", document_id="demo", chunk_index=2),
    ]

    embed_chunks(demo_chunks)

    for c in demo_chunks:
        print(f"chunk {c.chunk_index}: {c.text!r}")
        print(f"  embedding length: {len(c.embedding)}  (expect {EMBEDDING_DIMENSIONS})")
        print(f"  first 5 values: {c.embedding[:5]}")
        print()

    # Cosine similarity between two already-normalized vectors is just
    # their dot product (see the normalize_embeddings comment above) —
    # this is a small, self-contained proof of that, without needing
    # numpy or any extra library just for this demo.
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    sim_related = dot(demo_chunks[0].embedding, demo_chunks[1].embedding)
    sim_unrelated = dot(demo_chunks[0].embedding, demo_chunks[2].embedding)

    print(f"similarity(refund policy, return for reimbursement) = {sim_related:.4f}")
    print(f"similarity(refund policy, server maintenance)       = {sim_unrelated:.4f}")
    print("(expect the first number noticeably higher than the second — "
          "that's the model correctly recognizing the first pair means "
          "roughly the same thing, despite sharing almost no words)")
