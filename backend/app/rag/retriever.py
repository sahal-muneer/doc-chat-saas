"""
retriever.py
============

WHY THIS FILE EXISTS:

Every chunk in Postgres already has an embedding (bge-m3, 1024 dimensions,
per embedder.py) sitting in a pgvector column, computed once at upload time.
This file is what actually USES those vectors: given a user's question and
a document to search within, it embeds the question with the same model and
asks pgvector "which stored chunks are closest to this question, in vector
space?" That's the entire "R" in RAG — Retrieval. Nothing in this file
generates an answer; it only finds the raw material (chunks of the original
document) that a later step (Ollama + Qwen2.5, not built yet) will actually
read and reason over.

WHY THIS LIVES IN A NEW app/rag/ FOLDER, NOT app/ingestion/: ingestion/ is
everything that happens ONCE, at upload time (parse -> chunk -> embed ->
store). This file runs at a completely different time — every time a user
asks a question, potentially long after the document was uploaded — and
does the reverse operation (read from the database, not write to it). Same
principle as store.py's docstring about keeping boundaries sharp: this is a
distinct enough responsibility, running at a distinct enough time, to earn
its own top-level package rather than being bolted onto ingestion/.
"""

from dataclasses import dataclass

from sqlalchemy import select

from app.db.database import get_session
from app.db.models import DocumentChunk
from app.ingestion.embedder import embed_query


@dataclass
class RetrievedChunk:
    """
    A plain, database-independent snapshot of one search result — deliberately
    NOT the SQLAlchemy DocumentChunk object itself. Same reasoning as
    store.py's create_document() gotcha: once the `with get_session()` block
    in search_chunks() below closes, a DocumentChunk object would become
    "detached" and unsafe to read from. Building this plain dataclass INSIDE
    the session block, then returning a list of THESE instead, means the
    caller (the /chat/search endpoint) can read .text, .distance, etc. long
    after the database session is closed, with no risk of a
    DetachedInstanceError.
    """

    chunk_index: int
    page_number: int | None
    text: str
    token_count: int
    distance: float


def search_chunks(document_id: str, query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Find the `top_k` chunks of `document_id` most semantically similar to
    `query`, ordered closest-first.

    HOW THE SEARCH ACTUALLY WORKS: `query` gets embedded into the exact same
    1024-dimensional vector space every stored chunk already lives in (via
    embed_query() — critically, the SAME model, bge-m3, as embedder.py used
    at ingestion time). `DocumentChunk.embedding.cosine_distance(query_vector)`
    is pgvector's SQLAlchemy integration generating the `<=>` operator —
    Postgres itself computes the distance between the query vector and every
    stored chunk's vector, entirely inside the database, using an index-
    friendly operation. We `.order_by()` on that distance (smaller = more
    similar, 0 = identical direction) and `.limit(top_k)` — so Postgres only
    has to hand back the handful of rows that actually matter, not every
    chunk in the document.

    WHY SCOPED TO ONE document_id: a real deployment could have thousands of
    documents from many users. Without the `.where(document_id == ...)`
    filter, a question about Document A could return chunks from a totally
    unrelated Document B just because they happened to be closer in vector
    space — clearly wrong for a "chat with THIS document" product. Scoping
    the search is what keeps retrieval relevant to what the user is actually
    looking at.
    """
    query_vector = embed_query(query)

    with get_session() as session:
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        rows = session.execute(
            select(DocumentChunk, distance.label("distance"))
            .where(DocumentChunk.document_id == document_id)
            .order_by(distance)
            .limit(top_k)
        ).all()

        return [
            RetrievedChunk(
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                text=chunk.text,
                token_count=chunk.token_count,
                distance=chunk_distance,
            )
            for chunk, chunk_distance in rows
        ]


if __name__ == "__main__":
    # Manual test: search the real document already sitting in Postgres from
    # the earlier Swagger upload test, using a question that's clearly
    # related to its content, and one that clearly isn't — same
    # "prove it with real output" pattern as every other file in this
    # project so far.
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m app.rag.retriever <document_id>")
        sys.exit(1)

    demo_document_id = sys.argv[1]

    for demo_query in [
        "What does this document say about HTTP servers?",
        "What's the weather like today?",
    ]:
        print(f"query: {demo_query!r}")
        results = search_chunks(demo_document_id, demo_query, top_k=3)
        if not results:
            print("  no chunks found for this document_id")
        for r in results:
            print(f"  distance={r.distance:.4f}  chunk_index={r.chunk_index}  {r.text[:80]!r}")
        print()
