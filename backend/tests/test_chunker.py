"""
test_chunker.py
================

WHY THIS FILE EXISTS: chunk_text() is the single biggest lever on RAG
answer quality (see chunker.py's own module docstring) — and, like
security.py, it's pure: given the same text and settings, it always
produces the same chunks, with no database, network, or model call
involved. That makes it fully testable without any mocking, and worth
testing directly given how much retrieval quality depends on it behaving
correctly.
"""

from app.ingestion.chunker import chunk_text, count_tokens


def test_chunking_stamps_document_id_and_page_number_onto_every_chunk():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, document_id="doc-abc", page_number=3, chunk_size_tokens=600)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "doc-abc"
        assert chunk.page_number == 3


def test_chunk_index_starts_at_start_index_and_increases_sequentially():
    # A small chunk_size_tokens forces MULTIPLE chunks out of this text, so
    # there's actually a sequence of indices to check, not just one.
    text = "\n\n".join(f"This is paragraph number {i}, with some extra words to add length." for i in range(20))
    chunks = chunk_text(text, document_id="doc-abc", chunk_size_tokens=30, overlap_tokens=5, start_index=10)

    assert len(chunks) > 1
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(10, 10 + len(chunks)))


def test_chunks_stay_close_to_the_requested_token_size():
    # Not an exact bound (the greedy packing algorithm can slightly overshoot
    # on a single oversized piece) — but no chunk should be wildly larger
    # than what was asked for, which is the property that actually matters
    # for staying inside an LLM's context window.
    text = "\n\n".join(f"This is paragraph number {i}, with some extra words to add length." for i in range(20))
    chunks = chunk_text(text, document_id="doc-abc", chunk_size_tokens=50, overlap_tokens=10)

    for chunk in chunks:
        assert count_tokens(chunk.text) <= 50 * 1.5


def test_overlap_carries_trailing_content_into_the_next_chunk():
    # Overlap exists specifically so a sentence near a chunk boundary isn't
    # lost from BOTH chunks — the concrete, checkable version of that is:
    # the end of chunk N and the start of chunk N+1 should share content.
    text = "\n\n".join(f"Paragraph {i} has unique content about topic {i}." for i in range(10))
    chunks = chunk_text(text, document_id="doc-abc", chunk_size_tokens=20, overlap_tokens=8)

    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        tail_of_first = first.text[-20:]
        assert any(word in second.text for word in tail_of_first.split() if len(word) > 3)


def test_blank_text_produces_no_chunks():
    assert chunk_text("   \n\n  ", document_id="doc-abc") == []
