"""
chunker.py
==========

WHY THIS FILE EXISTS (read this before the code):

An LLM can't "read" a whole document the way you do. Two hard limits force this:

1. CONTEXT WINDOW: every LLM has a max number of tokens it can accept per request
   (prompt + retrieved context + answer). A 200-page PDF will not fit.

2. RETRIEVAL PRECISION: even if it *did* fit, dumping the whole document into every
   prompt is wasteful and actually hurts accuracy — the model has to "search" through
   irrelevant text to find the answer, and irrelevant text in the prompt measurably
   degrades LLM answer quality (this is sometimes called "lost in the middle").

The fix: break each document into small, independently-searchable pieces called
CHUNKS. Each chunk gets converted to a vector (next file, embedder.py) and stored.
When a user asks a question, we don't search the whole document — we search chunks,
find the few most relevant ones, and only send THOSE to the LLM. This is the "R" in
RAG: Retrieval-Augmented Generation.

So chunking is not a formatting detail. It is the single biggest lever on answer
quality in a RAG system. Bad chunking = bad retrieval = bad answers, even with a
perfect LLM.

KEY DESIGN DECISIONS (and why):

- CHUNK SIZE: too small (e.g. one sentence) → chunks lack context, the embedding
  can't capture meaning well, and you need many chunks to answer one question.
  Too large (e.g. whole pages) → the embedding blurs together multiple topics,
  making similarity search less precise, and you waste LLM context space.
  We use ~500-800 tokens as a starting point — a reasonable paragraph-to-section
  size. This is a tunable parameter, not a fixed law — you should experiment on
  your own documents.

- OVERLAP: if we cut chunks with hard boundaries, a sentence that explains the
  answer might get split across two chunks and lose meaning in both. We overlap
  chunks by ~10-15% so context isn't lost at the boundary.

- SPLIT ON NATURAL BOUNDARIES (paragraphs/sentences), NOT FIXED CHARACTER COUNTS:
  cutting mid-sentence produces chunks that don't mean anything on their own. We
  recursively try to split on paragraph breaks first, then sentences, then words,
  only falling back to a hard cut as a last resort. This is the same strategy
  used by most production RAG chunkers (e.g. LangChain's RecursiveCharacterTextSplitter) —
  we're implementing the core idea ourselves so you understand what it's actually doing.

- TOKENS, NOT CHARACTERS: LLMs and embedding models operate on tokens, not
  characters. "chunk_size=500" in characters is a rough proxy at best, since one
  token is roughly ~4 characters in English (this ratio varies by language and
  content — code and Arabic text tokenize differently). For accurate sizing, you
  should tokenize text with the actual tokenizer of the model you're using rather
  than assume this ratio. We use a token-aware approach below via `tiktoken` (a
  real, open-source tokenizer library) as an approximation — good enough for
  chunk sizing, though it isn't the literal tokenizer of every model you might use.
"""

from dataclasses import dataclass, field
from typing import Optional
import re

# IMPORTANT REAL-WORLD LESSON, caught while testing this file:
# tiktoken's encoding tables are NOT bundled in the pip package — the first
# call to get_encoding() downloads them from an OpenAI-hosted URL and caches
# them locally. That means tiktoken silently requires internet access on
# first use. For your 100% offline product, that's a real problem: either
# you must pre-download and bundle the encoding file into your deployment
# image, or you should use a tokenizer that ships its vocabulary locally
# (e.g. the HuggingFace tokenizer that comes with whatever embedding/LLM
# model you actually deploy, such as Qwen2.5's own tokenizer). We fall back
# to the character-based estimate here so the demo still runs, but note this
# as a real decision point for the production ingestion pipeline.
try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, else fall back to a rough
    character-based estimate (~4 chars/token for English). The fallback is
    intentionally crude — flagged so you don't mistake it for an exact count."""
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    """A single retrievable unit of a document.

    We keep metadata alongside the text because retrieval isn't just about
    finding similar text — you need to be able to cite WHERE the answer came
    from (source document, page number) for the "source citations" feature
    from the CR. Losing this metadata at chunking time means you can never
    get it back later.
    """
    text: str
    document_id: str
    chunk_index: int          # position of this chunk within the document
    page_number: Optional[int] = None
    token_count: int = field(default=0)
    # Filled in later, by embedder.py — NOT at chunk-creation time. Chunking
    # and embedding are two separate stages of the pipeline (see pipeline.py
    # and, once it exists, embedder.py), so a freshly-created Chunk always
    # starts with embedding=None and gets it filled in as a second pass.
    # A list[float] rather than a numpy array or anything embedding-model-
    # specific, so this file (chunker.py) never has to know or care which
    # embedding model produced it, or import anything ML-related at all.
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = count_tokens(self.text)


# Boundaries to try splitting on, in priority order: prefer splitting on
# structure that preserves meaning (paragraphs) before falling back to
# something more destructive (arbitrary characters).
_SPLIT_PRIORITY = [
    "\n\n",   # paragraph breaks — best case, preserves full semantic units
    "\n",     # line breaks
    ". ",     # sentence boundaries
    " ",      # word boundaries — last resort before a hard character cut
]


def _split_text(text: str, separators: list[str]) -> list[str]:
    """Recursively split `text` using the first separator that actually
    appears in it, falling through to the next if not found. This is the
    core trick: we don't commit to one splitting strategy — we adapt to
    whatever structure the document actually has."""
    if not separators:
        return [text]

    sep = separators[0]
    if sep not in text:
        return _split_text(text, separators[1:])

    parts = text.split(sep)
    # Re-attach the separator (except to the last part) so we don't lose
    # punctuation/spacing when we reassemble chunks below.
    return [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def chunk_text(
    text: str,
    document_id: str,
    chunk_size_tokens: int = 600,
    overlap_tokens: int = 80,
    page_number: Optional[int] = None,
    start_index: int = 0,
) -> list[Chunk]:
    """
    Turn raw extracted text into a list of Chunk objects ready for embedding.

    Algorithm:
    1. Recursively split the text into small pieces using natural boundaries
       (paragraph > line > sentence > word).
    2. Greedily pack those pieces back together until adding the next piece
       would exceed chunk_size_tokens.
    3. Start the next chunk `overlap_tokens` worth of content before where
       the previous one ended, so context isn't lost at the seam.

    This is intentionally a from-scratch implementation rather than a call to
    a library, so the mechanics are visible. In production you might swap
    this for a maintained library, but the algorithm is the same idea.

    WHY start_index EXISTS: this function only ever sees ONE page's worth of
    text at a time — it has no idea a document has other pages, before or
    after. A real document gets ingested by calling this function once PER
    PAGE (see pipeline.py), so left to its own devices, every single page
    would produce chunks numbered starting at 0 — page 1's first chunk is
    chunk 0, but so is page 2's first chunk, and page 3's, and so on. That's
    wrong: chunk_index is supposed to be this chunk's position in the WHOLE
    document, not its position within just one page. Rather than have the
    caller reach into the returned Chunk objects afterward and overwrite
    chunk_index by hand (which would mean two different places in the
    codebase are responsible for deciding a chunk's index — a recipe for
    them drifting out of sync), we let the caller simply say "start counting
    from N this time." The caller tracks how many chunks it's produced so
    far across all pages, and passes that running total in as start_index
    for the next page. This function still owns the actual numbering logic;
    it's just been told where to begin.
    """
    pieces = _split_text(text, _SPLIT_PRIORITY)
    pieces = [p for p in pieces if p.strip()]

    chunks: list[Chunk] = []
    current_pieces: list[str] = []
    current_tokens = 0
    chunk_index = start_index

    i = 0
    while i < len(pieces):
        piece = pieces[i]
        piece_tokens = count_tokens(piece)

        if current_tokens + piece_tokens > chunk_size_tokens and current_pieces:
            # Current chunk is full — finalize it.
            chunk_text_value = "".join(current_pieces).strip()
            chunks.append(Chunk(
                text=chunk_text_value,
                document_id=document_id,
                chunk_index=chunk_index,
                page_number=page_number,
            ))
            chunk_index += 1

            # Build the overlap: walk backwards from the end of the current
            # chunk, keeping pieces until we've accumulated ~overlap_tokens.
            # This becomes the seed for the next chunk.
            overlap_pieces: list[str] = []
            overlap_count = 0
            for p in reversed(current_pieces):
                pt = count_tokens(p)
                if overlap_count + pt > overlap_tokens:
                    break
                overlap_pieces.insert(0, p)
                overlap_count += pt

            current_pieces = overlap_pieces
            current_tokens = overlap_count
            continue  # re-process the same piece against the reset window

        current_pieces.append(piece)
        current_tokens += piece_tokens
        i += 1

    if current_pieces:
        chunk_text_value = "".join(current_pieces).strip()
        if chunk_text_value:
            chunks.append(Chunk(
                text=chunk_text_value,
                document_id=document_id,
                chunk_index=chunk_index,
                page_number=page_number,
            ))

    return chunks


if __name__ == "__main__":
    # Quick manual test — run this file directly to see chunking in action
    # before we wire it into the full pipeline.
    sample = """Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with text generation. Instead of relying solely on what
a language model learned during training, RAG retrieves relevant text from an
external knowledge source at query time and includes it in the prompt.

This matters for offline, on-premise systems especially, because the LLM
itself may be relatively small and can't memorize every document a customer
uploads. RAG lets a small model answer accurately about documents it has
never seen during training, as long as the right chunk is retrieved.

The quality of retrieval depends heavily on how the source documents were
chunked. Chunks that are too large dilute the embedding signal. Chunks that
are too small lose context. Overlap between chunks helps prevent answers
from being split across a chunk boundary and lost."""

    result = chunk_text(sample, document_id="demo-doc", chunk_size_tokens=60, overlap_tokens=15)
    for c in result:
        print(f"--- Chunk {c.chunk_index} ({c.token_count} tokens) ---")
        print(c.text)
        print()
