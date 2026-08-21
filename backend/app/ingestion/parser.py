"""
parser.py
=========

WHY THIS FILE EXISTS (read this before the code):

chunker.py turns raw text into retrievable Chunks — but it has to be handed
raw text first, and it needs to know what page that text came from (see
Chunk.page_number in chunker.py, which exists specifically to support
"source citation" answers like "see page 12 of quarterly_report.pdf"). This
file is the step BEFORE chunking: given an uploaded PDF or DOCX file, pull
the text back out of it, one page at a time.

"One page at a time" is the key design choice here, and it's worth being
explicit about why. We could extract one giant string per document and hand
that whole thing to chunk_text() once. That would be simpler code. But then
every Chunk produced would have no idea which page it came from — the
page_number field on Chunk would just be a guess or a null. Citing a source
is a core requirement of a document Q&A product, so instead: extract text
page by page, and the ingestion pipeline calls chunk_text() once PER PAGE,
passing that page's number through as page_number=. A chunk that happens to
span a page boundary in the original document gets attributed to whichever
page it was extracted from — an acceptable approximation for citation
purposes, and far better than no page number at all.

OFFLINE CONSTRAINT CHECK (per CLAUDE.md — this product makes zero external
API calls at runtime): both libraries used here are safe. PyMuPDF is a
compiled C library (MuPDF) linked directly into the Python extension module
— there is no network call anywhere in opening a file or extracting text.
python-docx just unzips the .docx (a .docx *is* a ZIP file full of XML) and
parses the XML with lxml, entirely local. Neither behaves like tiktoken,
which downloads its encoding table from an OpenAI-hosted URL on first use
(see chunker.py) — that was a real bug we had to design around; this file
doesn't have that problem, but it's exactly the kind of thing worth
double-checking and stating explicitly rather than assuming, given how
easily the tiktoken issue slipped through.

KEY DESIGN DECISIONS (and why):

- PDF AND DOCX ARE HANDLED BY TWO COMPLETELY DIFFERENT LIBRARIES, because
  they are fundamentally different kinds of files, not just two file
  extensions for "a document." A PDF is a FIXED-LAYOUT format: every page's
  exact size and content placement is baked into the file at save time, so
  "page 4" is a real, stable, file-level fact we can just read off. A DOCX
  is a REFLOWABLE format: Word does not store page boundaries in the file
  (with one narrow exception — see parse_docx() below). "Page 4" only comes
  into existence once a layout engine — Microsoft Word, LibreOffice — lays
  the content out for a specific page size, margin, and font, none of which
  python-docx does. That's not a bug in this file; it's an inherent property
  of the .docx format, and the docstring on parse_docx() explains exactly
  where that limitation shows up and what it costs us.

- WHY PyMuPDF (imported as `fitz`) FOR PDFS, instead of PyPDF2 or
  pdfplumber: PyMuPDF wraps MuPDF, a C library, so it's dramatically faster
  on large PDFs than the pure-Python parsers — this matters because parsing
  runs on every document a user uploads, not once at dev time like a
  one-off script. It's also more reliable than PyPDF2 at extracting text in
  correct reading order from multi-column layouts (PyPDF2 can interleave
  columns). pdfplumber is the better choice specifically for pulling
  structured TABLES out of PDFs — a reasonable upgrade to reach for later
  if table-heavy documents (financial statements, spec sheets) turn out to
  matter for this product — but it's slower, and we don't need that
  precision for a first version aimed at prose documents.

  REAL-WORLD LESSON, flagged the same way chunker.py flags the tiktoken
  download issue: PyMuPDF is licensed AGPL-3.0 (Artifex, the vendor, also
  sells a commercial license if you don't want to open-source your code).
  That's a non-issue for an internal tool or something you plan to
  open-source. For a closed-source commercial SaaS — which is exactly what
  this project is — AGPL is a real legal constraint, not a formality: it can
  require you to release your own source code. This needs a deliberate
  decision before shipping to production — either buy Artifex's commercial
  license, or swap this module's PDF backend for a permissively-licensed
  alternative such as pypdfium2 (Apache-2.0/BSD-3) or pdfplumber (MIT). We
  use PyMuPDF here because it's the clearest teaching example of a fast,
  accurate extractor, and CLAUDE.md's tech stack already names it — but
  don't let this ship without resolving the license question on purpose.

- WHAT WE DELIBERATELY DO NOT HANDLE: scanned / image-only PDFs.
  PyMuPDF's get_text() reads whatever text LAYER is embedded in the PDF. If
  a page is actually a photograph or scan with no embedded text (common for
  old contracts, faxed documents, receipts), get_text() returns an empty
  string for that page — there's no text there to find, only pixels. We
  detect this (an empty string after stripping whitespace) and skip the
  page rather than returning a useless empty chunk. Recovering text from
  those pages requires OCR (e.g. Tesseract), which is a meaningfully
  different pipeline — image preprocessing, a model tuned for OCR-specific
  error patterns — and is out of scope for this file. If your users upload
  scanned documents, surface the skipped-page count somewhere so they don't
  silently lose content with no explanation.

- EMPTY PAGES ARE FILTERED OUT, not returned as (page_number, "") tuples.
  A genuinely blank page (inserted between sections, for print pagination)
  or the image-only case above would otherwise flow straight into
  chunk_text(), producing a zero-content Chunk that wastes storage and can
  never legitimately be a search result. Filtering once, here, is simpler
  and safer than making every downstream consumer defend against empty
  chunks.

- DOCX PAGE NUMBERS ARE BEST-EFFORT, not exact — read this before trusting
  them. The .docx XML format has exactly one explicit, file-level signal
  for "a new page starts here": a hard page break, `<w:br w:type="page"/>`,
  inserted whenever someone pressed Ctrl+Enter (or Word auto-inserts one
  before certain section breaks). Everything else — text simply running
  past the bottom margin because there was too much of it on the page — is
  COMPUTED AT RENDER TIME by Word's layout engine and is never written to
  the file at all. So parse_docx() below counts explicit page breaks only.
  For a typical document where the author never manually paginated (most
  documents — people insert hard breaks before new chapters/sections, not
  every time a page happens to fill up), this means most or all paragraphs
  will report as being on "page 1". That's not a bug we can fix by writing
  cleverer code — it's an inherent property of the file format, the same
  way tiktoken's download requirement in chunker.py isn't something you can
  code around, only work around. If exact DOCX page numbers matter for your
  product, the only fully correct fix is round-tripping the file through
  something that actually renders it (e.g. LibreOffice headless, exporting
  to PDF) and then using parse_pdf() on the result.
"""

from pathlib import Path

# REAL-WORLD LESSON, caught while testing this file (same spirit as the
# tiktoken lesson in chunker.py — flagged, not silently patched): PyMuPDF's
# import name used to be `fitz` (a holdover from before Artifex acquired the
# project), and every tutorial/StackOverflow answer still says
# `import fitz`. As of PyMuPDF 1.24+, the package is importable as
# `pymupdf` directly, and the old `fitz` name is kept only as a deprecated
# compatibility alias — `import fitz` now works but prints a
# DeprecationWarning on every run. We import the current name and alias it
# to `fitz` ourselves, so the rest of this file (and anyone who's used
# PyMuPDF before) can still read `fitz.open(...)` without the warning noise.
import pymupdf as fitz  # PyMuPDF — see import note above
from docx import Document
from docx.oxml.ns import qn

# A single page's worth of extracted text, paired with its 1-indexed page
# number. Notice this is a plain tuple, not a dataclass like Chunk in
# chunker.py — that's a deliberate difference, not an oversight. Chunk is a
# dataclass because it's a long-lived object: it gets embedded, stored in a
# vector database, and retrieved again minutes/days later. A PageText tuple
# is short-lived — it exists only to travel from this file to the ingestion
# loop that calls chunk_text() once per page, and then it's discarded. There
# is no value in the ceremony of a dataclass (named fields, validation,
# __post_init__) for something with a lifetime measured in one function call.
PageText = tuple[int, str]


def parse_pdf(file_path: str | Path) -> list[PageText]:
    """
    Extract text from a PDF, one (page_number, text) tuple per page.

    page_number is 1-indexed (page 1, not page 0) because that matches how
    humans — and the citations we show them — refer to pages. PyMuPDF's own
    internal indexing is 0-indexed (doc[0] is the first page), which is a
    common source of off-by-one page-number bugs if you forget to convert;
    the "+ 1" below is that conversion, done in exactly one place so it
    can't be forgotten anywhere else.
    """
    pages: list[PageText] = []

    # fitz.open() accepts a path directly (no need to read the file into
    # memory ourselves first) and figures out it's a PDF from the content,
    # not just the extension. Using it as a context manager (`with`, not a
    # bare assignment) guarantees document.close() runs on the way out —
    # whether the loop finishes normally or an exception propagates from a
    # malformed page — releasing the native file handle / memory buffer
    # PyMuPDF holds open. This is the same guarantee a manual try/finally
    # gives you, but fitz.Document implements __enter__/__exit__ itself, so
    # there's no reason to hand-roll it: always check whether a resource
    # already supports `with` before reaching for try/finally yourself.
    with fitz.open(file_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]

            # "text" mode returns plain reading-order text — good enough for
            # RAG, where we care about semantic content, not visual layout.
            # PyMuPDF also supports "blocks", "words", "dict" (structured,
            # position-aware) extraction modes for use cases that need
            # layout (e.g. rebuilding a table's rows/columns) — we don't
            # need that precision here, and requesting it would just be
            # extra work we then throw away.
            text = page.get_text("text").strip()

            # Skip pages with no extractable text (blank pages, or
            # image-only/scanned pages — see the module docstring for why
            # we don't attempt OCR here) rather than emitting an empty
            # chunk that can never be a useful search result.
            if not text:
                continue

            pages.append((page_index + 1, text))

    return pages


def parse_docx(file_path: str | Path) -> list[PageText]:
    """
    Extract text from a DOCX, one (page_number, text) tuple per page — where
    "page" means "the text between one explicit hard page break and the
    next." Read the DOCX PAGE NUMBERS ARE BEST-EFFORT section of the module
    docstring before relying on these numbers for anything precise: most
    real-world .docx files contain zero hard page breaks, in which case this
    function correctly (if unhelpfully) reports the entire document as
    page 1 — that's the file format's limitation, not a parsing failure.
    """
    document = Document(file_path)

    pages: list[PageText] = []
    current_page_number = 1
    current_page_lines: list[str] = []

    def flush_current_page() -> None:
        """Close out the page we've been accumulating and start a new one.

        Defined as a nested function (rather than inlined at both call
        sites below) so the "join lines, strip, skip-if-empty" logic exists
        in exactly one place — the same empty-page filtering rule as
        parse_pdf(), applied consistently across both file types.
        """
        text = "\n".join(current_page_lines).strip()
        if text:
            pages.append((current_page_number, text))
        current_page_lines.clear()

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            current_page_lines.append(paragraph.text)

        # A hard page break isn't a property of the paragraph as a whole —
        # it's a special <w:br w:type="page"/> element buried inside one of
        # the paragraph's runs (Word represents a page break as if you'd
        # typed an invisible character). python-docx's high-level API
        # doesn't expose "does this paragraph contain a page break" as a
        # simple property, so we have to drop down to the paragraph's raw
        # OOXML and look for it ourselves via each run's ._element.
        has_page_break = any(
            br.get(qn("w:type")) == "page"
            for run in paragraph.runs
            for br in run._element.findall(qn("w:br"))
        )

        if has_page_break:
            # SIMPLIFICATION, worth being explicit about: a page break can
            # occur mid-paragraph (rare, but Word allows it — e.g. someone
            # inserts a break in the middle of typing a sentence). We treat
            # the WHOLE paragraph's text as belonging to the page it started
            # on, rather than trying to split the paragraph's own text at
            # the exact run where the break occurs. This is the same kind
            # of "good enough, clearly flagged" approximation as
            # count_tokens()'s character-based fallback in chunker.py: the
            # imprecision is confined to the rare mid-paragraph-break case,
            # and getting it exactly right would add real complexity for a
            # case that's uncommon in practice.
            flush_current_page()
            current_page_number += 1

    # The last page's content never hits a page break (nothing comes after
    # it to trigger one) — it just ends when the document does. Flush
    # whatever's left in the buffer so we don't silently drop the final
    # page's text.
    flush_current_page()

    return pages


def parse_document(file_path: str | Path) -> list[PageText]:
    """
    Single entry point for the ingestion pipeline: look at the file
    extension and dispatch to the right extractor.

    We dispatch on the file extension rather than sniffing the file's magic
    bytes (the first few bytes that identify a PDF as "%PDF-" or a DOCX as
    a ZIP container's "PK\\x03\\x04" signature). Magic-byte sniffing is more
    robust — it would catch a PDF that got renamed to .docx by mistake — but
    it's also more code for a problem that mostly doesn't happen in
    practice: files arrive here because a user picked them in an upload
    dialog that already filtered by extension, or because our own storage
    layer already validated the type on upload. If mis-typed files turn out
    to be a real problem in production, this is the function to harden
    first — everything downstream already treats "which extractor to use"
    as a single decision made in one place.
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix == ".docx":
        return parse_docx(file_path)
    else:
        # Fail loudly and specifically rather than letting some downstream
        # library raise a confusing error two calls later. Whoever's
        # debugging a failed ingestion job should see immediately that the
        # problem is "unsupported file type," not "mysterious crash in
        # fitz.open()."
        raise ValueError(
            f"Unsupported file type: '{suffix}'. "
            "parse_document() only supports .pdf and .docx files."
        )


if __name__ == "__main__":
    # Quick manual test — run this file directly to see extraction (and the
    # DOCX page-break behavior in particular) in action, without needing any
    # sample files lying around. We build both a demo .docx and a demo .pdf
    # from scratch, the same way chunker.py's __main__ block builds a demo
    # string in-line rather than requiring a fixture file on disk.
    import tempfile

    tmp_dir = Path(tempfile.gettempdir())

    print("=== DOCX demo ===")
    docx_path = tmp_dir / "parser_demo.docx"
    demo_doc = Document()
    demo_doc.add_paragraph("This is page one of the demo document.")
    demo_doc.add_paragraph("Still page one — no break has happened yet.")
    # add_page_break() is python-docx's way of inserting the exact
    # <w:br w:type="page"/> element parse_docx() looks for — i.e. this is
    # doing the same thing as a user pressing Ctrl+Enter in Word.
    demo_doc.add_page_break()
    demo_doc.add_paragraph("This is page two, after the explicit break.")
    demo_doc.save(docx_path)

    for page_number, text in parse_docx(docx_path):
        print(f"--- page {page_number} ---")
        print(text)
        print()

    print("=== PDF demo ===")
    pdf_path = tmp_dir / "parser_demo.pdf"
    demo_pdf = fitz.open()  # opens a new, empty in-memory PDF
    for content in ("Page one content.", "Page two content."):
        page = demo_pdf.new_page()
        page.insert_text((72, 72), content)  # (x, y) in points from top-left
    demo_pdf.save(pdf_path)
    demo_pdf.close()

    for page_number, text in parse_pdf(pdf_path):
        print(f"--- page {page_number} ---")
        print(text)
        print()
