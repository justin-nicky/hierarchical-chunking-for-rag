"""
pdf_parser.py
-------------
Extracts structured text from PDFs using PyMuPDF (fitz).

Output: a flat list of ParsedBlock objects, each tagged as either
  "heading" or "body".  The HierarchicalChunker reads this list and
  groups it into the two-level parent/child hierarchy.

Heading detection strategy (no ML required):
  1. Font size relative to the page's median body font size.
     Blocks whose font size >= median * HEADING_SIZE_RATIO are headings.
  2. Bold-only heuristic as a secondary signal.
  3. Very short lines (< MIN_HEADING_WORDS words) that are bold are
     treated as headings even if their font size is borderline.

This covers the vast majority of PDFs with conventional typesetting.
For PDFs that use colour or ALL-CAPS for headings, add those signals
to _classify_block() without touching the rest of the pipeline.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# ── Heading detection tunables ──────────────────────────────────────────────
HEADING_SIZE_RATIO = 1.15   # font size >= median_body_size * this → heading
MIN_HEADING_WORDS = 12      # bold lines shorter than this → always a heading
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ParsedBlock:
    """
    A single logical text block extracted from the PDF.
    Preserves enough information for metadata (page, offset, heading path).
    """
    text: str
    block_type: str          # "heading" or "body"
    page_num: int            # 0-indexed
    font_size: float
    is_bold: bool
    char_start: int          # character offset within this page's full text
    char_end: int


@dataclass
class ParsedDocument:
    """
    The normalised representation of an entire PDF.
    Consumed by HierarchicalChunker.
    """
    doc_id: str              # stem of the filename (caller can override)
    doc_path: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    page_count: int = 0
    title: str = ""          # from PDF metadata (may be empty)


class PDFParser:
    """
    Parses a PDF into a ParsedDocument using PyMuPDF.

    Usage:
        parser = PDFParser()
        doc = parser.parse("path/to/file.pdf")
    """

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        pdf = fitz.open(str(path))
        doc = ParsedDocument(
            doc_id=path.stem,
            doc_path=str(path.resolve()),
            page_count=pdf.page_count,
            title=pdf.metadata.get("title", "") or "",
        )

        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            blocks = self._extract_blocks(page, page_num)
            doc.blocks.extend(blocks)

        pdf.close()
        return doc

    # ── Private ─────────────────────────────────────────────────────────────

    def _extract_blocks(
        self, page: fitz.Page, page_num: int
    ) -> list[ParsedBlock]:
        """
        Extract text blocks from a single page.

        PyMuPDF's get_text("dict") returns a nested structure:
          page → blocks → lines → spans
        Each span has font, size, flags (bold = flags & 16).
        We flatten to block-level and classify as heading or body.
        """
        raw = page.get_text("dict")  # type: ignore[attr-defined]
        raw_blocks = raw.get("blocks", [])

        parsed: list[ParsedBlock] = []
        char_cursor = 0  # running offset within page text

        # ── Pass 1: collect all body font sizes for median computation ──
        all_sizes: list[float] = []
        for blk in raw_blocks:
            if blk.get("type") != 0:   # type 0 = text, type 1 = image
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    sz = span.get("size", 0)
                    if sz > 0:
                        all_sizes.append(sz)

        median_size = statistics.median(all_sizes) if all_sizes else 12.0

        # ── Pass 2: build ParsedBlock list ──────────────────────────────
        for blk in raw_blocks:
            if blk.get("type") != 0:
                continue

            block_text_parts: list[str] = []
            block_sizes: list[float] = []
            bold_span_count = 0
            total_spans = 0

            for line in blk.get("lines", []):
                line_parts: list[str] = []
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        line_parts.append(txt)
                    sz = span.get("size", 0)
                    if sz > 0:
                        block_sizes.append(sz)
                    flags = span.get("flags", 0)
                    if flags & 16:   # bold flag in PyMuPDF
                        bold_span_count += 1
                    total_spans += 1

                if line_parts:
                    block_text_parts.append(" ".join(line_parts))

            text = "\n".join(block_text_parts).strip()
            text = self._clean_text(text)
            if not text:
                continue

            avg_size = statistics.mean(block_sizes) if block_sizes else median_size
            is_bold = (bold_span_count / total_spans) > 0.5 if total_spans else False
            block_type = self._classify_block(
                text, avg_size, is_bold, median_size
            )

            end = char_cursor + len(text)
            parsed.append(ParsedBlock(
                text=text,
                block_type=block_type,
                page_num=page_num,
                font_size=avg_size,
                is_bold=is_bold,
                char_start=char_cursor,
                char_end=end,
            ))
            char_cursor = end + 1   # +1 for the implied newline between blocks

        return parsed

    def _classify_block(
        self,
        text: str,
        font_size: float,
        is_bold: bool,
        median_size: float,
    ) -> str:
        """
        Returns "heading" or "body".

        Rules (applied in priority order):
          1. Very large font → always a heading.
          2. Bold + short line → heading (catches bold section labels).
          3. Anything else → body.
        """
        word_count = len(text.split())

        if font_size >= median_size * HEADING_SIZE_RATIO:
            return "heading"

        if is_bold and word_count <= MIN_HEADING_WORDS:
            return "heading"

        return "body"

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Light-touch normalisation:
        - Collapse multiple spaces (PDF extraction artefact)
        - Strip soft hyphens (U+00AD) inserted by hyphenation engines
        - Normalise newlines
        """
        text = text.replace("\u00ad", "")          # soft hyphen
        text = re.sub(r"[ \t]+", " ", text)        # multiple spaces → one
        text = re.sub(r"\n{3,}", "\n\n", text)     # 3+ newlines → paragraph break
        return text.strip()
