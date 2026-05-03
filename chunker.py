"""
chunker.py
----------
Two-level hierarchical chunker for PDFs.

LEVEL 0  →  Parent chunk  =  one logical section.
              Formed by grouping all body blocks between two headings.
              Max size: config.parent_max_tokens.
              Sent to the LLM as the generation context.

LEVEL 1  →  Child chunk  =  one sentence within a parent.
              Formed by sentence-splitting the parent's text.
              Max size: config.child_max_tokens.
              Embedded and stored in the vector DB.

Small-to-big retrieval pattern:
  query → embed → retrieve top-k Children
        → expand each child to its Parent via parent_id
        → feed Parent.text (or text_with_context) to the LLM.

Design notes:
  - Stateless: no instance state beyond the injected config/counter.
  - All complexity lives in private methods; chunk() is the public API.
  - Oversized parents are recursively split by paragraph.
  - Oversized children (single sentence > child_max_tokens) are
    hard-split with overlap as a last resort.
"""

from __future__ import annotations

import re
import uuid

from .config import ChunkerConfig
from .models import Chunk, ChunkLevel, ChunkMetadata, ChunkTree
from .pdf_parser import ParsedBlock, ParsedDocument
from .token_counter import TokenCounter


class HierarchicalChunker:
    """
    Converts a ParsedDocument into a two-level ChunkTree.

    Args:
        config:  ChunkerConfig controlling token limits and overlap.
        counter: TokenCounter instance (injected for testability).
    """

    def __init__(
        self,
        config: ChunkerConfig,
        counter: TokenCounter | None = None,
    ) -> None:
        self.config = config
        self.counter = counter or TokenCounter(config.encoding_name)

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk(self, doc: ParsedDocument) -> ChunkTree:
        """
        Main entry point. Returns a fully-populated ChunkTree.

        Steps:
          1. Group ParsedBlocks into parent-level sections.
          2. For each section, sentence-split into child chunks.
          3. Inject overlap context into child chunks.
        """
        tree = ChunkTree(doc_id=doc.doc_id, doc_path=doc.doc_path)
        sections = self._group_into_sections(doc.blocks)

        heading_path: list[str] = []

        for section in sections:
            heading = section["heading"]
            body_blocks = section["body"]

            # Update the running heading breadcrumb
            if heading:
                heading_path = self._update_heading_path(
                    heading_path, heading.text
                )

            # Combine body blocks into the parent text
            body_text = self._blocks_to_text(body_blocks)
            if heading:
                # Prepend the heading so the parent chunk is self-contained
                parent_text = f"{heading.text}\n\n{body_text}".strip()
            else:
                parent_text = body_text.strip()

            if not parent_text:
                continue

            # If body is too large, split at paragraph boundaries first
            parent_segments = self._split_parent_if_needed(parent_text)

            for seg_text in parent_segments:
                parent_meta = ChunkMetadata(
                    doc_id=doc.doc_id,
                    doc_path=doc.doc_path,
                    page_num=body_blocks[0].page_num if body_blocks else (
                        heading.page_num if heading else 0
                    ),
                    heading_path=list(heading_path),
                    char_start=body_blocks[0].char_start if body_blocks else 0,
                    char_end=body_blocks[-1].char_end if body_blocks else 0,
                    token_count=self.counter.count(seg_text),
                    content_hash="",   # filled in by Chunk.create()
                )
                parent = Chunk.create(
                    text=seg_text,
                    level=ChunkLevel.PARENT,
                    metadata=parent_meta,
                )
                tree.add_parent(parent)

                # ── Sentence-split parent into children ──────────────────
                sentences = self._split_into_sentences(seg_text)
                child_groups = self._group_sentences_into_children(sentences)

                for child_text in child_groups:
                    child_meta = ChunkMetadata(
                        doc_id=doc.doc_id,
                        doc_path=doc.doc_path,
                        page_num=parent_meta.page_num,
                        heading_path=list(heading_path),
                        char_start=0,   # sentence-level offsets are approximate
                        char_end=0,
                        token_count=self.counter.count(child_text),
                        content_hash="",
                    )
                    child = Chunk.create(
                        text=child_text,
                        level=ChunkLevel.CHILD,
                        metadata=child_meta,
                        parent_id=parent.id,
                    )
                    tree.add_child(child)

        # ── Overlap injection (post-processing pass) ─────────────────────
        self._inject_overlap(tree)

        return tree

    # ── Section grouping ────────────────────────────────────────────────────

    def _group_into_sections(
        self, blocks: list[ParsedBlock]
    ) -> list[dict]:
        """
        Splits the flat block list into sections, where each section is:
          { "heading": ParsedBlock | None, "body": [ParsedBlock, ...] }

        A new section starts whenever a heading block is encountered.
        The very first body blocks (before any heading) form a preamble section
        with heading=None.
        """
        sections: list[dict] = []
        current_heading: ParsedBlock | None = None
        current_body: list[ParsedBlock] = []

        for block in blocks:
            if block.block_type == "heading":
                # Save the previous section (if it has content)
                if current_body or current_heading:
                    sections.append({
                        "heading": current_heading,
                        "body": current_body,
                    })
                current_heading = block
                current_body = []
            else:
                current_body.append(block)

        # Don't forget the last section
        if current_body or current_heading:
            sections.append({
                "heading": current_heading,
                "body": current_body,
            })

        return sections

    # ── Parent sizing ────────────────────────────────────────────────────────

    def _split_parent_if_needed(self, text: str) -> list[str]:
        """
        If the parent text exceeds parent_max_tokens, split at paragraph
        boundaries.  Each resulting segment becomes its own parent chunk
        (and its own set of children).

        Falls back to token-level splitting as a last resort.
        """
        if self.counter.fits(text, self.config.parent_max_tokens):
            return [text]

        # Try paragraph splits first
        paragraphs = re.split(r"\n{2,}", text)
        segments: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.counter.count(para)
            if current_tokens + para_tokens > self.config.parent_max_tokens and current_parts:
                segments.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            segments.append("\n\n".join(current_parts))

        # Last resort: any segment still too large → token-level hard split
        final: list[str] = []
        for seg in segments:
            if not self.counter.fits(seg, self.config.parent_max_tokens):
                final.extend(
                    self.counter.split_with_overlap(
                        seg,
                        self.config.parent_max_tokens,
                        self.config.overlap_tokens,
                    )
                )
            else:
                final.append(seg)

        return final

    # ── Sentence splitting ───────────────────────────────────────────────────

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Rule-based sentence splitter — no NLTK dependency.

        Handles:
          - Standard period/exclamation/question terminators.
          - Abbreviations (Mr., Dr., U.S., Fig., etc.) → NOT split.
          - Decimal numbers (3.14) → NOT split.
          - Newlines within paragraphs treated as soft sentence breaks.
        """
        # Common abbreviations that end with a period but are NOT sentence ends
        abbrevs = {
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
            "fig", "eq", "no", "vol", "pp", "ed", "est", "approx",
            "dept", "avg", "max", "min", "e.g", "i.e", "al", "cf", "u.s", "u.k", "ph.d", "m.d", "b.sc",
        }

        # Insert a sentinel "|||" at sentence boundaries
        # Strategy: look for ". " or ".\n" patterns that are NOT abbreviations
        def _is_sentence_end(match: re.Match) -> str:
            before = match.group(1).lower().rstrip()
            last_word = re.split(r"\s+", before)[-1].rstrip(".")
            if last_word in abbrevs:
                return match.group(0)               # not a sentence end
            if re.match(r"^\d+$", last_word):
                return match.group(0)               # decimal number
            return match.group(0).rstrip() + "|||"  # inject sentinel

        # Replace newlines with spaces so the regex works uniformly
        flat = text.replace("\n", " ")
        # Annotate sentence boundaries
        annotated = re.sub(
            r"([\w\s,;:\"'()\-]+[.!?])\s+",
            _is_sentence_end,
            flat,
        )
        raw_sentences = [s.strip() for s in annotated.split("|||")]
        return [s for s in raw_sentences if s]

    def _group_sentences_into_children(
        self, sentences: list[str]
    ) -> list[str]:
        """
        Accumulate sentences into child chunks that stay within
        child_max_tokens.  A sentence that alone exceeds the limit is
        hard-split with overlap (last resort — should be rare in practice).

        Returns a list of child texts.
        """
        children: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.counter.count(sent)

            # Sentence alone exceeds limit → hard-split it
            if sent_tokens > self.config.child_max_tokens:
                # Flush whatever is accumulated first
                if current_sentences:
                    children.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                # Hard-split the oversized sentence
                children.extend(
                    self.counter.split_with_overlap(
                        sent,
                        self.config.child_max_tokens,
                        self.config.overlap_tokens,
                    )
                )
                continue

            # Would adding this sentence exceed the limit?
            if current_tokens + sent_tokens > self.config.child_max_tokens and current_sentences:
                children.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sent)
            current_tokens += sent_tokens

        if current_sentences:
            children.append(" ".join(current_sentences))

        return children

    # ── Overlap injection ────────────────────────────────────────────────────

    def _inject_overlap(self, tree: ChunkTree) -> None:
        """
        For each child chunk, attach the tail of the previous sibling
        and the head of the next sibling as context_prefix / context_suffix.

        This ensures that information straddling a sentence boundary is
        available at retrieval time without doubling the index size.
        """
        if self.config.overlap_tokens == 0:
            return

        parents = tree.parents
        for parent in parents:
            children = tree.children_of(parent.id)
            for i, child in enumerate(children):
                if i > 0:
                    child.context_prefix = self.counter.tail_tokens(
                        children[i - 1].text, self.config.overlap_tokens
                    )
                if i < len(children) - 1:
                    child.context_suffix = self.counter.head_tokens(
                        children[i + 1].text, self.config.overlap_tokens
                    )

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _blocks_to_text(blocks: list[ParsedBlock]) -> str:
        return "\n\n".join(b.text for b in blocks)

    @staticmethod
    def _update_heading_path(
        current_path: list[str], heading_text: str
    ) -> list[str]:
        """
        Keeps the heading breadcrumb accurate.
        For simplicity (no heading-level info from PDFs), we just
        track depth by heuristic: if the heading is shorter it's
        probably higher level, so we reset.  For PDFs with well-
        structured bookmarks, wire in the PDF outline depth here.
        """
        # Keep at most 3 levels for sanity
        if len(current_path) >= 3:
            return [heading_text]
        return current_path + [heading_text]
