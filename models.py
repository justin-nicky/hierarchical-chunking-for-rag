"""
models.py
---------
Core data structures for the two-level hierarchical chunker.

Level 0 = Parent  → section / large paragraph group.
                    Sent to the LLM as generation context.
Level 1 = Child   → individual sentence / small text unit.
                    Embedded and stored in the vector DB for retrieval.

Retrieval flow (small-to-big):
  query → embed → retrieve top-k Children → expand to their Parents
        → send Parent text to LLM for generation.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Iterator


class ChunkLevel(IntEnum):
    PARENT = 0  # section / paragraph group
    CHILD = 1   # sentence / small unit inside a parent


@dataclass
class ChunkMetadata:
    """
    Provenance attached to every chunk.
    Carries enough information to:
      - cite the source (doc_id, page_num, heading_path)
      - debug retrieval (char_start, char_end, token_count)
      - detect stale chunks (content_hash)
      - filter by document in the vector DB (doc_id, doc_path)
    """
    doc_id: str                      # stable ID for the source PDF
    doc_path: str                    # absolute path or URL to the PDF
    page_num: int                    # 0-indexed page the chunk starts on
    heading_path: list[str]          # breadcrumb: ["Chapter 1", "Introduction"]
    char_start: int                  # byte offset in the page's extracted text
    char_end: int
    token_count: int
    content_hash: str                # sha256(text) — used for dedup & incremental updates
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    chunk_level: ChunkLevel = ChunkLevel.PARENT


@dataclass
class Chunk:
    """
    A single node in the two-level hierarchy.

    Children always point to their parent via parent_id.
    Parents do NOT store child IDs here — query the ChunkStore instead.
    This keeps the dataclass lightweight and serialization simple.
    """
    id: str
    text: str                        # raw content of this chunk
    level: ChunkLevel
    metadata: ChunkMetadata
    parent_id: str | None = None     # None for PARENT-level chunks

    # Optional context stitching (added by OverlapInjector after tree is built)
    context_prefix: str = ""         # tail of previous sibling parent (for child chunks)
    context_suffix: str = ""         # head of next sibling parent (for child chunks)

    @property
    def text_with_context(self) -> str:
        """
        Text as it should be presented to the LLM:
        includes overlap from neighbouring chunks when available.
        """
        parts = []
        if self.context_prefix:
            parts.append(self.context_prefix)
        parts.append(self.text)
        if self.context_suffix:
            parts.append(self.context_suffix)
        return "\n".join(parts)

    @classmethod
    def create(
        cls,
        text: str,
        level: ChunkLevel,
        metadata: ChunkMetadata,
        parent_id: str | None = None,
    ) -> "Chunk":
        """Factory — generates a stable ID and content hash automatically."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        metadata.content_hash = content_hash
        metadata.chunk_level = level
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            level=level,
            metadata=metadata,
            parent_id=parent_id,
        )


@dataclass
class ChunkTree:
    """
    Holds the complete two-level hierarchy for a single PDF document.

    parents: ordered list of PARENT chunks (section order = document order)
    children: map from parent_id → ordered list of CHILD chunks
    """
    doc_id: str
    doc_path: str
    parents: list[Chunk] = field(default_factory=list)
    _children: dict[str, list[Chunk]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Mutation helpers (used by HierarchicalChunker while building)
    # ------------------------------------------------------------------

    def add_parent(self, chunk: Chunk) -> None:
        assert chunk.level == ChunkLevel.PARENT
        self.parents.append(chunk)
        self._children.setdefault(chunk.id, [])

    def add_child(self, chunk: Chunk) -> None:
        assert chunk.level == ChunkLevel.CHILD
        assert chunk.parent_id is not None, "child must have a parent_id"
        self._children.setdefault(chunk.parent_id, []).append(chunk)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def children_of(self, parent_id: str) -> list[Chunk]:
        return self._children.get(parent_id, [])

    def iter_children(self) -> Iterator[Chunk]:
        """Yields all CHILD chunks in document order."""
        for parent in self.parents:
            yield from self._children.get(parent.id, [])

    def iter_all(self) -> Iterator[Chunk]:
        """Yields parents then their children, in document order."""
        for parent in self.parents:
            yield parent
            yield from self._children.get(parent.id, [])

    def get_parent_of(self, child: Chunk) -> Chunk | None:
        """Reverse-lookup: given a child chunk, return its parent."""
        if child.parent_id is None:
            return None
        for parent in self.parents:
            if parent.id == child.parent_id:
                return parent
        return None

    # ------------------------------------------------------------------
    # Stats (useful for logging / observability)
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        children = list(self.iter_children())
        child_tokens = [c.metadata.token_count for c in children]
        parent_tokens = [p.metadata.token_count for p in self.parents]
        return {
            "doc_id": self.doc_id,
            "parent_count": len(self.parents),
            "child_count": len(children),
            "parent_avg_tokens": round(sum(parent_tokens) / len(parent_tokens), 1) if parent_tokens else 0,
            "child_avg_tokens": round(sum(child_tokens) / len(child_tokens), 1) if child_tokens else 0,
            "child_max_tokens": max(child_tokens) if child_tokens else 0,
        }
