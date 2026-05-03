"""
pipeline.py
-----------
ChunkingPipeline: the single entry point callers need.

Orchestrates:
  PDFParser → HierarchicalChunker → ChunkStore

Also provides:
  - run()         Process a PDF and persist the chunk tree.
  - run_batch()   Process multiple PDFs (sequential; parallelise externally).
  - update()      Incremental re-chunking: only re-processes changed sections.
  - retrieve()    Given a child chunk, return its parent (small-to-big expansion).

Incremental update logic:
  1. Re-parse and re-chunk the PDF into a new tree.
  2. Compare each new chunk's content_hash against the stored tree.
  3. Persist new/changed chunks; delete removed chunks.
  4. Return a DiffResult so the caller can sync their vector DB.

Usage:
    pipeline = ChunkingPipeline()                # in-memory store, defaults
    tree = pipeline.run("paper.pdf")
    print(tree.stats())

    # With SQLite store:
    from chunker.store import SQLiteChunkStore
    pipeline = ChunkingPipeline(store=SQLiteChunkStore("chunks.db"))

    # Incremental update after editing the PDF:
    diff = pipeline.update("paper.pdf")
    for chunk_id in diff.removed:
        vector_db.delete(chunk_id)          # your vector DB client
    for chunk in diff.added:
        vector_db.upsert(chunk.id, embed(chunk.text), chunk.metadata.__dict__)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import HierarchicalChunker
from .config import ChunkerConfig
from .models import Chunk, ChunkTree
from .pdf_parser import PDFParser
from .store import ChunkStore, InMemoryChunkStore
from .token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class DiffResult:
    """
    Returned by pipeline.update().
    Use this to sync your vector DB after an incremental re-chunk.
    """
    doc_id: str
    added: list[Chunk] = field(default_factory=list)    # new or changed children
    removed: list[str] = field(default_factory=list)    # chunk IDs no longer present
    unchanged: list[str] = field(default_factory=list)  # chunk IDs identical to before

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)

    def summary(self) -> str:
        return (
            f"doc={self.doc_id} "
            f"added={len(self.added)} "
            f"removed={len(self.removed)} "
            f"unchanged={len(self.unchanged)}"
        )


class ChunkingPipeline:
    """
    The only class external code needs to import.

    Args:
        config:  ChunkerConfig (uses defaults if omitted).
        store:   Any ChunkStore implementation. Defaults to InMemoryChunkStore.
        parser:  PDFParser instance (injectable for testing).
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        store: ChunkStore | None = None,
        parser: PDFParser | None = None,
    ) -> None:
        self.config = config or ChunkerConfig()
        self.store = store if store is not None else InMemoryChunkStore()
        self.parser = parser or PDFParser()
        self._counter = TokenCounter(self.config.encoding_name)
        self._chunker = HierarchicalChunker(self.config, self._counter)

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, path: str | Path) -> ChunkTree:
        """
        Parse a PDF and chunk it.  Persists the tree to the store
        and returns it.

        If the doc already exists in the store it is replaced entirely.
        For incremental updates use pipeline.update() instead.
        """
        path = Path(path)
        logger.info("chunking %s", path.name)

        doc = self.parser.parse(path)
        tree = self._chunker.chunk(doc)

        # Replace any previous version in the store
        self.store.delete_doc(doc.doc_id)
        self.store.save_tree(tree)

        stats = tree.stats()
        logger.info(
            "done: %d parents, %d children, avg_child_tokens=%.1f",
            stats["parent_count"],
            stats["child_count"],
            stats["child_avg_tokens"],
        )
        return tree

    def run_batch(self, paths: list[str | Path]) -> list[ChunkTree]:
        """Process multiple PDFs and return their trees in order."""
        trees = []
        for path in paths:
            try:
                trees.append(self.run(path))
            except Exception as exc:
                logger.error("failed to chunk %s: %s", path, exc)
        return trees

    def update(self, path: str | Path) -> DiffResult:
        """
        Incrementally re-chunk a PDF.

        Only child chunks whose content_hash has changed are treated as
        new.  Callers should:
          - Delete diff.removed IDs from their vector DB.
          - Upsert diff.added chunks into their vector DB.
          - Skip diff.unchanged (embeddings are still valid).
        """
        path = Path(path)
        logger.info("incremental update for %s", path.name)

        doc = self.parser.parse(path)
        new_tree = self._chunker.chunk(doc)
        old_tree = self.store.get_tree(doc.doc_id)

        if old_tree is None:
            # First time seeing this doc
            self.store.save_tree(new_tree)
            result = DiffResult(
                doc_id=doc.doc_id,
                added=list(new_tree.iter_children()),
            )
            logger.info("new doc: %s", result.summary())
            return result

        # Compare by content_hash
        old_hashes: dict[str, str] = {
            c.metadata.content_hash: c.id
            for c in old_tree.iter_children()
        }
        new_hashes: dict[str, str] = {
            c.metadata.content_hash: c.id
            for c in new_tree.iter_children()
        }

        added: list[Chunk] = []
        removed: list[str] = []
        unchanged: list[str] = []

        for chunk in new_tree.iter_children():
            h = chunk.metadata.content_hash
            if h in old_hashes:
                unchanged.append(old_hashes[h])
            else:
                added.append(chunk)

        for h, old_id in old_hashes.items():
            if h not in new_hashes:
                removed.append(old_id)

        # Persist the new tree (full replace)
        self.store.delete_doc(doc.doc_id)
        self.store.save_tree(new_tree)

        result = DiffResult(doc.doc_id, added, removed, unchanged)
        logger.info("update done: %s", result.summary())
        return result

    # ── Retrieval helper (small-to-big) ──────────────────────────────────────

    def retrieve_with_context(self, child_chunk_id: str) -> dict | None:
        """
        Given a child chunk ID (e.g. from a vector DB hit), return both the
        child chunk and its expanded parent context.

        This is the small-to-big expansion step used at query time.

        Returns:
            {
              "child":  Chunk,    # the matched sentence chunk
              "parent": Chunk,    # the full section sent to the LLM
            }
            or None if child_chunk_id is not found.
        """
        child = self.store.get(child_chunk_id)
        if child is None:
            return None
        parent = self.store.get_parent(child)
        return {"child": child, "parent": parent}
