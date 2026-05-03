"""
example.py
----------
Demonstrates the full chunking pipeline end-to-end.

Run with:
    python example.py path/to/your/document.pdf

What this script shows:
  1. Basic usage with default config (in-memory store).
  2. Inspecting parents and their children.
  3. The small-to-big retrieval pattern.
  4. Switching to a SQLite store for persistence.
  5. Incremental update after a document changes.
  6. Custom config tuning.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s │ %(name)s │ %(message)s",
)

# ── Add project root to path so we can import without installing ─────────────
sys.path.insert(0, str(Path(__file__).parent))

from chunker import (
    Chunk,
    ChunkingPipeline,
    ChunkerConfig,
    ChunkLevel,
    SQLiteChunkStore,
)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def demo_basic(pdf_path: str) -> None:
    section("1. Basic usage — default config, in-memory store")

    pipeline = ChunkingPipeline()
    tree = pipeline.run(pdf_path)

    stats = tree.stats()
    print(f"  PDF:             {Path(pdf_path).name}")
    print(f"  Parent chunks:   {stats['parent_count']}")
    print(f"  Child chunks:    {stats['child_count']}")
    print(f"  Avg parent tok:  {stats['parent_avg_tokens']}")
    print(f"  Avg child tok:   {stats['child_avg_tokens']}")
    print(f"  Max child tok:   {stats['child_max_tokens']}")


def demo_inspect_tree(pdf_path: str) -> None:
    section("2. Inspecting the parent/child tree")

    pipeline = ChunkingPipeline()
    tree = pipeline.run(pdf_path)

    # Show first 3 parents and their children
    for parent in tree.parents[:3]:
        heading = " > ".join(parent.metadata.heading_path) or "(preamble)"
        print(f"\n  PARENT [{parent.id[:8]}…]  page={parent.metadata.page_num}  tok={parent.metadata.token_count}")
        print(f"  Heading: {heading}")
        print(f"  Text preview: {parent.text[:120].replace(chr(10), ' ')}…")

        children = tree.children_of(parent.id)
        print(f"  └─ {len(children)} children:")
        for child in children[:3]:
            prefix = "✓" if child.context_prefix else " "
            suffix = "✓" if child.context_suffix else " "
            print(
                f"     [{child.id[:8]}…] tok={child.metadata.token_count:3d}"
                f"  prefix={prefix} suffix={suffix}"
                f"  "{child.text[:70].replace(chr(10), ' ')}…""
            )
        if len(children) > 3:
            print(f"     … and {len(children) - 3} more children")


def demo_small_to_big(pdf_path: str) -> None:
    section("3. Small-to-big retrieval pattern")

    pipeline = ChunkingPipeline()
    tree = pipeline.run(pdf_path)

    # Simulate: vector DB returns a child chunk ID
    # In reality this comes from: results = vector_db.query(embed(query), top_k=5)
    first_child = next(tree.iter_children())

    print(f"  Query hits child: [{first_child.id[:8]}…]")
    print(f"  Child text:       "{first_child.text[:100]}…"")
    print()

    result = pipeline.retrieve_with_context(first_child.id)
    parent = result["parent"]

    print(f"  Expanded to parent: [{parent.id[:8]}…]")
    print(f"  Parent token count: {parent.metadata.token_count}")
    print(f"  Parent heading:     {' > '.join(parent.metadata.heading_path)}")
    print(f"  Parent text (first 200 chars):")
    print(f"    {parent.text[:200].replace(chr(10), ' ')}…")
    print()
    print("  → Send parent.text to LLM as context for generation.")


def demo_sqlite_store(pdf_path: str) -> None:
    section("4. SQLite store — durable persistence")

    db_path = "/tmp/demo_chunks.db"
    with SQLiteChunkStore(db_path) as store:
        pipeline = ChunkingPipeline(store=store)
        tree = pipeline.run(pdf_path)

        # Retrieve the tree from the store (simulates a second process)
        reloaded = store.get_tree(tree.doc_id)
        print(f"  Saved to: {db_path}")
        print(f"  Reloaded: {reloaded.stats()}")

        # Parent-of lookup
        first_child = next(reloaded.iter_children())
        parent = store.get_parent(first_child)
        print(f"  Child [{first_child.id[:8]}…] → Parent [{parent.id[:8]}…]  ✓")


def demo_incremental_update(pdf_path: str) -> None:
    section("5. Incremental update")

    pipeline = ChunkingPipeline()
    tree1 = pipeline.run(pdf_path)
    print(f"  Initial:  {tree1.stats()['child_count']} children")

    # Simulate a second call to the same file (no actual changes = all unchanged)
    diff = pipeline.update(pdf_path)
    print(f"  After update (same file):")
    print(f"    Added:     {len(diff.added)}")
    print(f"    Removed:   {len(diff.removed)}")
    print(f"    Unchanged: {len(diff.unchanged)}")
    print()
    print("  In a real update cycle:")
    print("    for chunk_id in diff.removed: vector_db.delete(chunk_id)")
    print("    for chunk in diff.added:      vector_db.upsert(chunk.id, embed(chunk.text), …)")


def demo_custom_config(pdf_path: str) -> None:
    section("6. Custom config")

    config = ChunkerConfig(
        parent_max_tokens=800,   # larger sections for narrative documents
        child_max_tokens=200,    # longer sentences for technical content
        overlap_tokens=30,       # more overlap for dense material
        encoding_name="cl100k_base",
    )
    pipeline = ChunkingPipeline(config=config)
    tree = pipeline.run(pdf_path)
    print(f"  Custom config stats: {tree.stats()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example.py path/to/document.pdf")
        sys.exit(1)

    pdf = sys.argv[1]
    demo_basic(pdf)
    demo_inspect_tree(pdf)
    demo_small_to_big(pdf)
    demo_sqlite_store(pdf)
    demo_incremental_update(pdf)
    demo_custom_config(pdf)
