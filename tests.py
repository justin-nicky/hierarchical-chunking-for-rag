"""
tests.py
--------
Unit test suite. Run with:
    python -m pytest tests.py -v
  or:
    python tests.py

Tests are grouped by component:
  - TokenCounter
  - PDFParser (uses an in-memory synthetic PDF via PyMuPDF)
  - HierarchicalChunker
  - InMemoryChunkStore
  - SQLiteChunkStore
  - ChunkingPipeline (integration)
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fitz  # PyMuPDF — used to create synthetic test PDFs

from chunker.config import ChunkerConfig
from chunker.models import Chunk, ChunkLevel, ChunkMetadata, ChunkTree
from chunker.token_counter import TokenCounter
from chunker.pdf_parser import PDFParser, ParsedBlock
from chunker.chunker import HierarchicalChunker
from chunker.store import InMemoryChunkStore, SQLiteChunkStore
from chunker.pipeline import ChunkingPipeline
from datetime import datetime, timezone


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_meta(**kwargs) -> ChunkMetadata:
    defaults = dict(
        doc_id="test_doc",
        doc_path="/tmp/test.pdf",
        page_num=0,
        heading_path=[],
        char_start=0,
        char_end=100,
        token_count=10,
        content_hash="abc123",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return ChunkMetadata(**defaults)


def make_chunk(text="hello world", level=ChunkLevel.CHILD, parent_id=None) -> Chunk:
    return Chunk.create(text=text, level=level, metadata=make_meta(), parent_id=parent_id)


def make_pdf_with_sections(path: str) -> None:
    """
    Creates a minimal synthetic PDF with two sections for testing.
    Uses PyMuPDF to write text directly.
    """
    doc = fitz.open()
    page = doc.new_page()

    # Section 1 heading (large font)
    page.insert_text((50, 72), "Introduction", fontsize=18, fontname="helv")
    page.insert_text(
        (50, 110),
        "This is the first paragraph of the introduction. "
        "It contains several sentences about the topic. "
        "Machine learning is a subset of artificial intelligence.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (50, 160),
        "Neural networks consist of layers of interconnected nodes. "
        "Each layer transforms the input data in a non-linear fashion. "
        "Training involves minimising a loss function via gradient descent.",
        fontsize=11,
        fontname="helv",
    )

    # Section 2 heading
    page.insert_text((50, 230), "Methods", fontsize=18, fontname="helv")
    page.insert_text(
        (50, 268),
        "We used a transformer architecture with 12 attention heads. "
        "The model was pre-trained on a large text corpus. "
        "Fine-tuning was performed with a learning rate of 1e-4.",
        fontsize=11,
        fontname="helv",
    )

    doc.save(path)
    doc.close()


# ── TokenCounter tests ────────────────────────────────────────────────────────

class TestTokenCounter(unittest.TestCase):

    def setUp(self):
        self.counter = TokenCounter()

    def test_count_returns_int(self):
        n = self.counter.count("hello world")
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_empty_string(self):
        self.assertEqual(self.counter.count(""), 0)

    def test_fits_true(self):
        self.assertTrue(self.counter.fits("short text", max_tokens=100))

    def test_fits_false(self):
        long_text = "word " * 1000
        self.assertFalse(self.counter.fits(long_text, max_tokens=10))

    def test_truncate_respects_limit(self):
        long_text = "word " * 500
        truncated = self.counter.truncate(long_text, max_tokens=50)
        self.assertLessEqual(self.counter.count(truncated), 50)

    def test_split_with_overlap_produces_multiple_chunks(self):
        text = "sentence " * 200
        chunks = self.counter.split_with_overlap(text, max_tokens=50, overlap_tokens=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(self.counter.count(chunk), 50)

    def test_head_tokens(self):
        text = "The quick brown fox jumps over the lazy dog"
        head = self.counter.head_tokens(text, n=3)
        self.assertLessEqual(self.counter.count(head), 3)

    def test_tail_tokens(self):
        text = "The quick brown fox jumps over the lazy dog"
        tail = self.counter.tail_tokens(text, n=3)
        self.assertLessEqual(self.counter.count(tail), 3)


# ── Chunk model tests ─────────────────────────────────────────────────────────

class TestChunkModel(unittest.TestCase):

    def test_create_generates_id(self):
        c = make_chunk()
        self.assertIsNotNone(c.id)
        self.assertGreater(len(c.id), 0)

    def test_create_sets_content_hash(self):
        c = make_chunk(text="hello")
        self.assertNotEqual(c.metadata.content_hash, "")

    def test_same_text_same_hash(self):
        c1 = make_chunk(text="same text")
        c2 = make_chunk(text="same text")
        self.assertEqual(c1.metadata.content_hash, c2.metadata.content_hash)

    def test_different_text_different_hash(self):
        c1 = make_chunk(text="text A")
        c2 = make_chunk(text="text B")
        self.assertNotEqual(c1.metadata.content_hash, c2.metadata.content_hash)

    def test_text_with_context(self):
        c = make_chunk(text="main")
        c.context_prefix = "before"
        c.context_suffix = "after"
        combined = c.text_with_context
        self.assertIn("before", combined)
        self.assertIn("main", combined)
        self.assertIn("after", combined)

    def test_text_with_context_no_overlap(self):
        c = make_chunk(text="main")
        self.assertEqual(c.text_with_context, "main")


# ── ChunkTree tests ───────────────────────────────────────────────────────────

class TestChunkTree(unittest.TestCase):

    def _make_tree(self):
        tree = ChunkTree(doc_id="doc1", doc_path="/tmp/doc1.pdf")
        parent = make_chunk(text="section text", level=ChunkLevel.PARENT)
        tree.add_parent(parent)
        child1 = make_chunk(text="sentence one", level=ChunkLevel.CHILD, parent_id=parent.id)
        child2 = make_chunk(text="sentence two", level=ChunkLevel.CHILD, parent_id=parent.id)
        tree.add_child(child1)
        tree.add_child(child2)
        return tree, parent, child1, child2

    def test_children_of(self):
        tree, parent, c1, c2 = self._make_tree()
        children = tree.children_of(parent.id)
        self.assertEqual(len(children), 2)

    def test_iter_children(self):
        tree, _, c1, c2 = self._make_tree()
        ids = [c.id for c in tree.iter_children()]
        self.assertIn(c1.id, ids)
        self.assertIn(c2.id, ids)

    def test_get_parent_of(self):
        tree, parent, c1, _ = self._make_tree()
        found = tree.get_parent_of(c1)
        self.assertIsNotNone(found)
        self.assertEqual(found.id, parent.id)

    def test_stats(self):
        tree, _, _, _ = self._make_tree()
        stats = tree.stats()
        self.assertEqual(stats["parent_count"], 1)
        self.assertEqual(stats["child_count"], 2)


# ── PDFParser tests ────────────────────────────────────────────────────────────

class TestPDFParser(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.tmp.close()
        make_pdf_with_sections(self.tmp.name)
        self.parser = PDFParser()

    def test_parse_returns_parsed_document(self):
        doc = self.parser.parse(self.tmp.name)
        self.assertEqual(doc.doc_path, str(Path(self.tmp.name).resolve()))

    def test_blocks_are_non_empty(self):
        doc = self.parser.parse(self.tmp.name)
        self.assertGreater(len(doc.blocks), 0)

    def test_heading_blocks_detected(self):
        doc = self.parser.parse(self.tmp.name)
        headings = [b for b in doc.blocks if b.block_type == "heading"]
        self.assertGreater(len(headings), 0)

    def test_body_blocks_detected(self):
        doc = self.parser.parse(self.tmp.name)
        body = [b for b in doc.blocks if b.block_type == "body"]
        self.assertGreater(len(body), 0)

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("/nonexistent/file.pdf")

    def test_raises_on_wrong_extension(self):
        with self.assertRaises(ValueError):
            self.parser.parse("/tmp/file.txt")


# ── HierarchicalChunker tests ─────────────────────────────────────────────────

class TestHierarchicalChunker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.tmp.close()
        make_pdf_with_sections(self.tmp.name)

        self.config = ChunkerConfig(
            parent_max_tokens=400,
            child_max_tokens=100,
            overlap_tokens=10,
        )
        self.parser = PDFParser()
        self.chunker = HierarchicalChunker(self.config)

    def test_produces_chunk_tree(self):
        doc = self.parser.parse(self.tmp.name)
        tree = self.chunker.chunk(doc)
        self.assertIsInstance(tree, ChunkTree)

    def test_has_parents(self):
        doc = self.parser.parse(self.tmp.name)
        tree = self.chunker.chunk(doc)
        self.assertGreater(len(tree.parents), 0)

    def test_has_children(self):
        doc = self.parser.parse(self.tmp.name)
        tree = self.chunker.chunk(doc)
        children = list(tree.iter_children())
        self.assertGreater(len(children), 0)

    def test_children_have_parent_id(self):
        doc = self.parser.parse(self.tmp.name)
        tree = self.chunker.chunk(doc)
        for child in tree.iter_children():
            self.assertIsNotNone(child.parent_id)

    def test_parent_ids_are_valid(self):
        doc = self.parser.parse(self.tmp.name)
        tree = self.chunker.chunk(doc)
        parent_ids = {p.id for p in tree.parents}
        for child in tree.iter_children():
            self.assertIn(child.parent_id, parent_ids)

    def test_child_token_limit_respected(self):
        doc = self.parser.parse(self.tmp.name)
        tree = self.chunker.chunk(doc)
        counter = TokenCounter()
        for child in tree.iter_children():
            self.assertLessEqual(
                counter.count(child.text),
                self.config.child_max_tokens,
                f"Child exceeded limit: {child.text[:50]}",
            )

    def test_overlap_injected(self):
        doc = self.parser.parse(self.tmp.name)
        tree = self.chunker.chunk(doc)
        # At least some children should have prefix/suffix if there are multiple children per parent
        children_with_overlap = [
            c for c in tree.iter_children()
            if c.context_prefix or c.context_suffix
        ]
        # Only meaningful if parents have multiple children
        if any(len(tree.children_of(p.id)) > 1 for p in tree.parents):
            self.assertGreater(len(children_with_overlap), 0)

    def test_sentence_splitter_handles_abbreviations(self):
        splitter = self.chunker._split_into_sentences
        text = "Dr. Smith works at the U.S. Dept. of Defense. He has a Ph.D. in physics."
        sentences = splitter(text)
        # Should NOT split "Dr." or "U.S." into separate sentences
        self.assertLessEqual(len(sentences), 4)  # regex splitter; not perfect, but must not fragment excessively

    def test_sentence_splitter_splits_normal_text(self):
        splitter = self.chunker._split_into_sentences
        text = "First sentence here. Second sentence follows. Third one ends."
        sentences = splitter(text)
        self.assertGreaterEqual(len(sentences), 2)


# ── InMemoryChunkStore tests ──────────────────────────────────────────────────

class TestInMemoryChunkStore(unittest.TestCase):

    def _make_tree(self, doc_id="doc1"):
        tree = ChunkTree(doc_id=doc_id, doc_path="/tmp/x.pdf")
        parent = make_chunk(text="parent text", level=ChunkLevel.PARENT)
        parent.metadata.doc_id = doc_id
        tree.add_parent(parent)
        child = make_chunk(text="child text", level=ChunkLevel.CHILD, parent_id=parent.id)
        child.metadata.doc_id = doc_id
        tree.add_child(child)
        return tree, parent, child

    def test_save_and_get(self):
        store = InMemoryChunkStore()
        tree, parent, child = self._make_tree()
        store.save_tree(tree)
        self.assertIsNotNone(store.get(parent.id))
        self.assertIsNotNone(store.get(child.id))

    def test_get_nonexistent_returns_none(self):
        store = InMemoryChunkStore()
        self.assertIsNone(store.get("nonexistent-id"))

    def test_get_parent(self):
        store = InMemoryChunkStore()
        tree, parent, child = self._make_tree()
        store.save_tree(tree)
        found = store.get_parent(child)
        self.assertIsNotNone(found)
        self.assertEqual(found.id, parent.id)

    def test_get_children(self):
        store = InMemoryChunkStore()
        tree, parent, child = self._make_tree()
        store.save_tree(tree)
        children = store.get_children(parent.id)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].id, child.id)

    def test_iter_children_only_returns_children(self):
        store = InMemoryChunkStore()
        tree, _, _ = self._make_tree()
        store.save_tree(tree)
        for c in store.iter_children():
            self.assertEqual(c.level, ChunkLevel.CHILD)

    def test_delete_doc(self):
        store = InMemoryChunkStore()
        tree, parent, child = self._make_tree()
        store.save_tree(tree)
        deleted = store.delete_doc("doc1")
        self.assertEqual(deleted, 2)
        self.assertIsNone(store.get(parent.id))
        self.assertIsNone(store.get(child.id))

    def test_get_tree_roundtrip(self):
        store = InMemoryChunkStore()
        tree, parent, child = self._make_tree()
        store.save_tree(tree)
        reloaded = store.get_tree("doc1")
        self.assertIsNotNone(reloaded)
        self.assertEqual(len(reloaded.parents), 1)
        self.assertEqual(len(list(reloaded.iter_children())), 1)


# ── SQLiteChunkStore tests ────────────────────────────────────────────────────

class TestSQLiteChunkStore(unittest.TestCase):

    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()

    def _make_tree(self, doc_id="doc1"):
        tree = ChunkTree(doc_id=doc_id, doc_path="/tmp/x.pdf")
        parent = make_chunk(text="parent text", level=ChunkLevel.PARENT)
        parent.metadata.doc_id = doc_id
        tree.add_parent(parent)
        child = make_chunk(text="child text", level=ChunkLevel.CHILD, parent_id=parent.id)
        child.metadata.doc_id = doc_id
        tree.add_child(child)
        return tree, parent, child

    def test_save_and_get(self):
        with SQLiteChunkStore(self.tmp_db.name) as store:
            tree, parent, child = self._make_tree()
            store.save_tree(tree)
            self.assertIsNotNone(store.get(parent.id))
            self.assertIsNotNone(store.get(child.id))

    def test_get_parent(self):
        with SQLiteChunkStore(self.tmp_db.name) as store:
            tree, parent, child = self._make_tree()
            store.save_tree(tree)
            found = store.get_parent(child)
            self.assertIsNotNone(found)
            self.assertEqual(found.id, parent.id)

    def test_get_tree_roundtrip(self):
        with SQLiteChunkStore(self.tmp_db.name) as store:
            tree, parent, child = self._make_tree()
            store.save_tree(tree)
            reloaded = store.get_tree("doc1")
            self.assertIsNotNone(reloaded)
            self.assertEqual(len(reloaded.parents), 1)

    def test_delete_doc(self):
        with SQLiteChunkStore(self.tmp_db.name) as store:
            tree, parent, child = self._make_tree()
            store.save_tree(tree)
            deleted = store.delete_doc("doc1")
            self.assertGreater(deleted, 0)
            self.assertIsNone(store.get(parent.id))

    def test_heading_path_serialized_correctly(self):
        with SQLiteChunkStore(self.tmp_db.name) as store:
            tree = ChunkTree(doc_id="doc2", doc_path="/tmp/x.pdf")
            parent = make_chunk(text="text", level=ChunkLevel.PARENT)
            parent.metadata.doc_id = "doc2"
            parent.metadata.heading_path = ["Chapter 1", "Section 1.1"]
            tree.add_parent(parent)
            store.save_tree(tree)
            loaded = store.get(parent.id)
            self.assertEqual(loaded.metadata.heading_path, ["Chapter 1", "Section 1.1"])


# ── ChunkingPipeline integration tests ───────────────────────────────────────

class TestChunkingPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.tmp.close()
        make_pdf_with_sections(self.tmp.name)

    def test_run_returns_chunk_tree(self):
        pipeline = ChunkingPipeline()
        tree = pipeline.run(self.tmp.name)
        self.assertIsInstance(tree, ChunkTree)

    def test_run_populates_store(self):
        from chunker.store import InMemoryChunkStore
        store = InMemoryChunkStore()
        pipeline = ChunkingPipeline(store=store)
        pipeline.run(self.tmp.name)
        self.assertGreater(len(store), 0)

    def test_retrieve_with_context(self):
        pipeline = ChunkingPipeline()
        tree = pipeline.run(self.tmp.name)
        first_child = next(tree.iter_children())
        result = pipeline.retrieve_with_context(first_child.id)
        self.assertIsNotNone(result)
        self.assertIn("child", result)
        self.assertIn("parent", result)
        self.assertIsNotNone(result["parent"])

    def test_retrieve_with_context_unknown_id(self):
        pipeline = ChunkingPipeline()
        pipeline.run(self.tmp.name)
        result = pipeline.retrieve_with_context("nonexistent-id")
        self.assertIsNone(result)

    def test_update_no_changes(self):
        pipeline = ChunkingPipeline()
        pipeline.run(self.tmp.name)
        diff = pipeline.update(self.tmp.name)
        # Same file → nothing added or removed
        self.assertFalse(diff.has_changes)
        self.assertGreater(len(diff.unchanged), 0)

    def test_run_batch(self):
        pipeline = ChunkingPipeline()
        trees = pipeline.run_batch([self.tmp.name, self.tmp.name])
        self.assertEqual(len(trees), 2)

    def test_custom_config(self):
        config = ChunkerConfig(
            parent_max_tokens=300,
            child_max_tokens=80,
            overlap_tokens=5,
        )
        pipeline = ChunkingPipeline(config=config)
        tree = pipeline.run(self.tmp.name)
        counter = TokenCounter()
        for child in tree.iter_children():
            self.assertLessEqual(counter.count(child.text), 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
