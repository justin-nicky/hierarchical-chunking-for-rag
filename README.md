# pdf-chunker

A production-grade, two-level hierarchical PDF chunker for Retrieval-Augmented Generation (RAG) systems. Built with PyMuPDF, tiktoken, and Pydantic — no ML models required.

---

## Table of contents

- [Why hierarchical chunking?](#why-hierarchical-chunking)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [Incremental updates](#incremental-updates)
- [Store backends](#store-backends)
- [Running tests](#running-tests)
- [Design decisions](#design-decisions)

---

## Why hierarchical chunking?

Standard fixed-size chunking forces a tradeoff: small chunks give precise vector search results but lose surrounding context; large chunks preserve context but reduce retrieval precision.

Hierarchical chunking breaks this tradeoff using a **small-to-big retrieval pattern**:

1. **Index small (child) chunks** — one or a few sentences each. These match queries precisely.
2. **Expand to large (parent) chunks** at retrieval time — the full section the sentence came from. This is what gets sent to the LLM.

```
PDF document
├── Parent chunk 1   (section: Introduction, ~400 tokens) ← sent to LLM
│   ├── Child chunk A  (~50 tokens) ← matched by vector search
│   ├── Child chunk B  (~60 tokens)
│   └── Child chunk C  (~45 tokens)
└── Parent chunk 2   (section: Methods, ~380 tokens)
    ├── Child chunk D  (~55 tokens)
    └── Child chunk E  (~70 tokens)
```

---

## How it works

The pipeline runs in five stages every time you call `pipeline.run("file.pdf")`:

```
PDF file
  │
  ▼
[PDFParser]          Extracts text blocks, classifies each as "heading" or "body"
  │                  using font-size ratios and bold detection. No ML needed.
  ▼
[_group_into_sections]   Groups body blocks between consecutive headings
  │                      into sections. Each section becomes one parent chunk.
  ▼
[_split_into_sentences]  Rule-based sentence splitter with abbreviation handling
  │                      (Dr., U.S., Ph.D. etc. are not treated as sentence ends).
  ▼
[_group_sentences_into_children]   Packs sentences into child chunks that
  │                                stay within child_max_tokens. Think of
  │                                filling buckets until each is full.
  ▼
[_inject_overlap]    Second pass: borrows overlap_tokens from the previous
                     and next sibling child and attaches them as
                     context_prefix / context_suffix.
```

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/your-username/pdf-chunker.git
cd pdf-chunker
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `pymupdf` | ≥ 1.23.0 | PDF parsing (imported as `fitz`) |
| `tiktoken` | ≥ 0.6.0 | Token counting |
| `pydantic` | ≥ 2.0.0 | Config validation |

---

## Quickstart

```python
from chunker import ChunkingPipeline

# 1. Create a pipeline with default config
pipeline = ChunkingPipeline()

# 2. Chunk a PDF — returns a ChunkTree
tree = pipeline.run("path/to/document.pdf")

# 3. Inspect the results
print(tree.stats())
# {'parent_count': 12, 'child_count': 87, 'child_avg_tokens': 54.3, ...}

# 4. Iterate over parents and their children
for parent in tree.parents:
    print(f"\nSection: {' > '.join(parent.metadata.heading_path)}")
    print(f"  {parent.metadata.token_count} tokens")
    for child in tree.children_of(parent.id):
        print(f"    [{child.id[:8]}] {child.text[:60]}...")
```

### Small-to-big retrieval (query time)

```python
# After your vector DB returns a child chunk ID from a similarity search:
result = pipeline.retrieve_with_context(child_chunk_id)

child  = result["child"]   # the matched sentence chunk (~50 tokens)
parent = result["parent"]  # the full section (~400 tokens)

# Send the parent text to your LLM — it has full context
llm_context = parent.text_with_context   # includes overlap from neighbours
```

### Using a persistent store (SQLite)

```python
from chunker import ChunkingPipeline
from chunker.store import SQLiteChunkStore

with SQLiteChunkStore("chunks.db") as store:
    pipeline = ChunkingPipeline(store=store)
    tree = pipeline.run("document.pdf")

# Chunks are now persisted — reload them in a later session:
with SQLiteChunkStore("chunks.db") as store:
    pipeline = ChunkingPipeline(store=store)
    result = pipeline.retrieve_with_context(some_child_id)
```

### Batch processing

```python
import glob

pipeline = ChunkingPipeline()
paths = glob.glob("docs/*.pdf")
trees = pipeline.run_batch(paths)
print(f"Processed {len(trees)} documents")
```

---

## Configuration

All tunable parameters live in `ChunkerConfig`. Pass it to `ChunkingPipeline`:

```python
from chunker import ChunkingPipeline, ChunkerConfig

config = ChunkerConfig(
    parent_max_tokens = 512,    # max tokens per parent (section) chunk
    child_max_tokens  = 128,    # max tokens per child (sentence group) chunk
    overlap_tokens    = 20,     # tokens borrowed from neighbouring children
    encoding_name     = "cl100k_base",  # tiktoken encoding
    heading_size_ratio = 1.15,  # font size ratio to classify a block as heading
    store_backend     = "sqlite",
    sqlite_path       = "chunks.db",
)

pipeline = ChunkingPipeline(config=config)
```

**All config fields:**

| Field | Default | Range | Description |
|---|---|---|---|
| `parent_max_tokens` | `512` | 64 – 4096 | Maximum tokens in a parent chunk. This is the context sent to the LLM. |
| `child_max_tokens` | `128` | 16 – 512 | Maximum tokens in a child chunk. Must be less than `parent_max_tokens`. |
| `overlap_tokens` | `20` | 0 – 64 | Tokens borrowed from adjacent siblings. Set to `0` to disable. |
| `encoding_name` | `"cl100k_base"` | — | tiktoken encoding. Use `"o200k_base"` for GPT-4o models. |
| `heading_size_ratio` | `1.15` | 1.0 – 2.0 | Font-size multiplier threshold for heading detection. Raise for PDFs with large pull-quotes. |
| `min_block_chars` | `20` | 0 – ∞ | Blocks shorter than this are discarded (page numbers, headers, footers). |
| `store_backend` | `"memory"` | `"memory"`, `"sqlite"` | Persistence backend. |
| `sqlite_path` | `"chunks.db"` | — | Path to the SQLite database (only used when `store_backend="sqlite"`). |

**Environment variable overrides** — any config field can be set via environment variable with the `CHUNKER_` prefix:

```bash
export CHUNKER_PARENT_MAX_TOKENS=800
export CHUNKER_CHILD_MAX_TOKENS=200
export CHUNKER_STORE_BACKEND=sqlite
```

---

## Project structure

```
pdf-chunker/
├── chunker/
│   ├── __init__.py          # public API — import from here
│   ├── config.py            # ChunkerConfig (pydantic settings)
│   ├── models.py            # Chunk, ChunkMetadata, ChunkTree, ChunkLevel
│   ├── token_counter.py     # tiktoken wrapper with offline fallback
│   ├── pdf_parser.py        # PDFParser — block extraction and heading detection
│   ├── chunker.py           # HierarchicalChunker — the core logic
│   ├── store.py             # ChunkStore protocol + InMemory + SQLite backends
│   └── pipeline.py          # ChunkingPipeline — orchestrator and public API
├── example.py               # runnable demos for all features
├── tests.py                 # 52-test unit suite
└── requirements.txt
```

---

## Architecture

### Data flow

```
ChunkingPipeline.run(path)
        │
        ├─► PDFParser.parse(path)
        │       └─► fitz.open()  →  per-page block extraction
        │           └─► _classify_block()  →  "heading" | "body"
        │           └─► returns ParsedDocument
        │
        └─► HierarchicalChunker.chunk(doc)
                ├─► _group_into_sections()      →  list[{heading, body[]}]
                ├─► _split_parent_if_needed()   →  respects parent_max_tokens
                ├─► _split_into_sentences()     →  rule-based, handles abbrevs
                ├─► _group_sentences_into_children()  →  fills token buckets
                └─► _inject_overlap()           →  prefix/suffix on each child
```

### Core data model

```python
@dataclass
class Chunk:
    id:              str           # stable UUID
    text:            str           # raw content
    level:           ChunkLevel    # PARENT (0) or CHILD (1)
    metadata:        ChunkMetadata # provenance — doc_id, page, heading path, etc.
    parent_id:       str | None    # None for parent chunks
    context_prefix:  str           # tail tokens from previous sibling child
    context_suffix:  str           # head tokens from next sibling child

@dataclass
class ChunkMetadata:
    doc_id:        str           # stable identifier for the source PDF
    doc_path:      str           # absolute path
    page_num:      int           # 0-indexed page the chunk starts on
    heading_path:  list[str]     # breadcrumb: ["Chapter 1", "Section 1.2"]
    token_count:   int
    content_hash:  str           # sha256(text) — used for incremental updates
    created_at:    datetime
```

### Dependency injection

Every component is injected into `ChunkingPipeline`. Swap implementations without changing the caller:

```python
# Production
pipeline = ChunkingPipeline(
    config=ChunkerConfig(parent_max_tokens=800),
    store=SQLiteChunkStore("chunks.db"),
)

# Testing — all in-memory, no files
pipeline = ChunkingPipeline(
    config=ChunkerConfig(child_max_tokens=50),
    store=InMemoryChunkStore(),
    parser=MockPDFParser(),   # your fake
)
```

---

## API reference

### `ChunkingPipeline`

```python
pipeline = ChunkingPipeline(
    config: ChunkerConfig | None = None,   # uses defaults if omitted
    store:  ChunkStore | None   = None,    # InMemoryChunkStore if omitted
    parser: PDFParser | None    = None,    # PDFParser() if omitted
)
```

| Method | Returns | Description |
|---|---|---|
| `.run(path)` | `ChunkTree` | Parse and chunk a PDF. Replaces any existing chunks for this document. |
| `.run_batch(paths)` | `list[ChunkTree]` | Process multiple PDFs in sequence. Errors are logged and skipped. |
| `.update(path)` | `DiffResult` | Incremental re-chunk. Returns only what changed. |
| `.retrieve_with_context(child_id)` | `dict \| None` | Small-to-big expansion. Returns `{"child": Chunk, "parent": Chunk}`. |

### `ChunkTree`

| Method / Property | Returns | Description |
|---|---|---|
| `.parents` | `list[Chunk]` | All parent chunks in document order. |
| `.children_of(parent_id)` | `list[Chunk]` | All children of a given parent, in order. |
| `.iter_children()` | `Iterator[Chunk]` | All child chunks in document order. |
| `.iter_all()` | `Iterator[Chunk]` | Parents and children interleaved in document order. |
| `.get_parent_of(child)` | `Chunk \| None` | Reverse lookup — given a child, return its parent. |
| `.stats()` | `dict` | `parent_count`, `child_count`, avg/max token counts. |

### `DiffResult`

Returned by `pipeline.update()`.

```python
diff.added      # list[Chunk]  — new or changed child chunks; re-embed these
diff.removed    # list[str]    — chunk IDs no longer present; delete from vector DB
diff.unchanged  # list[str]    — chunk IDs with identical content; skip re-embedding
diff.has_changes  # bool
diff.summary()    # str — human-readable summary
```

---

## Incremental updates

When a PDF is edited, `pipeline.update()` re-chunks it and returns only the diff. Use this to keep your vector database in sync without re-embedding everything.

```python
diff = pipeline.update("document.pdf")

if diff.has_changes:
    # Remove stale embeddings
    for chunk_id in diff.removed:
        vector_db.delete(chunk_id)

    # Upsert only new or changed chunks
    for chunk in diff.added:
        embedding = embed_model.embed(chunk.text)
        vector_db.upsert(chunk.id, embedding, chunk.metadata.__dict__)

    # diff.unchanged → do nothing, embeddings are still valid
    print(diff.summary())
    # "doc=report added=3 removed=1 unchanged=84"
```

**How it works:** Each chunk stores a `sha256` hash of its text. On update, the new tree's hashes are compared against the stored tree's hashes. Only chunks whose hash is absent from the old tree are treated as new.

---

## Store backends

### `InMemoryChunkStore`

Default backend. Fast, zero dependencies. Lost when the process exits.

```python
from chunker.store import InMemoryChunkStore
store = InMemoryChunkStore()
```

### `SQLiteChunkStore`

Durable persistence. Uses Python's built-in `sqlite3` module — no extra infrastructure needed. Suitable for corpora up to several GB.

```python
from chunker.store import SQLiteChunkStore

# Use as a context manager (auto-closes the connection)
with SQLiteChunkStore("chunks.db") as store:
    pipeline = ChunkingPipeline(store=store)
    pipeline.run("document.pdf")

# Or manage the lifecycle manually
store = SQLiteChunkStore("chunks.db")
pipeline = ChunkingPipeline(store=store)
pipeline.run("document.pdf")
store.close()
```

**SQLite schema:** Chunks are stored in a single `chunks` table with indexes on `doc_id`, `parent_id`, and `level` for fast parent/child lookups.

### Custom store

Implement the `ChunkStore` protocol to plug in any backend (PostgreSQL, Redis, etc.):

```python
from chunker.store import ChunkStore
from chunker.models import Chunk, ChunkTree
from typing import Iterator

class MyCustomStore:
    def save_tree(self, tree: ChunkTree) -> None: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def get_parent(self, child: Chunk) -> Chunk | None: ...
    def get_children(self, parent_id: str) -> list[Chunk]: ...
    def iter_children(self) -> Iterator[Chunk]: ...
    def get_tree(self, doc_id: str) -> ChunkTree | None: ...
    def delete_doc(self, doc_id: str) -> int: ...

pipeline = ChunkingPipeline(store=MyCustomStore())
```

---

## Running tests

```bash
pip install pytest
python -m pytest tests.py -v
```

The test suite covers 52 cases across all components:

| Test class | What it covers |
|---|---|
| `TestTokenCounter` | counting, truncation, overlap splitting, head/tail slicing |
| `TestChunkModel` | ID generation, content hashing, `text_with_context` |
| `TestChunkTree` | parent/child relationships, iterators, reverse lookup |
| `TestPDFParser` | heading detection, block extraction, error handling |
| `TestHierarchicalChunker` | token limits, parent/child structure, overlap injection, sentence splitting |
| `TestInMemoryChunkStore` | CRUD, roundtrip, delete |
| `TestSQLiteChunkStore` | same as above + serialization of heading_path |
| `TestChunkingPipeline` | end-to-end, incremental update, `retrieve_with_context` |

Tests use a synthetic PDF generated with PyMuPDF — no fixture files needed.

---

## Design decisions

**Heading detection without ML.** Font size relative to the page median plus a bold + short-line heuristic covers the vast majority of real-world PDFs. The ratio is configurable. For PDFs that use colour or ALL-CAPS for headings, extend `_classify_block()` without touching anything else.

**Tokens, not characters.** All size limits are measured in tiktoken tokens, the same unit LLMs and embedding models use. This prevents silent truncation at inference time.

**Offline fallback.** If tiktoken can't download its vocabulary file (air-gapped environments, CI without network access), `TokenCounter` automatically falls back to a character-based approximation (1 token ≈ 4 chars). This keeps the chunker usable everywhere; switch to the real tokenizer by pre-caching the vocab on a machine with internet access.

**Two-pass heading detection.** `_extract_blocks()` first collects all font sizes from the page to compute the median, then classifies each block. A single-pass approach would require a hardcoded font size threshold, which breaks on PDFs with unusual typography.

**Sentinel-based sentence splitting.** The sentence splitter injects `|||` markers at detected boundaries in one regex pass, then splits on the sentinel. This avoids running `str.split()` repeatedly and keeps abbreviation handling isolated in a single inner function.

**Protocol-based store.** `ChunkStore` is a `typing.Protocol` (structural subtyping), not an abstract base class. Any class with the right method signatures satisfies it automatically — no explicit `implements` required. This makes testing with fakes trivial.

**Children don't store child IDs; parents don't store child lists.** Relationships are resolved through the store at query time. This keeps `Chunk` objects lightweight, makes serialization straightforward, and means the tree structure can be reconstructed from a flat list of chunks without any additional metadata.
