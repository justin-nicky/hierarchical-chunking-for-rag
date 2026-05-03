"""
store.py
--------
Persistence layer for ChunkTree objects.

ChunkStore is a Protocol (structural typing interface).
Two concrete implementations ship here:

  InMemoryChunkStore   – for unit tests and one-off scripts.
                         Fast, zero dependencies. Lost on process exit.

  SQLiteChunkStore     – for production and long-running services.
                         Durable, queryable, zero infrastructure.
                         Suitable for corpora up to ~10 GB before you need Postgres.

Swap implementations without changing any caller:
  store = InMemoryChunkStore()   # tests
  store = SQLiteChunkStore("./chunks.db")  # production

Schema (SQLite):
  chunks(
    id TEXT PK,
    text TEXT,
    level INTEGER,
    parent_id TEXT,
    doc_id TEXT,
    doc_path TEXT,
    page_num INTEGER,
    heading_path TEXT,     -- JSON array
    token_count INTEGER,
    content_hash TEXT,
    created_at TEXT,
    context_prefix TEXT,
    context_suffix TEXT
  )
"""

from __future__ import annotations

import json
import sqlite3
from typing import Iterable, Iterator, Protocol, runtime_checkable

from .models import Chunk, ChunkLevel, ChunkMetadata, ChunkTree
from datetime import datetime, timezone


# ── Protocol (interface) ─────────────────────────────────────────────────────

@runtime_checkable
class ChunkStore(Protocol):
    def save_tree(self, tree: ChunkTree) -> None: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def get_parent(self, child: Chunk) -> Chunk | None: ...
    def get_children(self, parent_id: str) -> list[Chunk]: ...
    def iter_children(self) -> Iterator[Chunk]: ...
    def get_tree(self, doc_id: str) -> ChunkTree | None: ...
    def delete_doc(self, doc_id: str) -> int: ...


# ── In-Memory ────────────────────────────────────────────────────────────────

class InMemoryChunkStore:
    """
    Thread-unsafe, non-durable store.
    Perfect for tests, scripts, and Jupyter notebooks.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}        # id → Chunk
        self._by_doc: dict[str, list[str]] = {}    # doc_id → [chunk_id, ...]

    def save_tree(self, tree: ChunkTree) -> None:
        ids: list[str] = []
        for chunk in tree.iter_all():
            self._chunks[chunk.id] = chunk
            ids.append(chunk.id)
        self._by_doc[tree.doc_id] = ids

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_parent(self, child: Chunk) -> Chunk | None:
        if child.parent_id is None:
            return None
        return self._chunks.get(child.parent_id)

    def get_children(self, parent_id: str) -> list[Chunk]:
        return [
            c for c in self._chunks.values()
            if c.parent_id == parent_id
        ]

    def iter_children(self) -> Iterator[Chunk]:
        for chunk in self._chunks.values():
            if chunk.level == ChunkLevel.CHILD:
                yield chunk

    def get_tree(self, doc_id: str) -> ChunkTree | None:
        ids = self._by_doc.get(doc_id)
        if not ids:
            return None
        chunks = [self._chunks[i] for i in ids if i in self._chunks]
        if not chunks:
            return None

        tree = ChunkTree(
            doc_id=doc_id,
            doc_path=chunks[0].metadata.doc_path,
        )
        for c in chunks:
            if c.level == ChunkLevel.PARENT:
                tree.add_parent(c)
            else:
                tree.add_child(c)
        return tree

    def delete_doc(self, doc_id: str) -> int:
        ids = self._by_doc.pop(doc_id, [])
        for i in ids:
            self._chunks.pop(i, None)
        return len(ids)

    def __len__(self) -> int:
        return len(self._chunks)


# ── SQLite ───────────────────────────────────────────────────────────────────

class SQLiteChunkStore:
    """
    Durable SQLite-backed store.

    Connection is kept open for the lifetime of the store object.
    Call .close() explicitly or use as a context manager.

    Example:
        with SQLiteChunkStore("chunks.db") as store:
            store.save_tree(tree)
    """

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS chunks (
        id              TEXT PRIMARY KEY,
        text            TEXT NOT NULL,
        level           INTEGER NOT NULL,
        parent_id       TEXT,
        doc_id          TEXT NOT NULL,
        doc_path        TEXT NOT NULL,
        page_num        INTEGER NOT NULL,
        heading_path    TEXT NOT NULL,
        token_count     INTEGER NOT NULL,
        content_hash    TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        context_prefix  TEXT NOT NULL DEFAULT '',
        context_suffix  TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_doc_id  ON chunks(doc_id);
    CREATE INDEX IF NOT EXISTS idx_parent  ON chunks(parent_id);
    CREATE INDEX IF NOT EXISTS idx_level   ON chunks(level);
    """

    def __init__(self, db_path: str = "chunks.db") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.CREATE_SQL)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────

    def save_tree(self, tree: ChunkTree) -> None:
        rows = [self._to_row(c) for c in tree.iter_all()]
        self._conn.executemany(
            """INSERT OR REPLACE INTO chunks VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.commit()

    # ── Read ─────────────────────────────────────────────────────────────

    def get(self, chunk_id: str) -> Chunk | None:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        )
        row = cur.fetchone()
        return self._from_row(row) if row else None

    def get_parent(self, child: Chunk) -> Chunk | None:
        if child.parent_id is None:
            return None
        return self.get(child.parent_id)

    def get_children(self, parent_id: str) -> list[Chunk]:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE parent_id = ?", (parent_id,)
        )
        return [self._from_row(r) for r in cur.fetchall()]

    def iter_children(self) -> Iterator[Chunk]:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE level = ?", (int(ChunkLevel.CHILD),)
        )
        for row in cur:
            yield self._from_row(row)

    def get_tree(self, doc_id: str) -> ChunkTree | None:
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY level, rowid",
            (doc_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return None

        tree = ChunkTree(
            doc_id=doc_id,
            doc_path=rows[0]["doc_path"],
        )
        for row in rows:
            chunk = self._from_row(row)
            if chunk.level == ChunkLevel.PARENT:
                tree.add_parent(chunk)
            else:
                tree.add_child(chunk)
        return tree

    def delete_doc(self, doc_id: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
        )
        self._conn.commit()
        return cur.rowcount

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteChunkStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Serialization helpers ─────────────────────────────────────────────

    def _to_row(self, chunk: Chunk) -> tuple:
        m = chunk.metadata
        return (
            chunk.id,
            chunk.text,
            int(chunk.level),
            chunk.parent_id,
            m.doc_id,
            m.doc_path,
            m.page_num,
            json.dumps(m.heading_path),
            m.token_count,
            m.content_hash,
            m.created_at.isoformat(),
            chunk.context_prefix,
            chunk.context_suffix,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Chunk:
        meta = ChunkMetadata(
            doc_id=row["doc_id"],
            doc_path=row["doc_path"],
            page_num=row["page_num"],
            heading_path=json.loads(row["heading_path"]),
            char_start=0,
            char_end=0,
            token_count=row["token_count"],
            content_hash=row["content_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            chunk_level=ChunkLevel(row["level"]),
        )
        chunk = Chunk(
            id=row["id"],
            text=row["text"],
            level=ChunkLevel(row["level"]),
            metadata=meta,
            parent_id=row["parent_id"],
            context_prefix=row["context_prefix"] or "",
            context_suffix=row["context_suffix"] or "",
        )
        return chunk
