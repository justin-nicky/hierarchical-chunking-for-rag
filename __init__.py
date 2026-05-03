"""
chunker
-------
Two-level hierarchical PDF chunker for RAG systems.

Public API — import only these:

    from chunker import ChunkingPipeline, ChunkerConfig, ChunkTree, Chunk
    from chunker.store import SQLiteChunkStore
"""

from .config import ChunkerConfig
from .models import Chunk, ChunkLevel, ChunkTree
from .pipeline import ChunkingPipeline, DiffResult
from .store import InMemoryChunkStore, SQLiteChunkStore

__all__ = [
    "ChunkingPipeline",
    "ChunkerConfig",
    "ChunkTree",
    "Chunk",
    "ChunkLevel",
    "DiffResult",
    "InMemoryChunkStore",
    "SQLiteChunkStore",
]
