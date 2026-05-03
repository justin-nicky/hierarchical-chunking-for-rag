"""
config.py
---------
All tunable parameters for the chunker in one place.
Pydantic validates types and provides helpful error messages.

Usage:
    # defaults
    config = ChunkerConfig()

    # override specific fields
    config = ChunkerConfig(parent_max_tokens=600, child_max_tokens=100)

    # load from env vars (CHUNKER_ prefix)
    config = ChunkerConfig()  # pydantic-settings reads env automatically
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChunkerConfig(BaseModel):
    # ── Token limits ──────────────────────────────────────────────────────
    parent_max_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description=(
            "Maximum tokens in a PARENT chunk. "
            "This is the context sent to the LLM. "
            "Keep below your LLM's context window minus the prompt template. "
            "512 works well for most embedding models (ada-002 limit = 8191, "
            "but smaller parents are faster to generate from)."
        ),
    )
    child_max_tokens: int = Field(
        default=128,
        ge=16,
        le=512,
        description=(
            "Maximum tokens in a CHILD chunk. "
            "This is the unit that gets embedded and stored in the vector DB. "
            "128 is a sweet spot for text-embedding-3-* models. "
            "Shorter = more precise retrieval. Longer = more context per hit."
        ),
    )
    overlap_tokens: int = Field(
        default=20,
        ge=0,
        le=64,
        description=(
            "Tokens borrowed from adjacent sibling child chunks "
            "as context_prefix / context_suffix. "
            "Helps when information straddles a sentence boundary. "
            "Set to 0 to disable."
        ),
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    encoding_name: str = Field(
        default="cl100k_base",
        description=(
            "tiktoken encoding name. "
            "cl100k_base: GPT-4, GPT-3.5, text-embedding-ada-002, text-embedding-3-*. "
            "o200k_base: GPT-4o family. "
            "p50k_base: GPT-3 / Codex (legacy)."
        ),
    )

    # ── Parser ────────────────────────────────────────────────────────────
    heading_size_ratio: float = Field(
        default=1.15,
        ge=1.0,
        le=2.0,
        description=(
            "A block is classified as a heading if its font size >= "
            "median_page_font_size * heading_size_ratio. "
            "Lower = more blocks classified as headings. "
            "Raise to 1.3 for PDFs with large pull-quotes."
        ),
    )
    min_block_chars: int = Field(
        default=20,
        ge=0,
        description="Blocks with fewer characters than this are discarded (page numbers, headers, footers).",
    )

    # ── Store ─────────────────────────────────────────────────────────────
    store_backend: str = Field(
        default="memory",
        description="Chunk store backend: 'memory' or 'sqlite'.",
    )
    sqlite_path: str = Field(
        default="chunks.db",
        description="Path to the SQLite database file (only used when store_backend='sqlite').",
    )

    @field_validator("child_max_tokens")
    @classmethod
    def child_must_be_less_than_parent(cls, v: int, info) -> int:
        parent = info.data.get("parent_max_tokens", 512)
        if v >= parent:
            raise ValueError(
                f"child_max_tokens ({v}) must be less than "
                f"parent_max_tokens ({parent})"
            )
        return v

    model_config = {"env_prefix": "CHUNKER_"}
