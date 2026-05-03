from __future__ import annotations

import tiktoken


class TokenCounter:
    _CHARS_PER_TOKEN = 4   # GPT-family heuristic for fallback mode

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
            self._fallback = False
        except Exception:
            # Vocab file unavailable — use character-based approximation.
            self._enc = None  # type: ignore[assignment]
            self._fallback = True

    # ── internal helpers ─────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        if self._fallback:
            return max(1, len(text) // self._CHARS_PER_TOKEN) if text else 0
        return len(self._enc.encode(text))

    def _slice_chars(self, text: str, n_tokens: int) -> str:
        """Return approximately `n_tokens` worth of characters from `text`."""
        return text[:n_tokens * self._CHARS_PER_TOKEN]

    def _slice_chars_tail(self, text: str, n_tokens: int) -> str:
        return text[-(n_tokens * self._CHARS_PER_TOKEN):]

    # ── Public API ────────────────────────────────────────────────────────

    def count(self, text: str) -> int:
        """Return the number of tokens in `text`."""
        return self._count_tokens(text)

    def fits(self, text: str, max_tokens: int) -> bool:
        return self.count(text) <= max_tokens

    def truncate(self, text: str, max_tokens: int) -> str:
        """
        Hard-truncate text to at most `max_tokens` tokens.
        Used as a last-resort safety net on oversized chunks.
        """
        if self.count(text) <= max_tokens:
            return text
        if self._fallback:
            return self._slice_chars(text, max_tokens)
        tokens = self._enc.encode(text)
        return self._enc.decode(tokens[:max_tokens])

    def head_tokens(self, text: str, n: int) -> str:
        """Return the first `n` tokens of text, decoded back to a string."""
        if self._fallback:
            return self._slice_chars(text, n)
        tokens = self._enc.encode(text)
        return self._enc.decode(tokens[:n])

    def tail_tokens(self, text: str, n: int) -> str:
        """Return the last `n` tokens of text, decoded back to a string."""
        if self._fallback:
            return self._slice_chars_tail(text, n)
        tokens = self._enc.encode(text)
        return self._enc.decode(tokens[-n:])

    def split_with_overlap(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 20,
    ) -> list[str]:
        """
        Split `text` into token-bounded chunks with overlap.
        Used as the last resort when a text unit can't be split
        on sentence boundaries and still exceeds `max_tokens`.
        """
        if self.count(text) <= max_tokens:
            return [text]

        if self._fallback:
            # Character-based splitting in fallback mode
            step = max_tokens * self._CHARS_PER_TOKEN
            ovlp = overlap_tokens * self._CHARS_PER_TOKEN
            chunks, start = [], 0
            while start < len(text):
                end = min(start + step, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += step - ovlp
            return chunks

        tokens = self._enc.encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunks.append(self._enc.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += max_tokens - overlap_tokens
        return chunks
