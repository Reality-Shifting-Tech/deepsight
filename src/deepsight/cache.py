"""Content-addressed perception cache.

Every vision-model answer is cached by ``(image_hash, region, prompt_kind)``
so repeated looks at the same pixels never re-pay vision tokens. The cache
is the efficiency engine behind the benchmark claim: cache hits cost
nothing and keep tokens-per-correct-answer low.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

_REGION_NONE = (0, 0, 0, 0)


@dataclass(slots=True)
class CacheEntry:
    """One cached vision answer with its expiry."""

    answer: str
    expires_at: float


@dataclass(slots=True)
class PerceptionCache:
    """TTL cache keyed by image hash + region + prompt kind.

    Thread-safe enough for the server's use: lookups and stores are
    atomic dict operations under the GIL.
    """

    ttl_seconds: int = 3600
    _entries: dict[tuple[str, tuple[int, int, int, int], str], CacheEntry] = field(
        default_factory=dict
    )

    @staticmethod
    def image_hash(data: bytes) -> str:
        """SHA-256 of the raw image bytes."""
        return hashlib.sha256(data).hexdigest()

    def _key(
        self,
        image_hash: str,
        region: tuple[int, int, int, int] | None,
        kind: str,
    ) -> tuple[str, tuple[int, int, int, int], str]:
        return (image_hash, region or _REGION_NONE, kind)

    def get(
        self,
        image_hash: str,
        region: tuple[int, int, int, int] | None,
        kind: str,
    ) -> str | None:
        """Return a live cached answer or None (expired entries are dropped)."""
        key = self._key(image_hash, region, kind)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.monotonic():
            del self._entries[key]
            return None
        return entry.answer

    def put(
        self,
        image_hash: str,
        region: tuple[int, int, int, int] | None,
        kind: str,
        answer: str,
    ) -> None:
        """Store an answer under the composite key."""
        key = self._key(image_hash, region, kind)
        self._entries[key] = CacheEntry(
            answer=answer,
            expires_at=time.monotonic() + self.ttl_seconds,
        )

    def clear(self) -> None:
        """Drop every entry (used by tests and :meth:`deepsight.doctor`)."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
