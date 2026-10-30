"""Topic-depth scoring and the cut ledger for segmenter v2.

Depth = 1 - cos(mean vector of up to 3 exchanges on the left, mean of up to 3 on the
right); higher looks more like a topic change. Exchange text: each message truncated
to 300 characters, joined, at most 512 characters. Vectors are cached by the first
event ID of the exchange, so recursive splits never re-encode.

The embedder is any ``list[str] -> list[vector]`` callable (a sentence-embedding model
in production, a hashing embedder in tests). This module is pure Python; numpy is not
needed.

The ledger records the cuts of closed sessions so a replay reads them back instead of
re-scoring, which keeps segment IDs stable even if the embedding model changes. It is
a derived cache: delete it and cuts are recomputed.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .._util import atomic_json_dump, json_load
from .topic import SEGMENTER_V2, Unit

WINDOW = 3
MESSAGE_CHARS = 300
UNIT_CHARS = 512

Embedder = Callable[[List[str]], Sequence[Sequence[float]]]


def unit_text(unit: Unit) -> str:
    return "\n".join(str(event.content or "").strip()[:MESSAGE_CHARS] for event in unit)[:UNIT_CHARS]


def _normalize(vector: Sequence[float]) -> List[float]:
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values)) or 1.0
    return [x / norm for x in values]


def _mean(vectors: Sequence[Sequence[float]]) -> List[float]:
    return [sum(column) / len(vectors) for column in zip(*vectors)]


class EmbeddingScorer:
    """Usable directly as a ``segments.topic.Scorer``."""

    def __init__(self, embedder: Embedder, *, window: int = WINDOW) -> None:
        self._embedder = embedder
        self.window = window
        self._vectors: Dict[str, List[float]] = {}
        self.encoded = 0

    def prepare(self, units: Sequence[Unit]) -> None:
        """Encode the exchanges not cached yet (once per session is enough)."""
        missing = [unit for unit in units if unit[0].event_id not in self._vectors]
        if not missing:
            return
        vectors = self._embedder([unit_text(unit) for unit in missing])
        for unit, vector in zip(missing, vectors):
            self._vectors[unit[0].event_id] = _normalize(vector)
        self.encoded += len(missing)

    def __call__(self, units: Sequence[Unit], index: int) -> float:
        left = units[max(0, index - self.window) : index]
        right = units[index : index + self.window]
        self.prepare([*left, *right])
        mean_left = _mean([self._vectors[unit[0].event_id] for unit in left])
        mean_right = _mean([self._vectors[unit[0].event_id] for unit in right])
        dot = sum(a * b for a, b in zip(mean_left, mean_right))
        denom = math.sqrt(sum(a * a for a in mean_left)) * math.sqrt(sum(b * b for b in mean_right))
        return 1.0 - dot / (denom or 1.0)


class CutLedger:
    """``{"version": 1, "sessions": {digest: {"cuts": [...], "segmenter": "2", ...}}}``."""

    _lock = threading.Lock()

    def __init__(self, path: Path, *, embed_model: str = "") -> None:
        self.path = Path(path)
        self.embed_model = embed_model
        data = json_load(self.path, default=None)
        sessions = data.get("sessions") if isinstance(data, dict) else None
        self._sessions: Dict[str, Dict[str, Any]] = dict(sessions or {})
        self._dirty = False
        self.hits = 0
        self.recorded = 0

    def lookup(self, digest: str, _session: Any = None) -> Optional[List[str]]:
        entry = self._sessions.get(digest)
        if not entry or entry.get("segmenter") != SEGMENTER_V2:
            return None
        self.hits += 1
        return [str(item) for item in entry.get("cuts") or []]

    def record(self, digest: str, cuts: List[str]) -> None:
        if digest in self._sessions:
            return
        self._sessions[digest] = {
            "cuts": list(cuts),
            "segmenter": SEGMENTER_V2,
            "embed_model": self.embed_model,
            "recorded_at": time.time(),
        }
        self._dirty = True
        self.recorded += 1

    def save(self) -> None:
        if self._dirty:
            with self._lock:
                atomic_json_dump({"version": 1, "sessions": self._sessions}, self.path)
            self._dirty = False


__all__ = ["CutLedger", "EmbeddingScorer", "unit_text"]
