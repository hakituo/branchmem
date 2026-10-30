"""A minimal append-only JSONL event store (reference implementation).

One file per conversation, one event per line, never rewritten. It shows the write
path the rest of the library expects:

1. assign ``event_id`` / ``timestamp`` when missing;
2. for linear platforms, attach a server-side parent (``sources.linear``);
3. merge the normalized fields (``events.enrich_payload``);
4. drop re-deliveries by ``idempotency_key``;
5. append, then notify listeners such as ``TreeRegistry.on_append`` and
   ``CursorStore.advance``.

Any real store can replace it: the library only needs a ``conversation_id -> payloads``
reader and the post-append notifications.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..events import enrich_payload
from .linear import DEFAULT_LINEAR_PLATFORMS, LinearHeadTracker, attach_linear_parent

Listener = Callable[[Sequence[Mapping[str, Any]]], Any]


def _file_name(conversation_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", conversation_id)[:160] or "default"


class JsonlEventStore:
    def __init__(
        self,
        root: Path,
        *,
        linear_platforms: Iterable[str] = DEFAULT_LINEAR_PLATFORMS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self._linear = frozenset(str(p).lower() for p in linear_platforms)
        self._clock = clock
        self._tracker = LinearHeadTracker()
        self._listeners: List[Listener] = []
        self._keys: Dict[str, set] = {}
        self._lock = threading.RLock()

    def path_for(self, conversation_id: str) -> Path:
        return self.root / f"{_file_name(conversation_id)}.jsonl"

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    # ---- writes ------------------------------------------------------------
    def append(self, payloads: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Store events; returns the stored payloads (duplicates are skipped)."""
        stored: List[Dict[str, Any]] = []
        with self._lock:
            for raw in payloads:
                item = dict(raw)
                cid = str(item.get("conversation_id") or "default")
                item["conversation_id"] = cid
                item.setdefault("event_id", uuid.uuid4().hex)
                item.setdefault("timestamp", self._clock())
                path = self.path_for(cid)
                attach_linear_parent(
                    item,
                    file_path=path,
                    platform=str(item.get("platform") or ""),
                    tracker=self._tracker,
                    platforms=self._linear,
                )
                fields = enrich_payload(item, recorded_at=float(item["timestamp"]))
                for key, value in fields.items():
                    item.setdefault(key, value)
                keys = self._known_keys(cid)
                if item["idempotency_key"] in keys:
                    continue
                keys.add(item["idempotency_key"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                if item.get("client_node_id"):
                    self._tracker.advance(path, str(item["client_node_id"]))
                stored.append(item)
        if stored:
            for listener in self._listeners:
                listener(stored)
        return stored

    def _known_keys(self, conversation_id: str) -> set:
        if conversation_id not in self._keys:
            self._keys[conversation_id] = {
                str(p.get("idempotency_key")) for p in self.list_conversation_events(conversation_id)
            }
        return self._keys[conversation_id]

    # ---- reads -------------------------------------------------------------
    def list_conversation_events(self, conversation_id: str) -> List[Dict[str, Any]]:
        path = self.path_for(conversation_id)
        events: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except ValueError:
                            continue  # a torn last line must not hide the rest
        except OSError:
            return []
        return events

    __call__ = list_conversation_events  # usable directly as a TreeRegistry reader

    def conversations(self) -> List[str]:
        result = []
        for path in sorted(self.root.glob("*.jsonl")):
            first = next(iter(self._first_line(path)), None)
            if first:
                result.append(str(first.get("conversation_id") or path.stem))
        return result

    @staticmethod
    def _first_line(path: Path) -> Optional[List[Dict[str, Any]]]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        return [json.loads(line)]
        except (OSError, ValueError):
            return []
        return []
