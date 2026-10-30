"""Per-tree anchors: the control plane that says which branch a reader is on.

``anchors_by_tree`` is a hard constraint: every tree that may be read needs an
explicit anchor, and a tree without one returns nothing (fail-closed). Switching
branches only moves that tree's anchor. Events and derived notes never change state,
so there is no global ``inactive`` flag: "which branch is current" is reader state,
not a property of the history.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from ._util import atomic_json_dump

ANCHOR_STORE_VERSION = 1

_request_anchors: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "branchmem_request_anchors_by_tree", default=None
)


def _clean(anchors: Optional[Mapping[str, str]]) -> Dict[str, str]:
    return {
        str(tree_id).strip(): str(node_id).strip()
        for tree_id, node_id in (anchors or {}).items()
        if str(tree_id).strip() and str(node_id or "").strip()
    }


def set_request_anchors(anchors: Optional[Mapping[str, str]]) -> Token:
    """Bind this request's explicit anchors (they win over persisted cursors)."""
    return _request_anchors.set(_clean(anchors) or None)


def reset_request_anchors(token: Token) -> None:
    _request_anchors.reset(token)


def get_request_anchors() -> Optional[Dict[str, str]]:
    value = _request_anchors.get()
    return dict(value) if value else None


@contextmanager
def bound_anchors(anchors: Optional[Mapping[str, str]]) -> Iterator[None]:
    """Bind anchors for the duration of a call chain, then restore."""
    token = set_request_anchors(anchors)
    try:
        yield
    finally:
        reset_request_anchors(token)


class AnchorStore:
    """Persisted anchors keyed by tree (or conversation).

    The snapshot file is only a cache; the append-only event log is the audit record.
    """

    def __init__(self, base_dir: Path, *, scope: str = "default") -> None:
        self.base_dir = Path(base_dir)
        self.scope = str(scope or "default")
        self._lock = threading.RLock()

    @property
    def snapshot_path(self) -> Path:
        return self.base_dir / "anchors.json"

    @property
    def events_path(self) -> Path:
        return self.base_dir / "anchor_events.jsonl"

    def load(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._load_snapshot())

    def get(self, tree_id: str) -> str:
        return self._load_snapshot().get(str(tree_id or ""), "")

    def set(
        self,
        tree_id: str,
        node_id: str,
        *,
        reason: str = "user_request",
        source: str = "runtime",
        changed_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        tree_key = str(tree_id or "").strip()
        node_key = str(node_id or "").strip()
        if not tree_key or not node_key:
            raise ValueError("an anchor needs both tree_id and node_id")
        stamp = float(changed_at if changed_at is not None else time.time())
        with self._lock:
            anchors = self._load_snapshot()
            previous = anchors.get(tree_key, "")
            anchors[tree_key] = node_key
            record = {
                "type": "anchor_changed",
                "tree_id": tree_key,
                "node_id": node_key,
                "previous": previous,
                "reason": str(reason or ""),
                "source": str(source or ""),
                "changed_at": stamp,
            }
            self.append_event(record)
            self._write_snapshot(anchors, stamp)
            return record

    def append_event(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    # ---- internals ---------------------------------------------------------
    def _load_snapshot(self) -> Dict[str, str]:
        try:
            raw = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        anchors = raw.get("anchors") if isinstance(raw, dict) else None
        if not isinstance(anchors, dict):
            return {}
        result: Dict[str, str] = {}
        for tree_id, value in anchors.items():
            node_id = value.get("node_id") if isinstance(value, dict) else value
            if str(tree_id).strip() and str(node_id or "").strip():
                result[str(tree_id).strip()] = str(node_id).strip()
        return result

    def _write_snapshot(self, anchors: Mapping[str, str], stamp: float) -> None:
        payload = {
            "version": ANCHOR_STORE_VERSION,
            "scope": self.scope,
            "updated_at": stamp,
            "anchors": {tree_id: {"node_id": node_id} for tree_id, node_id in sorted(anchors.items())},
        }
        atomic_json_dump(payload, self.snapshot_path)


def resolve_anchors_by_tree(
    tree_ids: Iterable[str],
    *,
    explicit: Optional[Mapping[str, str]] = None,
    store: Optional[AnchorStore] = None,
) -> Dict[str, Optional[str]]:
    """Explicit request anchors first, then persisted ones; ``None`` when neither exists.

    A tree mapped to ``None`` must be skipped (and the reason recorded) by the reader;
    it must never default to "the whole tree is visible".
    """
    request = dict(explicit) if explicit is not None else (get_request_anchors() or {})
    stored = store.load() if store is not None else {}
    resolved: Dict[str, Optional[str]] = {}
    for tree_id in tree_ids:
        key = str(tree_id or "").strip()
        if key:
            anchor = str(request.get(key) or "").strip() or str(stored.get(key) or "").strip()
            resolved[key] = anchor or None
    return resolved
