"""Per-conversation branch cursors: reads without a request path still know the branch.

An online request can carry its anchor (the node the user is looking at). Background
work, such as nightly extraction, proactive messages or cross-conversation search,
has no such path. Without a cursor those reads could only let the whole tree through,
siblings included.

- Write: after events with node IDs are stored, call ``advance(payloads)``. The cursor
  moves to the newest node, and one audit line is appended only when it actually
  moves. A new node that is not a descendant of the old cursor is logged as
  ``branch_switch`` (the user switched branches on the client and continued).
- Read: ``cursor_for(conversation_id)``. Without a stored cursor it falls back to an
  optional ``overlay_leaf`` callback (e.g. the active leaf of a back-filled history).

Orphan guard: when the new node's parent chain ends at a node still waiting for its
parent, it is not a branch switch. Jumping the cursor onto an orphan root would make
every read without a request path hide the whole main line as ``not_on_anchor_path``.
If the old cursor is a grounded node, the orphan root is grafted under it
(``cursor_grafts.json``, ``orphan_grafted``) and the cursor advances; if the old cursor
is not reliable either, the cursor stays (``orphan_held``). A graft only applies while
the real parent is missing; once it arrives the real parent wins (see ``dangling``).

Cursors are keyed by conversation ID, not tree ID: the tree ID is derived from the
roots and is not known on the write path, and one conversation is one tree. A cursor
is reader state only. Switching back to an old branch just moves it; no event or note
is ever marked inactive.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ._util import atomic_json_dump
from .anchors import AnchorStore
from .registry import TreeRegistry
from .sources.overlay import is_structural_event_id

REASON_TURN = "turn_written"
REASON_SWITCH = "branch_switch"
REASON_GRAFT = "orphan_grafted"
REASON_HELD = "orphan_held"

GRAFTS_FILE = "cursor_grafts.json"


class CursorStore:
    """Cursors for one storage directory.

    Passing ``registry`` wires the graft records into tree builds and lets ``advance``
    tell a real branch switch from an orphan chain. Without a registry every move is
    recorded as a plain ``turn_written``.
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        registry: Optional[TreeRegistry] = None,
        overlay_leaf: Optional[Callable[[str], str]] = None,
        source: str = "event_store",
    ) -> None:
        self.store = AnchorStore(base_dir, scope="cursors")
        self.registry = registry
        self._overlay_leaf = overlay_leaf
        self._source = source
        self._lock = threading.RLock()
        self._snapshot: Optional[Tuple[float, Dict[str, str]]] = None
        self._grafts: Optional[Tuple[float, Dict[str, Dict[str, str]]]] = None
        if registry is not None:
            registry.set_graft_loader(self.grafts_for)

    @property
    def grafts_path(self) -> Path:
        return self.store.base_dir / GRAFTS_FILE

    # ---- reads -------------------------------------------------------------
    def _load(self) -> Dict[str, str]:
        """Cached by snapshot mtime: lookups happen per memory item during search."""
        try:
            mtime = self.store.snapshot_path.stat().st_mtime
        except OSError:
            mtime = -1.0
        with self._lock:
            if self._snapshot is not None and self._snapshot[0] == mtime:
                return self._snapshot[1]
        anchors = self.store.load() if mtime >= 0 else {}
        with self._lock:
            self._snapshot = (mtime, anchors)
        return anchors

    def cursor_for(self, conversation_id: str) -> str:
        cid = str(conversation_id or "").strip()
        if not cid:
            return ""
        return self._load().get(cid, "") or self.overlay_leaf(cid)

    def overlay_leaf(self, conversation_id: str) -> str:
        if self._overlay_leaf is None or not conversation_id:
            return ""
        return str(self._overlay_leaf(conversation_id) or "")

    def candidates(self, conversation_id: str) -> List[str]:
        """Anchor candidates in priority order (used by ``resolve_gate_anchors``)."""
        return [self.cursor_for(conversation_id), self.overlay_leaf(conversation_id)]

    # ---- writes ------------------------------------------------------------
    def advance(self, payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Move cursors after a write; returns the records written (conversation -> record)."""
        latest: Dict[str, Mapping[str, Any]] = {}
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            node_id = str(payload.get("client_node_id") or "").strip()
            if not node_id or is_structural_event_id(str(payload.get("event_id") or "")):
                continue
            conversation_id = str(payload.get("conversation_id") or "").strip()
            if conversation_id:
                latest[conversation_id] = payload  # write order: the last one is newest
        written: Dict[str, Dict[str, Any]] = {}
        for conversation_id, payload in latest.items():
            node_id = str(payload["client_node_id"]).strip()
            previous = self.cursor_for(conversation_id)
            if previous == node_id:
                continue
            reason, orphan_root = self._judge(conversation_id, previous, node_id)
            if reason == REASON_HELD:
                self._audit(conversation_id, "anchor_held", node_id, previous)
                continue
            if orphan_root:
                self._add_graft(conversation_id, orphan_root, previous)
            record = self.store.set(conversation_id, node_id, reason=reason, source=self._source)
            with self._lock:
                self._snapshot = None
            written[conversation_id] = record
        return written

    def _judge(self, conversation_id: str, previous: str, node_id: str) -> Tuple[str, str]:
        """(reason code, orphan root to graft). Only grounded nodes count as a switch."""
        if not previous or self.registry is None:
            return REASON_TURN, ""
        try:
            tree = self.registry.tree(conversation_id)
        except Exception:  # noqa: BLE001 - cannot judge: record a plain move
            return REASON_TURN, ""
        if node_id not in tree.nodes:
            return REASON_TURN, ""
        path = tree.ancestors(node_id)
        if previous in path:
            return REASON_TURN, ""  # moving down the same chain (orphan chains included)
        if tree.is_grounded(node_id):
            return REASON_SWITCH, ""
        if previous in tree.nodes and tree.is_grounded(previous):
            return REASON_GRAFT, path[0]
        return REASON_HELD, ""

    def server_message_fields(self, conversation_id: str, node_id: str) -> Dict[str, str]:
        """Node fields for a server-originated message (e.g. proactive), under the cursor.

        Without a cursor no fields are returned; otherwise the message would become a
        new root, i.e. a new world line.
        """
        node = str(node_id or "").strip()
        parent = self.cursor_for(conversation_id) if node else ""
        if not node or not parent or parent == node:
            return {}
        return {"client_node_id": node, "parent_node_id": parent}

    # ---- orphan grafts -----------------------------------------------------
    def _load_grafts(self) -> Dict[str, Dict[str, str]]:
        try:
            mtime = self.grafts_path.stat().st_mtime
        except OSError:
            return {}
        with self._lock:
            if self._grafts is not None and self._grafts[0] == mtime:
                return self._grafts[1]
        try:
            raw = json.loads(self.grafts_path.read_text(encoding="utf-8")).get("grafts") or {}
        except (OSError, ValueError, AttributeError):
            raw = {}
        grafts = {
            str(cid): {str(k): str(v) for k, v in items.items() if k and v}
            for cid, items in raw.items()
            if isinstance(items, dict)
        }
        with self._lock:
            self._grafts = (mtime, grafts)
        return grafts

    def grafts_for(self, conversation_id: str) -> Dict[str, str]:
        """Orphan root -> graft target; applied only to nodes that are still dangling."""
        cid = str(conversation_id or "").strip()
        return dict(self._load_grafts().get(cid) or {}) if cid else {}

    def _add_graft(self, conversation_id: str, orphan_root: str, parent: str) -> None:
        with self._lock:
            grafts = {cid: dict(items) for cid, items in self._load_grafts().items()}
            grafts.setdefault(conversation_id, {})[orphan_root] = parent
            atomic_json_dump({"version": 1, "grafts": grafts}, self.grafts_path)
            self._grafts = None
        self._audit(conversation_id, "graft_added", orphan_root, parent)
        if self.registry is not None:
            self.registry.invalidate(conversation_id)

    def _audit(self, conversation_id: str, kind: str, node_id: str, other: str) -> None:
        """Append an audit line only; the snapshot is untouched (IDs only, no text)."""
        held = kind == "anchor_held"
        record = {
            "type": kind,
            "tree_id": conversation_id,
            "node_id": node_id,
            "previous" if held else "parent": other,
            "reason": REASON_HELD if held else REASON_GRAFT,
            "source": self._source,
            "changed_at": time.time(),
        }
        try:
            self.store.append_event(record)
        except OSError:
            pass

    def clear_cache(self) -> None:
        with self._lock:
            self._snapshot = None
            self._grafts = None


__all__ = ["REASON_GRAFT", "REASON_HELD", "REASON_SWITCH", "REASON_TURN", "CursorStore"]
