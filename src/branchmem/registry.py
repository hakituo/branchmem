"""Per-conversation resident trees: full build on first use, incremental after that.

The tree structure is fully determined by parent pointers in the raw events, so it can
be maintained while writing instead of rebuilt on every read:

- ``snapshot(cid)``: on first use, read all events through ``reader`` and build;
  afterwards return the cached result.
- ``on_append(payloads)``: call after events are stored. Loaded conversations feed
  the new events to the same ``TreeBuilder`` (a child that arrives first waits in
  pending and is attached when its parent arrives).
- ``forget(cid)``: drop a conversation, e.g. after it was deleted.

Incremental and full builds share one builder and one normalization, so their
canonical hashes match. ``refresh_seconds`` only covers other processes writing the
store directly. Nothing is persisted: raw events stay the single source of truth.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .events import NormalizedEvent, normalize_payload
from .tree import TreeBuilder, TreeBuildResult

#: ``conversation_id -> raw payloads`` (any event store)
Reader = Callable[[str], Sequence[Mapping[str, Any]]]
#: ``conversation_id -> overlay`` (see ``sources.overlay``) or ``None``
OverlayLoader = Callable[[str], Optional[Mapping[str, Any]]]
#: ``conversation_id -> {orphan_root: graft_target}`` (see ``cursors``)
GraftLoader = Callable[[str], Mapping[str, str]]
EventIndex = Dict[str, str]

DEFAULT_REFRESH_SECONDS = 1800.0


@dataclass
class _Entry:
    builder: TreeBuilder
    loaded_at: float
    snapshot: Optional[Tuple[TreeBuildResult, EventIndex]] = None
    event_count: int = 0
    # event -> node aliases. A re-delivered event is deduplicated by the builder and
    # never enters ``node.event_ids``, but it still belongs to that node; without the
    # alias a lookup by its event ID would treat it as history with unknown topology.
    aliases: EventIndex = field(default_factory=dict)


class TreeRegistry:
    def __init__(
        self,
        reader: Reader,
        *,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        overlay_loader: Optional[OverlayLoader] = None,
        graft_loader: Optional[GraftLoader] = None,
    ) -> None:
        self._reader = reader
        self._overlay_loader = overlay_loader
        self._graft_loader = graft_loader
        self._refresh_seconds = float(refresh_seconds)
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.RLock()
        self.full_loads = 0  # observability: number of full builds

    def set_graft_loader(self, loader: Optional[GraftLoader]) -> None:
        with self._lock:
            self._graft_loader = loader
            for entry in self._entries.values():
                entry.snapshot = None

    # ---- reads -------------------------------------------------------------
    def snapshot(self, conversation_id: str) -> Tuple[TreeBuildResult, EventIndex]:
        cid = str(conversation_id or "")
        with self._lock:
            entry = self._entries.get(cid)
            if entry is None or self._expired(entry):
                entry = self._load(cid)
            if entry.snapshot is None:
                tree = entry.builder.build(grafts=self._grafts(cid))
                index = {e: n for e, n in entry.aliases.items() if n in tree.nodes}
                index.update(_event_index(tree))
                entry.snapshot = (tree, index)
            return entry.snapshot

    def tree(self, conversation_id: str) -> TreeBuildResult:
        return self.snapshot(conversation_id)[0]

    # ---- writes ------------------------------------------------------------
    def on_append(self, payloads: Iterable[Mapping[str, Any]]) -> None:
        """Incremental update after a write; unloaded conversations wait for first read."""
        with self._lock:
            for payload in payloads:
                if not isinstance(payload, Mapping):
                    continue
                entry = self._entries.get(str(payload.get("conversation_id") or ""))
                if entry is None:
                    continue
                event = normalize_payload(payload)
                entry.builder.add(event)
                _remember_alias(entry, event)
                entry.event_count += 1
                entry.snapshot = None

    def with_known_nodes(self, payloads: Sequence[Mapping[str, Any]]) -> list:
        """Copies of ``payloads`` with node fields filled from the tree (e.g. from an overlay).

        A conversation is only loaded for this when it is already loaded or has an
        overlay, so pure legacy history never pays for a full build.
        """
        items = [dict(payload) for payload in payloads if isinstance(payload, Mapping)]
        wanted: Dict[str, list] = {}
        for item in items:
            if not item.get("client_node_id") and item.get("event_id"):
                wanted.setdefault(str(item.get("conversation_id") or ""), []).append(item)
        for conversation_id, group in wanted.items():
            if not conversation_id or not self._worth_loading(conversation_id):
                continue
            tree, index = self.snapshot(conversation_id)
            for item in group:
                node_id = index.get(str(item.get("event_id")))
                if node_id:
                    item["client_node_id"] = node_id
                    item["parent_node_id"] = tree.nodes[node_id].parent_id
        return items

    def _worth_loading(self, conversation_id: str) -> bool:
        if self.loaded(conversation_id):
            return True
        return self._overlay_loader is not None and bool(self._overlay_loader(conversation_id))

    def invalidate(self, conversation_id: str) -> None:
        """Grafts changed: rebuild on next snapshot without re-reading events."""
        with self._lock:
            entry = self._entries.get(str(conversation_id or ""))
            if entry is not None:
                entry.snapshot = None

    def _grafts(self, conversation_id: str) -> Mapping[str, str]:
        if self._graft_loader is None:
            return {}
        try:
            return self._graft_loader(conversation_id) or {}
        except Exception:  # noqa: BLE001 - grafts are a fallback; build the plain tree
            return {}

    def forget(self, conversation_id: str) -> None:
        with self._lock:
            self._entries.pop(str(conversation_id or ""), None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.full_loads = 0

    def loaded(self, conversation_id: str) -> bool:
        with self._lock:
            return str(conversation_id or "") in self._entries

    # ---- internals ---------------------------------------------------------
    def _expired(self, entry: _Entry) -> bool:
        return self._refresh_seconds > 0 and time.monotonic() - entry.loaded_at > self._refresh_seconds

    def _load(self, conversation_id: str) -> _Entry:
        payloads = [p for p in (self._reader(conversation_id) or []) if isinstance(p, Mapping)]
        if self._overlay_loader is not None:
            overlay = self._overlay_loader(conversation_id)
            if overlay:
                from .sources.overlay import apply_overlay

                payloads = apply_overlay(payloads, overlay, conversation_id=conversation_id)
        events = [normalize_payload(payload) for payload in payloads]
        builder = TreeBuilder(conversation_id=conversation_id)
        # Same stable order as ``build_tree``; fewer pending replays, same result.
        for event in sorted(events, key=lambda item: (item.occurred_at, item.event_id)):
            builder.add(event)
        entry = _Entry(builder=builder, loaded_at=time.monotonic(), event_count=len(events))
        for event in events:
            _remember_alias(entry, event)
        self._entries[conversation_id] = entry
        self.full_loads += 1
        return entry


def _remember_alias(entry: _Entry, event: NormalizedEvent) -> None:
    if event.event_id and event.client_node_id:
        entry.aliases[event.event_id] = event.client_node_id


def _event_index(tree: TreeBuildResult) -> EventIndex:
    index: EventIndex = {}
    for node_id, node in tree.nodes.items():
        for event_id in node.event_ids:
            index[str(event_id)] = node_id
    return index


__all__ = ["TreeRegistry"]
