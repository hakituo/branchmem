"""Side-loaded topology overlay: add tree structure to history without rewriting it.

Typical case: old server-side events were stored without client node IDs, while the
client still holds the full message tree. Events can be matched back to nodes offline
(e.g. by time + text). Raw events are append-only, so the matches are not written back;
they live in one overlay file per conversation and are applied when the tree is built:

- ``nodes``: the client tree, ``node_id -> [parent_id, timestamp, is_user]``; the
  client's topology is authoritative;
- ``events``: server ``event_id -> node_id`` (only confident matches);
- ``active_leaf``: the leaf that was active on the client, usable as the initial cursor.

Nodes without a server event (middle bubbles of a multi-part reply, parents outside the
export window, ...) enter the tree as structural nodes with no text, so the parent chain
stays connected, but ``content_eligible=False`` keeps them out of extraction.
Deleting the overlay file fully reverts. Overlay files hold IDs and timestamps only.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .._util import json_load

OVERLAY_VERSION = 1
STRUCTURAL_EVENT_TYPE = "structural"
STRUCTURAL_PREFIX = "struct:"

_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_cache_lock = threading.Lock()


def _safe_name(conversation_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(conversation_id or ""))[:160] or "default"


def overlay_path(base_dir: Path, conversation_id: str) -> Path:
    return Path(base_dir) / f"{_safe_name(conversation_id)}.json"


def load_overlay(base_dir: Path, conversation_id: str) -> Optional[Dict[str, Any]]:
    """Cached by mtime; ``None`` when the file is missing or malformed."""
    path = overlay_path(base_dir, conversation_id)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = str(path)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    raw = json_load(path, default=None)
    overlay = raw if _valid(raw, conversation_id) else None
    with _cache_lock:
        _cache[key] = (mtime, overlay)
    return overlay


def overlay_loader(base_dir: Path):
    """A ``TreeRegistry`` overlay loader reading ``<base_dir>/<conversation>.json``."""
    return lambda conversation_id: load_overlay(base_dir, conversation_id)


def active_leaf_loader(base_dir: Path):
    """A ``CursorStore`` fallback returning the overlay's ``active_leaf``."""
    return lambda cid: str((load_overlay(base_dir, cid) or {}).get("active_leaf") or "")


def clear_overlay_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _valid(raw: Any, conversation_id: str) -> bool:
    return (
        isinstance(raw, dict)
        and raw.get("version") == OVERLAY_VERSION
        and str(raw.get("conversation_id") or "") == str(conversation_id or "")
        and isinstance(raw.get("nodes"), dict)
        and isinstance(raw.get("events"), dict)
    )


def apply_overlay(
    payloads: Sequence[Mapping[str, Any]],
    overlay: Optional[Mapping[str, Any]],
    *,
    conversation_id: str,
    scope: str = "",
) -> List[Dict[str, Any]]:
    """Fill node fields of events that lack them and add structural nodes.

    Events that already carry a node are kept as they are.
    """
    items = [dict(payload) for payload in payloads if isinstance(payload, Mapping)]
    if not overlay:
        return items
    nodes: Mapping[str, Any] = overlay.get("nodes") or {}
    events: Mapping[str, Any] = overlay.get("events") or {}
    if not scope:
        scope = next((str(p.get("storage_scope")) for p in items if p.get("storage_scope")), "")
    covered: set = set()
    for item in items:
        node_id = str(item.get("client_node_id") or "")
        if not node_id:
            node_id = str(events.get(str(item.get("event_id") or "")) or "")
            if node_id and node_id in nodes:
                item["client_node_id"] = node_id
                item["parent_node_id"] = _parent(nodes, node_id)
                item["topology_source"] = "overlay"
        if node_id:
            covered.add(node_id)
    for node_id, spec in nodes.items():
        if node_id in covered:
            continue
        parent_id, timestamp, is_user = _unpack(spec)
        items.append(
            {
                "event_id": f"{STRUCTURAL_PREFIX}{node_id}",
                "conversation_id": conversation_id,
                "role": "user" if is_user else "assistant",
                "content": "",
                "timestamp": timestamp,
                "event_type": STRUCTURAL_EVENT_TYPE,
                "client_node_id": node_id,
                "parent_node_id": parent_id,
                "storage_scope": scope,
                "topology_source": "overlay",
            }
        )
    return items


def _unpack(spec: Any) -> Tuple[str, float, bool]:
    if isinstance(spec, (list, tuple)) and len(spec) >= 3:
        return str(spec[0] or ""), float(spec[1] or 0.0), bool(spec[2])
    return "", 0.0, False


def _parent(nodes: Mapping[str, Any], node_id: str) -> str:
    return _unpack(nodes.get(node_id))[0]


def is_structural_event_id(event_id: str) -> bool:
    return str(event_id or "").startswith(STRUCTURAL_PREFIX)
