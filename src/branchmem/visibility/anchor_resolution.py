"""Anchor resolution for the read gate.

Priority per tree:

1. an explicit request anchor (argument, or the context variable from ``anchors``);
2. the deepest node of the client-supplied active path, for the request's own
   conversation;
3. a persisted cursor, but only one whose parent chain reaches a real root. A cursor
   sitting on an orphan chain would hide the whole main line as
   ``not_on_anchor_path``, so it is skipped and the tree gets no anchor instead.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

from ..anchors import get_request_anchors
from ..tree import TreeBuildResult

#: ``conversation_id -> cursor candidates in priority order`` (``CursorStore.candidates``)
CursorLookup = Callable[[str], Sequence[str]]


def resolve_gate_anchors(
    trees: Mapping[str, TreeBuildResult],
    *,
    conversation_id: str = "",
    active_path_ids: Iterable[str] = (),
    explicit: Optional[Mapping[str, str]] = None,
    cursors: Optional[CursorLookup] = None,
) -> Dict[str, str]:
    explicit = dict(explicit) if explicit is not None else (get_request_anchors() or {})
    allowed_ids = {str(item).strip() for item in active_path_ids or () if str(item).strip()}
    anchors: Dict[str, str] = {}
    for tree_id, tree in trees.items():
        anchor = str(explicit.get(tree_id) or "").strip()
        if anchor:
            anchors[tree_id] = anchor
            continue
        if conversation_id and tree.conversation_id == conversation_id and allowed_ids:
            deepest = _deepest_node(tree, allowed_ids)
            if deepest:
                anchors[tree_id] = deepest
                continue
        cursor = _grounded_cursor(tree, cursors)
        if cursor:
            anchors[tree_id] = cursor
    return anchors


def _grounded_cursor(tree: TreeBuildResult, cursors: Optional[CursorLookup]) -> str:
    if cursors is None:
        return ""
    try:
        for candidate in cursors(tree.conversation_id) or ():
            if candidate and candidate in tree.nodes and tree.is_grounded(candidate):
                return candidate
    except Exception:  # noqa: BLE001 - an unreadable cursor counts as "no anchor"
        return ""
    return ""


def _deepest_node(tree: TreeBuildResult, allowed_ids: Iterable[str]) -> str:
    best, best_depth = "", -1
    for node_id in allowed_ids:
        if node_id in tree.nodes:
            depth = len(tree.ancestors(node_id))
            if depth > best_depth:
                best, best_depth = node_id, depth
    return best


__all__ = ["CursorLookup", "resolve_gate_anchors"]
