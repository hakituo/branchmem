"""Re-attach dangling parents: pending nodes whose parent never arrived get an attach point.

Works on a copy of the ``TreeBuilder.build()`` result and never changes builder state;
the outcome depends only on the event set and the graft records, not arrival order.
Once the real parent arrives, ``add`` attaches the node normally and this step no
longer applies.

1. ``segment_alias``: a client splits one multi-bubble reply into ``X``, ``X-1``,
   ``X-2``…, while the server only stored ``X``. A user message whose parent is the
   last bubble is attached to the assistant node ``X``.
2. ``cursor_graft``: a graft recorded by the cursor guard (orphan root -> the cursor at
   that time), see ``cursors.py``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Mapping

if TYPE_CHECKING:
    from .tree import TreeNode

INFER_SEGMENT = "segment_alias"
INFER_GRAFT = "cursor_graft"

_SEGMENT_SUFFIX = re.compile(r"^(?P<base>.+)-(?P<index>\d{1,3})$")


def resolve_dangling(
    nodes: Dict[str, TreeNode],
    pending: Mapping[str, Any],
    grafts: Mapping[str, str],
) -> Dict[str, str]:
    """``pending``: pending node -> its event. Replaces re-attached nodes in ``nodes`` and returns node -> reason code."""
    inferred: Dict[str, str] = {}
    for node_id in sorted(pending):
        parent = str(getattr(pending[node_id], "parent_node_id", "") or "")
        if not parent or node_id not in nodes:
            continue
        target, code = segment_base(nodes, parent), INFER_SEGMENT
        if not target:
            target, code = str(grafts.get(node_id) or ""), INFER_GRAFT
        if not target or target not in nodes or _reaches(nodes, target, node_id):
            continue
        nodes[node_id] = replace(nodes[node_id], parent_id=target)
        inferred[node_id] = code
    return inferred


def segment_base(nodes: Mapping[str, TreeNode], parent_id: str) -> str:
    """``X-<n>`` where ``X`` is an assistant node in the tree -> ``X``; otherwise empty."""
    match = _SEGMENT_SUFFIX.match(str(parent_id or ""))
    if not match:
        return ""
    base = nodes.get(match.group("base"))
    return base.node_id if base is not None and "assistant" in base.speaker_roles else ""


def _reaches(nodes: Mapping[str, TreeNode], start: str, target: str) -> bool:
    """Whether walking up from ``start`` reaches ``target`` (prevents creating a cycle)."""
    seen: set[str] = set()
    current = start
    while current and current not in seen:
        if current == target:
            return True
        seen.add(current)
        node = nodes.get(current)
        current = node.parent_id if node is not None else ""
    return False


__all__ = ["INFER_GRAFT", "INFER_SEGMENT", "resolve_dangling", "segment_base"]
