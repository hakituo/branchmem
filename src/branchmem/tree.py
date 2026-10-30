"""Deterministic message tree: the same event set yields the same tree in any arrival order."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .dangling import resolve_dangling
from .events import (
    TOPOLOGY_COMPLETE,
    TOPOLOGY_PARTIAL,
    TOPOLOGY_UNKNOWN,
    NormalizedEvent,
)

ADDED = "added"
DUPLICATE = "duplicate"
PENDING = "pending"
REJECTED = "rejected"


@dataclass
class TreeNode:
    node_id: str
    parent_id: str = ""
    variant_of: str = ""
    event_ids: List[str] = field(default_factory=list)
    turn_ids: List[str] = field(default_factory=list)
    speaker_roles: List[str] = field(default_factory=list)
    content_eligible: bool = False
    occurred_at: float = 0.0


@dataclass
class TreeBuildResult:
    tree_id: str
    conversation_id: str = ""
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    roots: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    unknown_event_ids: List[str] = field(default_factory=list)
    topology_status: str = TOPOLOGY_COMPLETE
    reasons: List[str] = field(default_factory=list)
    #: nodes whose parent never arrived and were re-attached at build time -> reason code
    inferred_parents: Dict[str, str] = field(default_factory=dict)

    # ---- path queries ----------------------------------------------------
    def ancestors(self, node_id: str, *, include_self: bool = True) -> List[str]:
        """Path from the root to ``node_id``; stops at a broken link or cycle (fail-closed)."""
        chain: List[str] = []
        seen: set[str] = set()
        current = node_id if include_self else self.nodes.get(node_id, TreeNode(node_id)).parent_id
        while current:
            if current in seen:
                break
            seen.add(current)
            node = self.nodes.get(current)
            if node is None:
                break
            chain.append(current)
            current = node.parent_id
        chain.reverse()
        return chain

    def is_visible(self, node_id: str, anchor_id: str) -> bool:
        """Only nodes on the anchor's ancestor path are visible; an unknown anchor sees nothing."""
        if not anchor_id or anchor_id not in self.nodes:
            return False
        if node_id not in self.nodes:
            return False
        path = self.ancestors(anchor_id)
        if not path or path[-1] != anchor_id:
            # the anchor's own chain is broken: trust it only up to the break
            return node_id in path
        return node_id in path

    def is_grounded(self, node_id: str) -> bool:
        """True when the parent chain reaches a real root, not a node still waiting for its parent."""
        chain = self.ancestors(node_id)
        if not chain or chain[-1] != node_id:
            return False
        return chain[0] not in set(self.pending)

    def visible_event_ids(self, anchor_id: str) -> set[str]:
        visible: set[str] = set()
        for node_id in self.ancestors(anchor_id):
            node = self.nodes.get(node_id)
            if node is not None:
                visible.update(node.event_ids)
        return visible

    def path_event_ids(self, anchor_id: str, *, eligible_only: bool = False) -> List[str]:
        """Event IDs in path order, for segmentation and extraction."""
        result: List[str] = []
        for node_id in self.ancestors(anchor_id):
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if eligible_only and not node.content_eligible:
                continue
            result.extend(node.event_ids)
        return result

    # ---- audit -----------------------------------------------------------
    def canonical_hash(self) -> str:
        """Canonical hash: depends only on graph structure, not on input order or arrival time."""
        payload = [
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "variant_of": node.variant_of,
                "event_ids": sorted(node.event_ids),
            }
            for node in sorted(self.nodes.values(), key=lambda item: item.node_id)
        ]
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_report(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "conversation_id": self.conversation_id,
            "node_count": len(self.nodes),
            "root_count": len(self.roots),
            "pending_count": len(self.pending),
            "conflict_count": len(self.conflicts),
            "unknown_event_count": len(self.unknown_event_ids),
            "inferred_parent_count": len(self.inferred_parents),
            "topology_status": self.topology_status,
            "reasons": list(self.reasons),
            "canonical_hash": self.canonical_hash(),
        }


class TreeBuilder:
    """Incremental builder; ``build()`` is idempotent and can be called repeatedly."""

    def __init__(self, *, tree_id: str = "", conversation_id: str = "") -> None:
        self.tree_id = tree_id
        self.conversation_id = conversation_id
        self._nodes: Dict[str, TreeNode] = {}
        self._pending: Dict[str, NormalizedEvent] = {}
        self._conflicts: List[str] = []
        self._unknown_event_ids: List[str] = []
        self._seen_event_ids: set[str] = set()
        self._seen_keys: set[str] = set()
        self._reasons: List[str] = []

    # ---- writes ----------------------------------------------------------
    def add(self, event: NormalizedEvent) -> str:
        event_id = event.event_id or event.idempotency_key
        node_id = event.client_node_id
        if not node_id:
            if event_id not in self._seen_event_ids:
                self._seen_event_ids.add(event_id)
                self._unknown_event_ids.append(event_id)
                self._reasons.append(f"event {event_id} has no client node ID; topology unknown")
            return REJECTED
        if event.event_id and event.event_id in self._seen_event_ids:
            return DUPLICATE
        if event.idempotency_key and event.idempotency_key in self._seen_keys:
            return DUPLICATE
        if event.event_id:
            self._seen_event_ids.add(event.event_id)
        if event.idempotency_key:
            self._seen_keys.add(event.idempotency_key)

        node = self._nodes.get(node_id)
        if node is None:
            node = TreeNode(node_id=node_id, occurred_at=event.occurred_at)
            self._nodes[node_id] = node
        node.event_ids.append(event.event_id or event.idempotency_key)
        if event.turn_id and event.turn_id not in node.turn_ids:
            node.turn_ids.append(event.turn_id)
        if event.speaker_role and event.speaker_role not in node.speaker_roles:
            node.speaker_roles.append(event.speaker_role)
        node.content_eligible = node.content_eligible or event.content_eligible
        node.occurred_at = node.occurred_at or event.occurred_at

        parent_id = event.parent_node_id
        variant_of = event.variant_of_node_id
        if variant_of:
            old = self._nodes.get(variant_of)
            if old is None:
                if not parent_id:
                    # no parent and the old version is missing: wait for it and inherit its parent
                    self._pending[node_id] = event
                    return PENDING
                # With a parent, attach right away: the parent defines the branch and
                # variant_of is only a hint. The old version may never arrive (edited
                # on the client, or the first send failed); waiting for it would
                # stall the whole rest of the conversation.
                node.variant_of = variant_of
                old = None
            elif parent_id and old.parent_id and parent_id != old.parent_id:
                self._conflicts.append(
                    f"node {node_id} parent={parent_id} disagrees with old version {variant_of} "
                    f"parent={old.parent_id}"
                )
            if old is not None:
                parent_id = parent_id or old.parent_id
                node.variant_of = variant_of

        if parent_id and parent_id not in self._nodes and parent_id != node_id:
            self._pending[node_id] = event
            return PENDING

        node.parent_id = parent_id
        self._pending.pop(node_id, None)
        self._drain_pending()
        return ADDED

    def _drain_pending(self) -> None:
        """Child-before-parent: replay pending nodes in a stable order once parents arrive."""
        progressed = True
        while progressed:
            progressed = False
            for node_id in sorted(
                self._pending,
                key=lambda key: (self._pending[key].occurred_at, key),
            ):
                event = self._pending[node_id]
                old = self._nodes.get(event.variant_of_node_id) if event.variant_of_node_id else None
                parent_id = event.parent_node_id or (old.parent_id if old else "")
                if parent_id and parent_id not in self._nodes:
                    continue
                node = self._nodes.get(node_id)
                if node is None:
                    continue
                node.parent_id = parent_id
                if old is not None:
                    node.variant_of = event.variant_of_node_id
                self._pending.pop(node_id, None)
                progressed = True

    # ---- result ----------------------------------------------------------
    def build(self, *, grafts: Optional[Mapping[str, str]] = None) -> TreeBuildResult:
        """``grafts``: pending node -> attach target, recorded by the cursor guard (see cursors.py)."""
        nodes = dict(self._nodes)
        inferred = resolve_dangling(nodes, self._pending, grafts or {})
        roots = sorted(
            (node_id for node_id, node in nodes.items() if not node.parent_id),
            key=lambda node_id: (nodes[node_id].occurred_at, node_id),
        )
        pending = sorted(node_id for node_id in self._pending if node_id not in inferred)
        for node_id in pending:
            if node_id not in self._reasons:
                parent = self._pending[node_id].parent_node_id
                self._reasons.append(f"node {node_id}: parent {parent or 'unknown'} has not arrived")

        status = TOPOLOGY_COMPLETE
        if pending or self._conflicts or inferred:
            status = TOPOLOGY_PARTIAL
        elif len(roots) > 1:
            status = TOPOLOGY_PARTIAL
            self._reasons.append(f"{len(roots)} roots; the parent chain is not unique")
        elif self._unknown_event_ids and not self._nodes:
            status = TOPOLOGY_UNKNOWN
        elif self._unknown_event_ids:
            status = TOPOLOGY_PARTIAL

        reasons = list(self._reasons) + [
            f"node {node_id}: dangling parent re-attached to {nodes[node_id].parent_id} by {code}"
            for node_id, code in sorted(inferred.items())
        ]
        tree_id = self.tree_id or compute_tree_id(self.conversation_id, roots)
        return TreeBuildResult(
            tree_id=tree_id,
            conversation_id=self.conversation_id,
            nodes=nodes,
            roots=roots,
            pending=pending,
            conflicts=list(self._conflicts),
            unknown_event_ids=list(self._unknown_event_ids),
            topology_status=status,
            reasons=list(dict.fromkeys(reasons)),
            inferred_parents=inferred,
        )


def compute_tree_id(conversation_id: str, root_node_ids: Sequence[str]) -> str:
    """``tree_id`` depends on the conversation and its roots, so recomputing gives the same ID."""
    basis = "|".join([str(conversation_id or ""), ",".join(sorted(root_node_ids))])
    return "tree_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def build_tree(
    events: Iterable[NormalizedEvent],
    *,
    conversation_id: str = "",
) -> TreeBuildResult:
    """Build a tree in one go (audit / replay). Input order does not change the result."""
    builder = TreeBuilder(conversation_id=conversation_id)
    # A stable order reduces pending replays; correctness does not depend on it.
    for event in sorted(events, key=lambda item: (item.occurred_at, item.event_id)):
        builder.add(event)
    return builder.build()
