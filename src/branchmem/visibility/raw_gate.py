"""Read gate for raw events.

Two modes:

- ``strict``: anything that cannot be proven to be on the current branch is hidden.
- ``compat``: for stores whose history mostly predates topology. Missing-information
  reasons (``COMPAT_PASS_REASONS``) are let through but counted; once an anchor is
  known, a node off the anchor path (a sibling branch) is still hidden. namespace,
  perspective, as-of and disallowed trees are never relaxed.

The gate is a pure filter and writes no state. If topology evaluation itself fails,
compat lets everything through with ``gate_error`` (a topology bug must not empty a
search) and strict hides everything.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..events import normalize_payload
from ..registry import TreeRegistry
from ..tree import TreeBuildResult, build_tree
from .anchor_resolution import CursorLookup, resolve_gate_anchors
from .context import MODE_COMPAT, MODE_STRICT, build_read_context, event_visibility_reason

#: Missing-information reasons that compat mode lets through (and counts).
COMPAT_PASS_REASONS = frozenset(
    {"legacy_unknown", "tree_unresolvable", "anchor_missing", "topology_incomplete"}
)

TreeReader = Callable[[str], Sequence[Mapping[str, Any]]]
_LOG = logging.getLogger("branchmem.read_gate")


@dataclass
class GateOutcome:
    """Filter result; ``reasons`` counts both passes and hides by reason code."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def visible(self) -> int:
        return len(self.events)

    @property
    def hidden(self) -> int:
        return sum(n for reason, n in self.reasons.items() if not reason.startswith("compat_pass:"))


def load_trees(
    conversation_ids: Iterable[str],
    *,
    registry: Optional[TreeRegistry] = None,
    tree_reader: Optional[TreeReader] = None,
    fallback_events: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, TreeBuildResult]:
    """Trees for the given conversations, keyed by tree ID.

    Sources in order: a resident ``registry``, a ``tree_reader`` (full build per
    call), or the events being filtered themselves.
    """
    trees: Dict[str, TreeBuildResult] = {}
    for conversation_id in conversation_ids:
        if registry is not None:
            tree = registry.tree(conversation_id)
        else:
            if tree_reader is not None:
                payloads = tree_reader(conversation_id) or []
            else:
                payloads = [
                    p for p in fallback_events if str(p.get("conversation_id") or "") == conversation_id
                ]
            events = [normalize_payload(p) for p in payloads if isinstance(p, Mapping)]
            tree = build_tree(events, conversation_id=conversation_id)
        trees[tree.tree_id] = tree
    return trees


def filter_raw_events(
    events: Sequence[Mapping[str, Any]],
    *,
    conversation_id: str = "",
    mode: str = MODE_COMPAT,
    active_path_ids: Iterable[str] = (),
    anchors: Optional[Mapping[str, str]] = None,
    registry: Optional[TreeRegistry] = None,
    tree_reader: Optional[TreeReader] = None,
    cursors: Optional[CursorLookup] = None,
    logger: Optional[logging.Logger] = None,
) -> GateOutcome:
    """Filter raw events with the shared visibility predicate, keeping input order.

    ``conversation_id`` + ``active_path_ids`` describe the request's own branch (the
    node IDs the client is showing). Events of other conversations are judged by their
    own trees and anchors; they never share this path.
    """
    raw = [dict(item) for item in events if isinstance(item, Mapping)]
    if not raw:
        return GateOutcome()
    sink = logger or _LOG
    try:
        # History without node IDs: fill them from the resident tree (e.g. overlay).
        judged = registry.with_known_nodes(raw) if registry is not None else raw
        normalized = [normalize_payload(payload) for payload in judged]
        node_conversations = sorted({e.conversation_id for e in normalized if e.client_node_id})
        trees = load_trees(
            node_conversations, registry=registry, tree_reader=tree_reader, fallback_events=judged
        )
        resolved = resolve_gate_anchors(
            trees,
            conversation_id=conversation_id,
            active_path_ids=active_path_ids,
            explicit=anchors,
            cursors=cursors,
        )
        ctx = build_read_context(mode=mode, anchors_by_tree=resolved, allowed_trees=trees.keys())
        # Compute the real (strict) reason first, then let the compat policy decide.
        strict_ctx = replace(ctx, mode=MODE_STRICT)
        visible: List[Dict[str, Any]] = []
        reasons: Counter = Counter()
        for payload, event in zip(raw, normalized):
            reason = event_visibility_reason(strict_ctx, event, trees)
            if reason is None:
                visible.append(payload)
            elif mode != MODE_STRICT and reason in COMPAT_PASS_REASONS:
                visible.append(payload)
                reasons[f"compat_pass:{reason}"] += 1
            else:
                reasons[reason] += 1
        outcome = GateOutcome(events=visible, reasons=dict(sorted(reasons.items())))
        level = logging.INFO if outcome.hidden else logging.DEBUG
        sink.log(
            level,
            "read_gate mode=%s conversation=%s visible=%d hidden=%d reasons=%s",
            mode, conversation_id or "-", outcome.visible, outcome.hidden, outcome.reasons,
        )
        return outcome
    except Exception as exc:  # noqa: BLE001 - a topology failure must not empty a search
        sink.warning("read_gate topology evaluation failed: %s: %s", type(exc).__name__, exc)
        if mode == MODE_STRICT:
            return GateOutcome(events=[], reasons={"gate_error": len(raw)})
        return GateOutcome(events=raw, reasons={"compat_pass:gate_error": len(raw)})


__all__ = ["COMPAT_PASS_REASONS", "GateOutcome", "filter_raw_events", "load_trees"]
