"""Gate for derived records: summaries, facts, profile entries, graph edges.

A derived record is visible only if **every** source event is on the current branch
of its tree. The record's own labels (lineage IDs, "active" flags) are never trusted;
provenance is. One source on a sibling branch hides the whole record.

A record is any mapping with:

- ``conversation_id`` (or ``source_ref.conversation_id``) naming the source tree;
- ``source_event_ids`` (list) and/or ``source_node_ids`` (list); a single
  ``source_ref.event_id`` or ``client_node_id`` is accepted too.

The same gate instance is meant to be reused for one search, so it can serve graph
seeds, every expansion hop and the final result; trees are loaded once per
conversation and verdicts are cached per record ``id``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..events import normalize_event
from ..registry import TreeRegistry
from ..tree import TreeBuildResult
from .anchor_resolution import CursorLookup, resolve_gate_anchors
from .context import MODE_COMPAT, MODE_STRICT, build_read_context, event_visibility_reason
from .raw_gate import COMPAT_PASS_REASONS, TreeReader, load_trees

#: Old records without a conversation cannot be placed in any tree.
CONVERSATION_UNKNOWN = "conversation_unknown"
PASS_REASONS = COMPAT_PASS_REASONS | {CONVERSATION_UNKNOWN}


def record_conversation_id(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("conversation_id") or "").strip()
    if explicit:
        return explicit
    ref = record.get("source_ref")
    return str(ref.get("conversation_id") or "").strip() if isinstance(ref, Mapping) else ""


def record_sources(record: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """``(event_ids, node_ids)`` the record was derived from."""

    def as_list(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in (value or []) if str(item or "").strip()]

    events = as_list(record.get("source_event_ids"))
    ref = record.get("source_ref")
    if not events and isinstance(ref, Mapping):
        events = as_list(ref.get("event_ids") or ref.get("event_id"))
    nodes = as_list(record.get("source_node_ids") or record.get("client_node_id"))
    return events, nodes


class DerivedGate:
    def __init__(
        self,
        *,
        mode: str = MODE_COMPAT,
        conversation_id: str = "",
        active_path_ids: Iterable[str] = (),
        anchors: Optional[Mapping[str, str]] = None,
        registry: Optional[TreeRegistry] = None,
        tree_reader: Optional[TreeReader] = None,
        cursors: Optional[CursorLookup] = None,
        cursor_for: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.mode = MODE_STRICT if mode == MODE_STRICT else MODE_COMPAT
        self.scope_conversation = str(conversation_id or "").strip()
        self.active_path_ids = tuple(active_path_ids or ())
        self.explicit_anchors = dict(anchors or {})
        self._registry = registry
        self._tree_reader = tree_reader
        self._cursors = cursors
        self._cursor_for = cursor_for or (
            (lambda cid: next((c for c in cursors(cid) if c), "")) if cursors else (lambda cid: "")
        )
        self._trees: Dict[str, Optional[TreeBuildResult]] = {}
        self._event_nodes: Dict[str, Dict[str, str]] = {}
        self._verdicts: Dict[str, bool] = {}
        self.reasons: Counter = Counter()

    # ---- public ------------------------------------------------------------
    def is_visible(self, record: Mapping[str, Any]) -> bool:
        record_id = str(record.get("id") or record.get("note_id") or "")
        if record_id and record_id in self._verdicts:
            return self._verdicts[record_id]
        verdict = self._judge(record)
        if record_id:
            self._verdicts[record_id] = verdict
        return verdict

    def filter(self, records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        return [record for record in records if self.is_visible(record)]

    @property
    def hidden(self) -> int:
        return sum(n for reason, n in self.reasons.items() if not reason.startswith("compat_pass:"))

    # ---- judgement ---------------------------------------------------------
    def _judge(self, record: Mapping[str, Any]) -> bool:
        try:
            reason = self.strict_reason(record)
        except Exception:  # noqa: BLE001 - a topology failure must not empty a search
            reason = "gate_error"
            if self.mode != MODE_STRICT:
                self.reasons["compat_pass:gate_error"] += 1
                return True
        if reason is None:
            return True
        if self.mode != MODE_STRICT and reason in PASS_REASONS:
            self.reasons[f"compat_pass:{reason}"] += 1
            return True
        self.reasons[reason] += 1
        return False

    def strict_reason(self, record: Mapping[str, Any]) -> Optional[str]:
        conversation_id = record_conversation_id(record)
        if not conversation_id:
            return CONVERSATION_UNKNOWN
        if (
            conversation_id != self.scope_conversation
            and not self.explicit_anchors
            and not self._cursor_for(conversation_id)
        ):
            return "anchor_missing"  # no anchor source at all: skip reading the tree
        event_ids, node_ids = record_sources(record)
        if not event_ids and not node_ids:
            return "legacy_unknown"
        tree = self._tree_for(conversation_id)
        if tree is None:
            return "tree_unresolvable"
        anchors = resolve_gate_anchors(
            {tree.tree_id: tree},
            conversation_id=self.scope_conversation,
            active_path_ids=self.active_path_ids,
            explicit=self.explicit_anchors,
            cursors=self._cursors,
        )
        ctx = build_read_context(mode=MODE_STRICT, anchors_by_tree=anchors, allowed_trees=[tree.tree_id])
        index = self._event_nodes.get(conversation_id, {})
        nodes = [n if n in tree.nodes else "" for n in node_ids]
        nodes += [index.get(e, "") for e in event_ids]
        passes: List[str] = []
        for node_id in nodes:
            reason = self._node_reason(ctx, tree, conversation_id, node_id)
            if reason is None:
                continue
            if reason not in PASS_REASONS:
                return reason  # a hard reason (e.g. a sibling source) always wins
            passes.append(reason)
        return passes[0] if passes else None

    @staticmethod
    def _node_reason(ctx, tree: TreeBuildResult, conversation_id: str, node_id: str) -> Optional[str]:
        if not node_id:
            return "legacy_unknown"
        node = tree.nodes[node_id]
        event = normalize_event(
            {
                "conversation_id": conversation_id,
                "client_node_id": node_id,
                "parent_node_id": node.parent_id,
            },
            tree_id=tree.tree_id,
        )
        return event_visibility_reason(ctx, event, {tree.tree_id: tree})

    def _tree_for(self, conversation_id: str) -> Optional[TreeBuildResult]:
        if conversation_id not in self._trees:
            index: Dict[str, str] = {}
            if self._registry is not None:
                tree, index = self._registry.snapshot(conversation_id)
            elif self._tree_reader is not None:
                tree = next(iter(load_trees([conversation_id], tree_reader=self._tree_reader).values()), None)
                for node_id, node in (tree.nodes.items() if tree else ()):
                    for event_id in node.event_ids:
                        index[str(event_id)] = node_id
            else:
                tree = None
            self._trees[conversation_id] = tree
            self._event_nodes[conversation_id] = dict(index)
        return self._trees[conversation_id]


def filter_derived(records: Sequence[Mapping[str, Any]], **kwargs: Any) -> Tuple[list, Dict[str, int]]:
    """One-shot helper: ``(visible_records, reasons)``."""
    gate = DerivedGate(**kwargs)
    visible = gate.filter(records)
    return visible, dict(sorted(gate.reasons.items()))


__all__ = [
    "CONVERSATION_UNKNOWN",
    "DerivedGate",
    "filter_derived",
    "record_conversation_id",
    "record_sources",
]
