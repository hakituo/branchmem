"""Synthetic tree fixtures: branch visibility, deterministic topology, idempotency, gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from branchmem.events import normalize_event
from branchmem.identity import mapping_resolver
from branchmem.tree import build_tree
from branchmem.visibility.context import (
    MODE_COMPAT,
    build_read_context,
    event_visibility_reason,
    note_visibility_reason,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "synthetic_trees.json"


def _load_cases() -> List[Dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def _normalize(case: Dict[str, Any]) -> List[Any]:
    groups = case.get("continuity_groups")
    resolver = mapping_resolver(groups) if groups else None
    return [
        normalize_event(
            {
                **event,
                "conversation_id": event.get("conversation_id") or case["conversation_id"],
                "metadata": {"message_id": event.get("turn_id") or ""},
            },
            storage_scope="role_a",
            perspective_role="role_a",
            continuity_resolver=resolver,
        )
        for event in case["events"]
    ]


def _build(events: List[Any]) -> Dict[str, Any]:
    """One tree per conversation; a case may cover several conversations."""
    grouped: Dict[str, List[Any]] = {}
    for event in events:
        grouped.setdefault(event.conversation_id, []).append(event)
    return {conv: build_tree(items, conversation_id=conv) for conv, items in grouped.items()}


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["id"])
def test_synthetic_case(case):
    events = _normalize(case)
    expect = case["expect"]
    trees_by_conv = _build(events)
    tree = trees_by_conv[events[0].conversation_id]
    by_id = {event.event_id: event for event in events}
    trees = {item.tree_id: item for item in trees_by_conv.values()}

    if "topology_status" in expect:
        assert tree.topology_status == expect["topology_status"]
    if "conversation_count" in expect:
        assert len({event.conversation_id for event in events}) == expect["conversation_count"]
    if "pending_nodes" in expect:
        assert tree.pending == expect["pending_nodes"]
    if "event_count_in_tree" in expect:
        assert sum(len(node.event_ids) for node in tree.nodes.values()) == expect["event_count_in_tree"]
    if expect.get("order_independent"):
        reversed_tree = build_tree(list(reversed(events)), conversation_id=events[0].conversation_id)
        assert reversed_tree.canonical_hash() == tree.canonical_hash()
    for node_id, expected_ids in (expect.get("visible_event_ids") or {}).items():
        assert sorted(tree.visible_event_ids(node_id)) == sorted(expected_ids)
    for event_id, eligible in (expect.get("content_eligible") or {}).items():
        assert by_id[event_id].content_eligible is eligible
    pair = expect.get("continuity_equal")
    if pair:
        assert by_id[pair[0]].continuity_id == by_id[pair[1]].continuity_id

    for anchor, notes in (expect.get("note_visible") or {}).items():
        ctx = build_read_context(
            namespace="role_a",
            perspective_role="role_a",
            continuity_id=events[0].continuity_id,
            anchors_by_tree={tree.tree_id: anchor},
        )
        for note in case.get("notes") or []:
            reason = note_visibility_reason(ctx, note, by_id, trees)
            assert (reason is None) is notes[note["note_id"]], (note["note_id"], reason)

    if "strict_visible" in expect:
        strict = build_read_context(namespace="role_a")
        compat = build_read_context(namespace="role_a", mode=MODE_COMPAT)
        event = events[0]
        assert (event_visibility_reason(strict, event, {}) is None) is expect["strict_visible"]
        assert (event_visibility_reason(compat, event, {}) is None) is expect["compat_visible"]


def test_conversations_do_not_share_continuity_by_default():
    a = normalize_event({"event_id": "x", "conversation_id": "conv-1"})
    b = normalize_event({"event_id": "y", "conversation_id": "conv-2"})
    assert a.continuity_id != b.continuity_id
