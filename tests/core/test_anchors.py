"""Per-tree anchors: request first, recoverable, switching moves the anchor only."""

from __future__ import annotations

import json

import pytest

from branchmem.anchors import (
    AnchorStore,
    bound_anchors,
    get_request_anchors,
    reset_request_anchors,
    resolve_anchors_by_tree,
    set_request_anchors,
)


def test_store_persists_anchor_and_appends_event(tmp_path):
    store = AnchorStore(tmp_path)
    store.set("tree-1", "node-a", reason="switch", changed_at=100.0)
    store.set("tree-1", "node-b", reason="switch", changed_at=200.0)

    assert AnchorStore(tmp_path).get("tree-1") == "node-b"
    lines = [
        json.loads(line)
        for line in (tmp_path / "anchor_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [item["node_id"] for item in lines] == ["node-a", "node-b"]
    assert lines[1]["previous"] == "node-a"
    assert lines[1]["type"] == "anchor_changed"


def test_switch_back_to_old_branch_only_changes_anchor(tmp_path):
    store = AnchorStore(tmp_path)
    store.set("tree-1", "branch-b", changed_at=1.0)
    store.set("tree-1", "branch-a", changed_at=2.0)
    assert store.get("tree-1") == "branch-a"
    events = (tmp_path / "anchor_events.jsonl").read_text(encoding="utf-8").splitlines()
    # both switch events remain: going back moves the anchor, history is kept
    assert len([line for line in events if line.strip()]) == 2


def test_resolve_uses_explicit_before_stored(tmp_path):
    store = AnchorStore(tmp_path)
    store.set("tree-1", "stored-node", changed_at=1.0)
    resolved = resolve_anchors_by_tree(["tree-1", "tree-2"], explicit={"tree-1": "request-node"}, store=store)
    assert resolved["tree-1"] == "request-node"
    assert resolved["tree-2"] is None


def test_missing_anchor_is_none_not_wildcard(tmp_path):
    assert resolve_anchors_by_tree(["tree-x"], store=AnchorStore(tmp_path)) == {"tree-x": None}


def test_request_context_var_returns_copy_and_resets():
    token = set_request_anchors({"tree-1": "node-1"})
    try:
        current = get_request_anchors()
        assert current == {"tree-1": "node-1"}
        current["tree-1"] = "mutated"
        assert get_request_anchors() == {"tree-1": "node-1"}
    finally:
        reset_request_anchors(token)
    assert get_request_anchors() is None


def test_bound_anchors_restores_previous_binding():
    token = set_request_anchors({"tree-1": "outer"})
    try:
        with bound_anchors({"tree-2": "inner"}):
            assert get_request_anchors() == {"tree-2": "inner"}
        assert get_request_anchors() == {"tree-1": "outer"}
    finally:
        reset_request_anchors(token)


def test_empty_anchor_value_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        AnchorStore(tmp_path).set("tree-1", "  ")
