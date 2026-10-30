"""Reference JSONL store, linear-source attachment and topology overlays."""

from __future__ import annotations

import json

from branchmem.cursors import CursorStore
from branchmem.registry import TreeRegistry
from branchmem.sources.jsonl_store import JsonlEventStore
from branchmem.sources.linear import LinearHeadTracker, attach_linear_parent, read_tail_node
from branchmem.sources.overlay import (
    active_leaf_loader,
    apply_overlay,
    clear_overlay_cache,
    overlay_loader,
)
from branchmem.visibility import filter_raw_events


def _msg(node, parent, ts, role="user", conv="conv-main"):
    return {"conversation_id": conv, "role": role, "content": f"text-{node}", "timestamp": ts,
            "client_node_id": node, "parent_node_id": parent}


def test_store_round_trip_dedup_and_listeners(tmp_path):
    store = JsonlEventStore(tmp_path / "events")
    registry = TreeRegistry(store)
    cursors = CursorStore(tmp_path / "cursors", registry=registry)
    store.add_listener(registry.on_append)
    store.add_listener(cursors.advance)

    store.append([_msg("root", "", 1), _msg("a", "root", 2, "assistant")])
    registry.tree("conv-main")
    store.append([_msg("b", "a", 3), _msg("c", "a", 4)])
    assert store.append([_msg("c", "a", 4)]) == []  # re-delivery by idempotency key

    events = store.list_conversation_events("conv-main")
    assert len(events) == 4 and all(e["topology_status"] == "complete" for e in events[1:])
    assert cursors.cursor_for("conv-main") == "c"
    visible = filter_raw_events(events, registry=registry, cursors=cursors.candidates)
    assert sorted(e["client_node_id"] for e in visible.events) == ["a", "c", "root"]
    assert store.conversations() == ["conv-main"]


def test_linear_platform_events_form_one_chain_across_restarts(tmp_path):
    root = tmp_path / "events"
    store = JsonlEventStore(root, linear_platforms={"irc"})
    for index in range(3):
        store.append([{"conversation_id": "room", "role": "user", "content": str(index),
                       "timestamp": index, "platform": "irc"}])
    restarted = JsonlEventStore(root, linear_platforms={"irc"})
    restarted.append([{"conversation_id": "room", "role": "user", "content": "3",
                       "timestamp": 3, "platform": "irc"}])
    tree = TreeRegistry(restarted).tree("room")
    assert len(tree.roots) == 1 and tree.topology_status == "complete"
    leaf = restarted.list_conversation_events("room")[-1]["client_node_id"]
    assert len(tree.ancestors(leaf)) == 4


def test_linear_attach_on_empty_file_is_partial_not_a_fake_parent(tmp_path):
    path = tmp_path / "x.jsonl"
    payload = {"event_id": "e1"}
    assert attach_linear_parent(payload, file_path=path, platform="linear", tracker=LinearHeadTracker()) == ""
    assert payload["client_node_id"] == "srv:e1" and payload["topology_status"] == "partial"
    path.write_text(json.dumps({"event_id": "e9"}) + "\n", encoding="utf-8")
    assert read_tail_node(path) == "srv:e9"


def test_non_linear_platform_is_left_alone(tmp_path):
    payload = {"event_id": "e1"}
    attach_linear_parent(payload, file_path=tmp_path / "x", platform="app", tracker=LinearHeadTracker())
    assert "client_node_id" not in payload


def _write_overlay(directory, conversation_id="conv-old"):
    directory.mkdir(parents=True, exist_ok=True)
    overlay = {
        "version": 1,
        "conversation_id": conversation_id,
        "nodes": {"n1": ["", 1.0, True], "n2": ["n1", 2.0, False], "n3": ["n2", 3.0, True],
                  "n3b": ["n2", 3.5, True]},
        "events": {"old-1": "n1", "old-3": "n3", "old-3b": "n3b"},
        "active_leaf": "n3",
    }
    (directory / f"{conversation_id}.json").write_text(json.dumps(overlay), encoding="utf-8")


def test_overlay_adds_topology_without_rewriting_events(tmp_path):
    clear_overlay_cache()
    _write_overlay(tmp_path / "overlays")
    history = [
        {"event_id": "old-1", "conversation_id": "conv-old", "role": "user", "content": "q", "timestamp": 1},
        {"event_id": "old-3", "conversation_id": "conv-old", "role": "user", "content": "kept", "timestamp": 3},
        {"event_id": "old-3b", "conversation_id": "conv-old", "role": "user", "content": "gone", "timestamp": 3.5},
    ]
    applied = apply_overlay(history, json.loads((tmp_path / "overlays" / "conv-old.json").read_text()),
                            conversation_id="conv-old")
    structural = [item for item in applied if item.get("event_type") == "structural"]
    assert [item["client_node_id"] for item in structural] == ["n2"]
    assert "client_node_id" not in history[0]  # originals untouched

    registry = TreeRegistry(lambda _cid: history, overlay_loader=overlay_loader(tmp_path / "overlays"))
    cursors = CursorStore(tmp_path / "cursors", registry=registry,
                          overlay_leaf=active_leaf_loader(tmp_path / "overlays"))
    outcome = filter_raw_events(history, registry=registry, cursors=cursors.candidates)
    assert [e["event_id"] for e in outcome.events] == ["old-1", "old-3"]
    assert outcome.reasons == {"not_on_anchor_path": 1}
