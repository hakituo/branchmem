"""Per-conversation cursors, branch-switch labels and the orphan-chain guard."""

from __future__ import annotations

import random

import pytest

from branchmem.cursors import REASON_GRAFT, REASON_SWITCH, REASON_TURN, CursorStore
from branchmem.events import enrich_payload, normalize_event
from branchmem.registry import TreeRegistry
from branchmem.tree import build_tree
from branchmem.visibility import DerivedGate, filter_raw_events

CONV = "conv-main"


def _event(node, parent, ts, role="user", event_id=""):
    return {
        "event_id": event_id or f"evt-{node}",
        "conversation_id": CONV,
        "role": role,
        "content": f"text-{node}",
        "timestamp": ts,
        "event_type": "chat_message",
        "client_node_id": node,
        "parent_node_id": parent,
    }


# root -> a -> {b, c}
EVENTS = [_event("root", "", 1.0), _event("a", "root", 2.0), _event("b", "a", 3.0), _event("c", "a", 4.0)]
# main line root -> a -> b (b holds the cursor)
MAIN = [_event("root", "", 1.0), _event("a", "root", 2.0, "assistant"), _event("b", "a", 3.0)]
# a server message stored without a node; the client gave it a local ID the server never saw
SERVER_LEGACY = {
    "event_id": "evt-proactive",
    "conversation_id": CONV,
    "role": "assistant",
    "content": "proactive",
    "timestamp": 4.0,
    "event_type": "proactive_message",
}
REPLY = _event("bg-3", "client-local-proactive", 5.0)
REPLY_ANSWER = _event("bg-4", "bg-3", 5.1, "assistant")


class World:
    """A store-like list of rows, a registry over it and cursors wired together."""

    def __init__(self, tmp_path, rows, overlay_leaf=None):
        self.rows = list(rows)
        self.registry = TreeRegistry(lambda _cid: list(self.rows))
        self.cursors = CursorStore(tmp_path / "cursors", registry=self.registry, overlay_leaf=overlay_leaf)

    def write(self, payloads):
        self.rows.extend(payloads)
        self.registry.on_append(payloads)
        return self.cursors.advance(payloads)

    def gate(self, events, conversation_id="", **kwargs):
        return filter_raw_events(
            events, conversation_id=conversation_id, registry=self.registry,
            cursors=self.cursors.candidates, **kwargs,
        )


def _ids(events):
    return sorted(item["event_id"] for item in events)


@pytest.fixture
def main_world(tmp_path):
    world = World(tmp_path, MAIN)
    world.registry.tree(CONV)
    world.cursors.advance([MAIN[-1]])
    assert world.cursors.cursor_for(CONV) == "b"
    return world


def test_written_turn_advances_cursor_and_switch_is_labelled(tmp_path):
    world = World(tmp_path, EVENTS[:2])
    world.registry.tree(CONV)
    first = world.write([EVENTS[2]])
    assert first[CONV]["node_id"] == "b" and first[CONV]["reason"] == REASON_TURN
    assert world.cursors.advance([EVENTS[2]]) == {}  # unchanged position: nothing written
    switched = world.write([EVENTS[3]])
    assert switched[CONV]["reason"] == REASON_SWITCH
    assert world.cursors.cursor_for(CONV) == "c"


def test_read_gate_without_request_path_uses_cursor(tmp_path):
    world = World(tmp_path, EVENTS)
    assert len(world.gate(EVENTS).events) == 4  # no anchor at all: compat lets it through
    world.cursors.advance([EVENTS[2]])  # now on the b line
    assert _ids(world.gate(EVENTS).events) == ["evt-a", "evt-b", "evt-root"]


def test_request_path_still_wins_over_cursor(tmp_path):
    world = World(tmp_path, EVENTS)
    world.cursors.advance([EVENTS[2]])
    outcome = world.gate(EVENTS, conversation_id=CONV, active_path_ids={"root", "a", "c"})
    assert _ids(outcome.events) == ["evt-a", "evt-c", "evt-root"]


def test_derived_gate_uses_cursor_without_request(tmp_path):
    world = World(tmp_path, EVENTS)
    world.cursors.advance([EVENTS[3]])  # now on the c line
    gate = DerivedGate(registry=world.registry, cursors=world.cursors.candidates)

    def record(record_id, event_id):
        return {"id": record_id, "conversation_id": CONV, "source_event_ids": [event_id]}

    assert gate.is_visible(record("m-c", "evt-c"))
    assert not gate.is_visible(record("m-b", "evt-b"))


def test_overlay_active_leaf_seeds_missing_cursor(tmp_path):
    world = World(tmp_path, EVENTS, overlay_leaf=lambda _cid: "b")
    assert world.cursors.cursor_for(CONV) == "b"
    world.cursors.advance([EVENTS[3]])
    assert world.cursors.cursor_for(CONV) == "c"  # a real cursor replaces the exported leaf


def test_unknown_parent_reply_is_grafted_not_switched(main_world):
    written = [SERVER_LEGACY, REPLY, REPLY_ANSWER]
    records = main_world.write(written)
    assert records[CONV]["reason"] == REASON_GRAFT  # not branch_switch
    assert main_world.cursors.grafts_for(CONV) == {"bg-3": "b"}
    tree = main_world.registry.tree(CONV)
    assert tree.roots == ["root"] and tree.pending == []
    assert tree.ancestors("bg-4") == ["root", "a", "b", "bg-3", "bg-4"]
    outcome = main_world.gate(main_world.rows, conversation_id=CONV)
    assert outcome.hidden == 0, outcome.reasons  # main line and new reply both on the cursor path


def test_later_turns_on_grafted_chain_keep_advancing(main_world):
    main_world.write([REPLY, REPLY_ANSWER])
    nxt = _event("bg-5", "bg-4", 6.0)
    assert main_world.write([nxt])[CONV]["reason"] == REASON_TURN
    assert main_world.cursors.cursor_for(CONV) == "bg-5"


def test_real_parent_arriving_later_wins_over_graft(main_world):
    main_world.write([REPLY])
    main_world.write([_event("client-local-proactive", "a", 4.5, "assistant")])
    tree = main_world.registry.tree(CONV)
    assert tree.nodes["bg-3"].parent_id == "client-local-proactive"
    assert "bg-3" not in tree.inferred_parents


def test_orphan_without_grounded_previous_holds_cursor(tmp_path):
    world = World(tmp_path, MAIN + [REPLY])
    world.registry.tree(CONV)
    world.cursors.store.set(CONV, "ghost-node", reason="test")  # old cursor is not in the tree
    world.cursors.clear_cache()
    assert world.cursors.advance([REPLY]) == {}
    assert world.cursors.cursor_for(CONV) == "ghost-node"
    assert '"anchor_held"' in world.cursors.store.events_path.read_text(encoding="utf-8")


def test_read_side_ignores_cursor_on_orphan_chain(tmp_path):
    world = World(tmp_path, MAIN + [REPLY, REPLY_ANSWER])
    world.cursors.store.set(CONV, "bg-4")  # a bad cursor written by an older version
    world.cursors.clear_cache()
    outcome = world.gate(world.rows)
    assert outcome.reasons.get("not_on_anchor_path", 0) == 0, outcome.reasons
    assert outcome.hidden == 0


def test_server_message_fields_follow_cursor(main_world):
    assert main_world.cursors.server_message_fields(CONV, "") == {}
    assert main_world.cursors.server_message_fields(CONV, "ws-1") == {
        "client_node_id": "ws-1",
        "parent_node_id": "b",
    }
    assert main_world.cursors.server_message_fields("conv-other", "ws-1") == {}  # no new root


def test_server_message_then_reply_on_last_bubble_is_plain_turn(main_world):
    proactive = dict(SERVER_LEGACY, **main_world.cursors.server_message_fields(CONV, "ws-1"))
    proactive.update(enrich_payload(proactive))
    reply = _event("bg-3", "ws-1-2", 5.0)  # the client split it into three bubbles
    assert main_world.write([proactive])[CONV]["reason"] == REASON_TURN
    assert main_world.write([reply])[CONV]["reason"] == REASON_TURN
    tree = main_world.registry.tree(CONV)
    assert tree.ancestors("bg-3") == ["root", "a", "b", "ws-1", "bg-3"]
    assert tree.inferred_parents == {"bg-3": "segment_alias"}
    assert main_world.cursors.grafts_for(CONV) == {}


def test_segment_alias_is_order_independent():
    proactive = dict(SERVER_LEGACY, client_node_id="ws-1", parent_node_id="b")
    events = [normalize_event(item) for item in MAIN + [proactive, _event("bg-3", "ws-1-2", 5.0)]]
    hashes = set()
    for seed in range(5):
        random.Random(seed).shuffle(events)
        tree = build_tree(events, conversation_id=CONV)
        hashes.add(tree.canonical_hash())
        assert tree.nodes["bg-3"].parent_id == "ws-1"
    assert len(hashes) == 1


def test_segment_alias_only_for_assistant_nodes():
    events = [normalize_event(item) for item in MAIN + [_event("x", "b-1", 5.0)]]
    tree = build_tree(events, conversation_id=CONV)  # b is a user node, not a reply bubble
    assert tree.pending == ["x"] and not tree.is_grounded("x")


def test_multi_root_reason_is_reported():
    events = [normalize_event(item) for item in MAIN + [_event("other", "", 9.0)]]
    tree = build_tree(events, conversation_id=CONV)
    assert any("roots" in reason for reason in tree.reasons)
