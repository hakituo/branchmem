"""Deterministic tree: order independence, child-before-parent, variants share only the parent path."""

from __future__ import annotations

from branchmem.events import normalize_event
from branchmem.tree import TreeBuilder, build_tree


def _ev(
    node_id,
    parent="",
    *,
    variant_of="",
    event_id="",
    role="user",
    content="text",
    ts=1.0,
    conversation_id="cid",
):
    return normalize_event(
        {
            "event_id": event_id or f"evt-{node_id}",
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": ts,
            "event_type": "chat_message",
            "client_node_id": node_id,
            "parent_node_id": parent,
            "variant_of_node_id": variant_of,
            "metadata": {"message_id": f"turn-{node_id}"},
        }
    )


def _tree_nodes():
    """root -> a -> b (main line), a -> c (sibling branch)."""
    return [
        _ev("root", event_id="e-root", ts=1.0),
        _ev("a", "root", event_id="e-a", ts=2.0),
        _ev("b", "a", event_id="e-b", ts=3.0),
        _ev("c", "a", event_id="e-c", ts=4.0),
    ]


def test_build_is_order_independent():
    events = _tree_nodes()
    forward = build_tree(events, conversation_id="cid")
    backward = build_tree(list(reversed(events)), conversation_id="cid")
    shuffled = build_tree([events[2], events[0], events[3], events[1]], conversation_id="cid")

    assert forward.canonical_hash() == backward.canonical_hash() == shuffled.canonical_hash()
    assert forward.tree_id == backward.tree_id
    assert forward.topology_status == "complete"


def test_child_before_parent_is_pending_then_attached():
    events = _tree_nodes()
    builder = TreeBuilder(conversation_id="cid")
    for event in [events[2], events[3], events[1], events[0]]:
        builder.add(event)
    result = builder.build()

    assert result.pending == []
    assert result.nodes["b"].parent_id == "a"
    assert result.nodes["c"].parent_id == "a"
    assert result.ancestors("b") == ["root", "a", "b"]
    assert result.canonical_hash() == build_tree(events).canonical_hash()


def test_missing_parent_keeps_topology_partial():
    builder = TreeBuilder(conversation_id="cid")
    builder.add(_ev("b", "a", event_id="e-b"))
    result = builder.build()

    assert result.topology_status == "partial"
    assert result.pending == ["b"]
    assert any("parent a" in reason for reason in result.reasons)


def test_variant_inherits_old_sibling_parent_but_not_ancestry():
    events = [
        _ev("root", event_id="e-root", ts=1.0),
        _ev("a", "root", event_id="e-a", ts=2.0),
        _ev("b", "a", event_id="e-b", ts=3.0),
        _ev("b2", variant_of="b", event_id="e-b2", ts=4.0),
    ]
    result = build_tree(events)

    assert result.nodes["b2"].parent_id == "a"
    assert result.ancestors("b2") == ["root", "a", "b2"]
    assert "b" not in result.ancestors("b2")
    assert result.topology_status == "complete"


def test_variant_parent_conflict_is_reported():
    events = [
        _ev("root", event_id="e-root", ts=1.0),
        _ev("a", "root", event_id="e-a", ts=2.0),
        _ev("b", "a", event_id="e-b", ts=3.0),
        _ev("b2", "root", variant_of="b", event_id="e-b2", ts=4.0),
    ]
    result = build_tree(events)

    assert result.conflicts
    assert result.topology_status == "partial"


def test_sibling_is_not_visible_from_other_branch():
    tree = build_tree(_tree_nodes())
    anchor_b = tree.nodes["b"].node_id

    assert tree.is_visible("root", anchor_b) is True
    assert tree.is_visible("a", anchor_b) is True
    assert tree.is_visible("b", anchor_b) is True
    assert tree.is_visible("c", anchor_b) is False
    assert tree.visible_event_ids(anchor_b) == {"e-root", "e-a", "e-b"}


def test_unknown_anchor_never_defaults_to_whole_tree():
    tree = build_tree(_tree_nodes())

    assert tree.is_visible("b", "") is False
    assert tree.visible_event_ids("no-such-node") == set()


def test_events_without_client_node_id_are_unknown_not_new_roots():
    event = normalize_event(
        {
            "event_id": "legacy-1",
            "conversation_id": "cid",
            "role": "user",
            "content": "legacy",
            "timestamp": 1.0,
        }
    )
    result = build_tree([event])

    assert result.nodes == {}
    assert result.unknown_event_ids == ["legacy-1"]
    assert result.topology_status == "unknown"


def test_canonical_hash_differs_for_different_shapes():
    left = build_tree(_tree_nodes())
    right = build_tree(
        [
            _ev("root", event_id="e-root", ts=1.0),
            _ev("a", "root", event_id="e-a", ts=2.0),
            _ev("b", "root", event_id="e-b", ts=3.0),
        ]
    )

    assert left.canonical_hash() != right.canonical_hash()


def test_duplicate_event_id_is_not_counted_twice():
    event = _ev("a", "root", event_id="e-a")
    builder = TreeBuilder(conversation_id="cid")
    builder.add(_ev("root", event_id="e-root"))
    assert builder.add(event) == "added"
    assert builder.add(event) == "duplicate"
    result = builder.build()

    assert result.nodes["a"].event_ids == ["e-a"]


def test_variant_with_parent_attaches_even_if_old_version_never_arrives():
    """An edited user message whose old version never reached the server."""
    events = _tree_nodes() + [
        _ev("b2", "a", variant_of="never-sent", event_id="e-b2", ts=5.0),
        _ev("b2-reply", "b2", event_id="e-b2r", ts=6.0),
    ]
    tree = build_tree(events, conversation_id="cid")
    assert not tree.pending
    assert tree.ancestors("b2-reply") == ["root", "a", "b2", "b2-reply"]
    assert tree.nodes["b2"].variant_of == "never-sent"
    # the old version arriving later does not change the structure
    late = build_tree(events + [_ev("never-sent", "a", event_id="e-ns", ts=4.5)], conversation_id="cid")
    assert late.ancestors("b2-reply") == ["root", "a", "b2", "b2-reply"]


def test_variant_without_parent_still_waits_for_old_version():
    builder = TreeBuilder(conversation_id="cid")
    for event in _tree_nodes():
        builder.add(event)
    assert builder.add(_ev("b2", variant_of="b9", event_id="e-b2", ts=5.0)) == "pending"
