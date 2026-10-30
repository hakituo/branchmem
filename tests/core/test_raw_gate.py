"""Raw-event gate: compat reason codes, sibling hiding, consistent A -> B -> A switches."""

from __future__ import annotations

from branchmem.anchors import reset_request_anchors, set_request_anchors
from branchmem.events import normalize_event
from branchmem.tree import build_tree
from branchmem.visibility import MODE_COMPAT, MODE_STRICT, filter_raw_events

CONV = "conv-main"


def _payload(node: str, parent: str = "", *, ts: float = 1.0) -> dict:
    payload = {
        "event_id": f"evt-{node}",
        "conversation_id": CONV,
        "role": "user",
        "content": f"text-{node}",
        "timestamp": ts,
        "event_type": "chat_message",
    }
    if node:
        payload["client_node_id"] = node
        payload["parent_node_id"] = parent
    return payload


def _fixture():
    return [
        _payload("root", ts=1.0),
        _payload("a", "root", ts=2.0),
        _payload("b", "a", ts=3.0),
        _payload("c", "a", ts=4.0),
        _payload("", ts=0.5),  # legacy: no client node ID
    ]


def _ids(events):
    return sorted(str(item.get("event_id") or "") for item in events)


def _run(events, *, mode=MODE_COMPAT, path=()):
    return filter_raw_events(
        events,
        conversation_id=CONV,
        mode=mode,
        active_path_ids=path,
        tree_reader=lambda _cid: events,
    )


def test_compat_passes_missing_topology_with_reason():
    events = _fixture()
    outcome = _run(events)
    assert outcome.hidden == 0
    assert outcome.visible == len(events)
    assert outcome.reasons["compat_pass:legacy_unknown"] == 1
    assert outcome.reasons["compat_pass:anchor_missing"] == 4


def test_active_path_hides_sibling():
    outcome = _run(_fixture(), path={"root", "a", "b"})
    assert _ids(outcome.events) == ["evt-", "evt-a", "evt-b", "evt-root"]
    assert outcome.reasons["not_on_anchor_path"] == 1
    assert outcome.reasons["compat_pass:legacy_unknown"] == 1


def test_strict_mode_rejects_legacy_and_sibling():
    outcome = _run(_fixture(), mode=MODE_STRICT, path={"root", "a", "b"})
    assert _ids(outcome.events) == ["evt-a", "evt-b", "evt-root"]
    assert outcome.reasons == {"legacy_unknown": 1, "not_on_anchor_path": 1}


def test_explicit_request_anchor_wins():
    events = _fixture()
    tree = build_tree([normalize_event(item) for item in events], conversation_id=CONV)
    token = set_request_anchors({tree.tree_id: "c"})
    try:
        outcome = _run(events, path={"root", "a", "b"})
    finally:
        reset_request_anchors(token)
    assert "evt-b" not in _ids(outcome.events)
    assert "evt-c" in _ids(outcome.events)


def test_switch_a_b_a_keeps_same_visible_set():
    events = _fixture()
    first_a = _ids(_run(events, path={"root", "a", "b"}).events)
    view_b = _ids(_run(events, path={"root", "a", "c"}).events)
    again_a = _ids(_run(events, path={"root", "a", "b"}).events)
    assert first_a == again_a
    assert "evt-c" in view_b and "evt-b" not in view_b


def test_pending_node_passes_in_compat_but_not_strict():
    events = _fixture() + [_payload("orphan", "missing-parent", ts=5.0)]
    compat = _run(events, path={"root", "a", "b"})
    strict = _run(events, mode=MODE_STRICT, path={"root", "a", "b"})
    assert "evt-orphan" in _ids(compat.events)
    assert compat.reasons["compat_pass:topology_incomplete"] == 1
    assert "evt-orphan" not in _ids(strict.events)
    assert strict.reasons["topology_incomplete"] == 1


def test_tree_reader_failure_falls_back_with_reason():
    events = _fixture()

    def _boom(_cid):
        raise RuntimeError("tree unavailable")

    outcome = filter_raw_events(events, conversation_id=CONV, tree_reader=_boom)
    assert outcome.visible == len(events)
    assert outcome.reasons == {"compat_pass:gate_error": len(events)}
    strict = filter_raw_events(events, conversation_id=CONV, tree_reader=_boom, mode=MODE_STRICT)
    assert strict.visible == 0 and strict.reasons == {"gate_error": len(events)}


def test_without_reader_trees_come_from_the_events_themselves():
    events = _fixture()
    outcome = filter_raw_events(events, conversation_id=CONV, active_path_ids={"root", "a", "c"})
    assert "evt-b" not in _ids(outcome.events) and "evt-c" in _ids(outcome.events)
