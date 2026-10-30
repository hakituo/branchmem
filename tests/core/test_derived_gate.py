"""Derived-record gate: visible only if every source event is on the current branch."""

from __future__ import annotations

from branchmem.visibility import MODE_STRICT, DerivedGate, filter_derived

CONV = "conv-main"


def _event(node, parent, ts):
    return {
        "event_id": f"evt-{node}",
        "conversation_id": CONV,
        "role": "user",
        "content": node,
        "timestamp": ts,
        "client_node_id": node,
        "parent_node_id": parent,
    }


# root -> a -> b (current), a -> c (abandoned sibling)
EVENTS = [_event("root", "", 1), _event("a", "root", 2), _event("b", "a", 3), _event("c", "a", 4)]


def _gate(**kwargs):
    defaults = {"conversation_id": CONV, "active_path_ids": {"root", "a", "b"}, "tree_reader": lambda _c: EVENTS}
    defaults.update(kwargs)
    return DerivedGate(**defaults)


def _note(note_id, *sources, conversation_id=CONV):
    return {"id": note_id, "conversation_id": conversation_id, "source_event_ids": list(sources)}


def test_shared_prefix_and_current_branch_are_visible():
    gate = _gate()
    assert gate.is_visible(_note("n1", "evt-root", "evt-a"))
    assert gate.is_visible(_note("n2", "evt-a", "evt-b"))


def test_one_sibling_source_hides_the_whole_record():
    gate = _gate()
    assert not gate.is_visible(_note("n3", "evt-a", "evt-c"))
    assert gate.reasons["not_on_anchor_path"] == 1


def test_hard_reason_wins_over_a_compat_pass_reason():
    gate = _gate()
    # one source unknown to the tree (compat would pass it), one on the sibling
    assert not gate.is_visible(_note("n4", "evt-unknown", "evt-c"))


def test_switch_back_restores_visibility_without_touching_records():
    note = _note("n5", "evt-c")
    assert not _gate().is_visible(note)
    assert _gate(active_path_ids={"root", "a", "c"}).is_visible(note)
    assert not _gate().is_visible(note)


def test_source_node_ids_are_accepted():
    gate = _gate()
    assert gate.is_visible({"id": "x", "conversation_id": CONV, "source_node_ids": ["b"]})
    assert not gate.is_visible({"id": "y", "conversation_id": CONV, "source_node_ids": ["c"]})


def test_compat_passes_missing_information_strict_hides_it():
    records = [
        {"id": "no-conv", "source_event_ids": ["evt-a"]},
        {"id": "no-source", "conversation_id": CONV},
        _note("unknown-source", "evt-zzz"),
    ]
    visible, reasons = filter_derived(records, conversation_id=CONV, active_path_ids={"b"},
                                      tree_reader=lambda _c: EVENTS)
    assert len(visible) == 3
    assert reasons == {
        "compat_pass:conversation_unknown": 1,
        "compat_pass:legacy_unknown": 2,
    }
    strict, _ = filter_derived(records, mode=MODE_STRICT, conversation_id=CONV,
                               active_path_ids={"b"}, tree_reader=lambda _c: EVENTS)
    assert strict == []


def test_other_conversation_without_any_anchor_is_not_read():
    calls = []

    def reader(cid):
        calls.append(cid)
        return EVENTS

    gate = _gate(tree_reader=reader)
    assert gate.is_visible(_note("far", "evt-x", conversation_id="conv-other"))
    assert gate.reasons == {"compat_pass:anchor_missing": 1}
    assert calls == []


def test_verdicts_are_cached_per_record_id():
    calls = []
    gate = _gate(tree_reader=lambda cid: calls.append(cid) or EVENTS)
    note = _note("same", "evt-b")
    assert gate.is_visible(note) and gate.is_visible(note)
    assert calls == [CONV]
