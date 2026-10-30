"""Inferring regenerations from repeated identical user input in linear history."""

from __future__ import annotations

from branchmem.sources.inferred import drop_superseded, infer_regenerations, superseded_event_ids


def _turn(turn, user_text, ts, *, reply="回复", thought=True):
    events = []
    if user_text is not None:
        events.append({"event_id": f"{turn}-u", "message_id": turn, "role": "user", "content": user_text, "timestamp": ts})
    events.append({"event_id": f"{turn}-a", "message_id": turn, "role": "assistant", "content": reply, "timestamp": ts + 1})
    if thought:
        events.append({"event_id": f"{turn}-t", "message_id": turn, "role": "system", "content": "想", "timestamp": ts + 1})
    return events


def test_consecutive_same_input_keeps_last_reply_only():
    events = _turn("m1", "你好", 0) + _turn("m2", " 你好 ", 30) + _turn("m3", "你好", 60)
    marks = infer_regenerations(events)
    assert {key: value.superseded_by for key, value in marks.items()} == {"m1": "m3", "m2": "m3"}
    dropped = superseded_event_ids(events)
    # the user message reuses one node and is not superseded; only old replies/thoughts go
    assert dropped == {"m1-a", "m1-t", "m2-a", "m2-t"}
    kept = [item["event_id"] for item in drop_superseded(events)]
    assert kept == ["m1-u", "m2-u", "m3-u", "m3-a", "m3-t"]


def test_different_input_or_long_gap_breaks_the_run():
    events = _turn("m1", "你好", 0) + _turn("m2", "在吗", 10) + _turn("m3", "你好", 20)
    assert infer_regenerations(events) == {}
    far = _turn("m1", "晚安", 0) + _turn("m2", "晚安", 86400)
    assert infer_regenerations(far) == {}


def test_proactive_turn_without_user_breaks_the_run():
    events = _turn("m1", "你好", 0) + _turn("p1", None, 5) + _turn("m2", "你好", 10)
    assert infer_regenerations(events) == {}


def test_input_order_does_not_matter():
    events = _turn("m1", "你好", 0) + _turn("m2", "你好", 30)
    assert superseded_event_ids(list(reversed(events))) == superseded_event_ids(events)
