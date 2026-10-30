"""Deterministic segmentation (v1).

- cuts only at complete turn boundaries; idle / max_turns / max_chars closures;
- no hard cut at a fixed hour; a parent in the previous segment does not close one;
- shuffled input keeps segment IDs; structural (media placeholder) turns stay out;
- only one root-to-leaf path is segmented, siblings never mix in;
- a late event invalidates only the affected segment (``diff_segments``).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from branchmem.events import normalize_event
from branchmem.segments.v1 import IDLE, MAX_TURNS, build_segments, diff_segments
from branchmem.tree import build_tree

CONV = "conv-main"
SHANGHAI = timezone(timedelta(hours=8))


def _event(node, parent, ts, *, role="user", turn="", content=None):
    return normalize_event(
        {
            "event_id": f"evt-{node}",
            "conversation_id": CONV,
            "role": role,
            "content": f"text-{node}" if content is None else content,
            "timestamp": ts,
            "event_type": "chat_message",
            "client_node_id": node,
            "parent_node_id": parent,
            "turn_id": turn or f"t-{node}",
        }
    )


def _exchange(index, ts, parent="", prefix=""):
    """One complete exchange: user + assistant sharing a turn_id."""
    key = f"{prefix}{index}"
    user = _event(f"u{key}", parent, ts, turn=f"t{key}")
    bot = _event(f"a{key}", f"u{key}", ts + 1, role="assistant", turn=f"t{key}")
    return [user, bot]


def _chain(count, *, start=1000.0, step=60.0, prefix="", parent=""):
    events = []
    for index in range(count):
        pair = _exchange(index, start + index * step, parent, prefix)
        events.extend(pair)
        parent = pair[-1].client_node_id
    return events


def test_idle_gap_splits_on_turn_boundary():
    events = _chain(2) + _exchange(9, 1000.0 + 3 * 3600, parent="a1")
    segments = build_segments(events, tree_id="t", idle_seconds=1800)
    assert [len(item.turn_ids) for item in segments] == [2, 1]
    assert segments[0].boundary_reason == IDLE
    # a segment always closes after a complete turn: user/assistant are never split
    assert segments[0].source_event_ids[-1] == "evt-a1"


def test_max_turns_seals_without_sealing_every_following_turn():
    segments = build_segments(_chain(7), tree_id="t", max_turns=3)
    assert [len(item.turn_ids) for item in segments] == [3, 3, 1]
    assert [item.boundary_reason for item in segments[:2]] == [MAX_TURNS, MAX_TURNS]


def test_no_hard_break_at_four_am():
    night = datetime(2026, 10, 25, 3, 50, tzinfo=SHANGHAI).timestamp()
    events = []
    parent = ""
    for index in range(4):  # 03:50 -> 04:05, five minutes apart
        pair = _exchange(index, night + index * 300, parent)
        events.extend(pair)
        parent = pair[-1].client_node_id
    segments = build_segments(events, tree_id="t", idle_seconds=1800)
    assert len(segments) == 1


def test_input_order_does_not_change_segment_ids():
    events = _chain(6)
    shuffled = list(events)
    random.Random(7).shuffle(shuffled)
    first = [item.segment_id for item in build_segments(events, tree_id="t", max_turns=4)]
    second = [item.segment_id for item in build_segments(shuffled, tree_id="t", max_turns=4)]
    assert first == second


def test_media_placeholder_turn_is_kept_out_of_extraction():
    events = _chain(1) + [
        _event("img", "a0", 1100.0, turn="t-img", content="[图片]"),
    ]
    segments = build_segments(events, tree_id="t")
    assert all("evt-img" not in item.source_event_ids for item in segments)
    with_structure = build_segments(events, tree_id="t", include_ineligible=True)
    assert any("evt-img" in item.source_event_ids for item in with_structure)


def test_segments_follow_single_path_only():
    shared = _chain(2)
    left = _exchange(5, 2000.0, parent="a1")
    right = _exchange(6, 2100.0, parent="a1")
    tree = build_tree(shared + left + right, conversation_id=CONV)
    by_id = {event.event_id: event for event in shared + left + right}
    path_events = [by_id[item] for item in tree.path_event_ids("a5")]
    segments = build_segments(path_events, tree_id=tree.tree_id)
    sources = {item for segment in segments for item in segment.source_event_ids}
    assert "evt-u5" in sources and "evt-u6" not in sources


def test_late_event_only_invalidates_affected_segment():
    early = _chain(2, start=1000.0)
    late_block = _chain(2, start=1000.0 + 5 * 3600, prefix="L", parent="a1")
    before = build_segments(early + late_block, tree_id="t", idle_seconds=1800)
    straggler = _event("late", "a1", 1000.0 + 5 * 3600 + 200.0, turn="t-late")
    after = build_segments(early + late_block + [straggler], tree_id="t", idle_seconds=1800)
    diff = diff_segments(before, after)
    assert diff.kept == [before[0].segment_id]
    assert diff.invalidated == [before[1].segment_id]
    assert len(diff.added) == 1
    assert not diff_segments(after, after).changed


def test_shared_prefix_segments_are_reused_across_siblings():
    shared = _chain(4, start=1000.0)
    left = _chain(2, start=1000.0 + 4 * 3600, prefix="L", parent="a3")
    right = _chain(2, start=1000.0 + 5 * 3600, prefix="R", parent="a3")
    tree = build_tree(shared + left + right, conversation_id=CONV)
    by_id = {event.event_id: event for event in shared + left + right}

    def path_segments(leaf):
        events = [by_id[item] for item in tree.path_event_ids(leaf)]
        return build_segments(events, tree_id=tree.tree_id, idle_seconds=1800, max_turns=3)

    left_ids = [item.segment_id for item in path_segments("aL1")]
    right_ids = [item.segment_id for item in path_segments("aR1")]
    # the 4-turn shared prefix splits into [3, 1]; both siblings reuse those two segment IDs
    assert left_ids[:2] == right_ids[:2]
    assert left_ids[2:] != right_ids[2:]
