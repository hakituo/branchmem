"""Segmenter v2: exchanges as units, idle hard cuts, recursive topic-depth splits."""

from __future__ import annotations

import random

from branchmem.events import normalize_event
from branchmem.segments.topic_scoring import EmbeddingScorer
from branchmem.segments.v1 import IDLE, PATH_END
from branchmem.segments.topic import (
    SEGMENTER_V2,
    TOPIC,
    TopicParams,
    build_topic_segments,
    build_units,
)

CONV = "conv-main"


def _event(index, ts, role, content, *, turn=""):
    return normalize_event(
        {
            "event_id": f"e{index:04d}",
            "conversation_id": CONV,
            "role": role,
            "content": content,
            "timestamp": ts,
            "event_type": "chat_message" if role == "user" else "chat_reply",
            "turn_id": turn or f"m{index}",  # multi-bubble: each message has its own ID
        }
    )


def _dialog(spec, *, start=1000.0, step=30.0):
    """spec: [(role, text), ...], 30 seconds apart."""
    return [_event(i, start + i * step, role, text) for i, (role, text) in enumerate(spec)]


def _topic_scorer(units, index):
    """Test scorer: a different first character means a topic change."""
    return 1.0 if units[index - 1][0].content[0] != units[index][0].content[0] else 0.0


def test_reply_bubbles_stay_with_their_question():
    events = _dialog(
        [("user", "问"), ("assistant", "答一"), ("assistant", "答二"), ("user", "再问")]
    )
    units = build_units(events, TopicParams())
    assert [[e.event_id for e in unit] for unit in units] == [
        ["e0000", "e0001", "e0002"],
        ["e0003"],
    ]


def test_idle_gap_is_the_only_hard_boundary_for_short_sessions():
    first = _dialog([("user", "甲" * 50), ("assistant", "乙" * 50)] * 20)
    later = _dialog([("user", "丙"), ("assistant", "丁")], start=first[-1].occurred_at + 3600)
    result = build_topic_segments(first + later, tree_id=CONV, scorer=_topic_scorer)
    assert [len(seg.turn_ids) for seg in result.segments] == [20, 1]
    assert [seg.boundary_reason for seg in result.segments] == [IDLE, PATH_END]
    assert result.tail_segments == 1
    assert all(seg.segmenter_version == SEGMENTER_V2 for seg in result.segments)


def test_long_session_is_cut_at_topic_shift_with_min_part_size():
    spec = [("user", "猫" * 100), ("assistant", "猫" * 200)] * 8
    spec += [("user", "车" * 100), ("assistant", "车" * 200)] * 8
    result = build_topic_segments(_dialog(spec), tree_id=CONV, scorer=_topic_scorer)
    assert [seg.char_count for seg in result.segments] == [2400, 2400]
    assert result.segments[0].boundary_reason == TOPIC
    # the segment after a cut carries the previous tail as context, not as sources
    second = result.segments[1]
    assert second.context_event_ids
    assert not set(second.context_event_ids) & set(second.source_event_ids)
    assert 0 < len(second.context_event_ids) <= 4  # at most the last 2 exchanges


def test_cut_never_leaves_a_side_below_min_part_chars():
    spec = [("user", "车" * 100), ("assistant", "车" * 50)]  # an isolated 150-character topic
    spec += [("user", "猫" * 200), ("assistant", "猫" * 300)] * 9
    params = TopicParams(session_max_chars=4000, min_part_chars=300)
    result = build_topic_segments(_dialog(spec), tree_id=CONV, params=params, scorer=_topic_scorer)
    assert all(seg.char_count >= 300 for seg in result.segments)
    assert all(seg.char_count <= 4000 for seg in result.segments)


def test_segment_ids_are_stable_under_shuffled_input():
    spec = [
        item
        for i in range(12)
        for item in (("user", f"话题{i % 3}" * 40), ("assistant", "回" * 300))
    ]
    events = _dialog(spec)
    shuffled = list(events)
    random.Random(7).shuffle(shuffled)
    a = build_topic_segments(events, tree_id=CONV, scorer=_topic_scorer)
    b = build_topic_segments(shuffled, tree_id=CONV, scorer=_topic_scorer)
    assert [s.segment_id for s in a.segments] == [s.segment_id for s in b.segments]
    assert a.sessions == b.sessions


def test_cached_cuts_are_replayed_without_scoring():
    spec = [("user", "猫" * 100), ("assistant", "猫" * 200)] * 8
    spec += [("user", "车" * 100), ("assistant", "车" * 200)] * 8
    events = _dialog(spec)
    first = build_topic_segments(events, tree_id=CONV, scorer=_topic_scorer)
    digest, cuts, _tail = first.sessions[0]

    def boom(units, index):
        raise AssertionError("a cache hit must not score")

    replay = build_topic_segments(
        events, tree_id=CONV, scorer=boom, lookup=lambda d, _s: cuts if d == digest else None
    )
    assert [s.segment_id for s in replay.segments] == [s.segment_id for s in first.segments]


def test_oversized_single_exchange_is_split_on_message_boundaries():
    spec = [("user", "问")] + [("assistant", "长" * 3000)] * 4
    result = build_topic_segments(_dialog(spec), tree_id=CONV, scorer=_topic_scorer)
    assert len(result.segments) >= 3
    assert all(seg.char_count <= 6001 for seg in result.segments)


def test_embedding_scorer_prefers_dissimilar_windows():
    def fake_embed(texts):
        return [[1.0, 0.0] if text.startswith("猫") else [0.0, 1.0] for text in texts]

    units = build_units(
        _dialog([("user", "猫"), ("user", "猫"), ("user", "车")] * 1), TopicParams()
    )
    units = [[e] for unit in units for e in unit]
    scorer = EmbeddingScorer(fake_embed)
    scorer.prepare(units)
    assert scorer(units, 2) > scorer(units, 1)
    assert abs(scorer(units, 2) - 1.0) < 1e-9
    assert scorer.encoded == 3
