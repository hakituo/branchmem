"""Extraction pipeline: branch-only inputs, ledger states, budgets, relay order."""

from __future__ import annotations

import asyncio

from branchmem.extraction import Budget, JobLedger, SegmentParams, build_conversation_inputs, run_extraction
from branchmem.extraction.llm_extractor import LLMExtractor, build_messages, group_exchanges
from branchmem.extraction.relay import carry_start, ordered_chains
from branchmem.extraction.testing import FakeExtractor
from branchmem.notes import NoteStore

CONV = "conv-main"
HOUR = 3600.0


def _ev(node, parent, role, text, ts, turn):
    return {"event_id": f"e-{node}", "conversation_id": CONV, "role": role, "content": text,
            "timestamp": ts, "client_node_id": node, "parent_node_id": parent, "turn_id": turn,
            "event_type": "chat_message"}


def _history():
    """Two idle-separated sessions; the second one has an abandoned sibling reply."""
    return [
        _ev("u1", "", "user", "I adopted a cat", 0, "t1"),
        _ev("a1", "u1", "assistant", "Congratulations!", 1, "t1"),
        _ev("u2", "a1", "user", "her name is Mochi", 2 * HOUR, "t2"),
        _ev("a2", "u2", "assistant", "Mochi is a lovely name", 2 * HOUR + 1, "t2"),
        _ev("a2x", "u2", "assistant", "SIBLING reply that was regenerated away", 2 * HOUR + 2, "t2x"),
    ]


def _inputs(events=None, now=10 * HOUR, path=("u1", "a1", "u2", "a2")):
    return build_conversation_inputs(
        events or _history(), conversation_id=CONV, now=now,
        tree_reader=lambda _c: events or _history(),
        cursors=lambda _cid: [path[-1]],
    )


def test_inputs_follow_current_branch_only():
    items = _inputs()
    assert [i.closed for i in items] == [True, True]
    sources = {e for item in items for e in item.segment.source_event_ids}
    assert "e-a2" in sources and "e-a2x" not in sources


def test_tail_stays_open_until_idle():
    items = _inputs(now=2 * HOUR + 10)
    assert [i.closed for i in items] == [True, False]


def test_v2_segmenter_gives_same_ids_on_replay():
    params = SegmentParams(segmenter="v2")
    first = build_conversation_inputs(_history(), conversation_id=CONV, params=params, now=10 * HOUR,
                                      tree_reader=lambda _c: _history(), cursors=lambda _c: ["a2"])
    again = build_conversation_inputs(list(reversed(_history())), conversation_id=CONV, params=params,
                                      now=10 * HOUR, tree_reader=lambda _c: _history(), cursors=lambda _c: ["a2"])
    assert [i.segment_id for i in first] == [i.segment_id for i in again]


def _run(items, store, extractor, ledger=None, **kwargs):
    return asyncio.run(run_extraction(items, extractor=extractor, store=store, ledger=ledger, now=100.0, **kwargs))


def test_second_run_does_not_rerun(tmp_path):
    store, fake = NoteStore(tmp_path), FakeExtractor()
    first = _run(_inputs(), store, fake)
    second = _run(_inputs(), store, fake)
    assert (first["done"], second["pending"]) == (2, 0)
    assert len(fake.calls) == 2


def test_version_change_recomputes(tmp_path):
    store = NoteStore(tmp_path)
    _run(_inputs(), store, FakeExtractor("v1"))
    assert _run(_inputs(), store, FakeExtractor("v2"))["done"] == 2


def test_failure_backs_off_then_retries(tmp_path):
    store, ledger = NoteStore(tmp_path), JobLedger(tmp_path / "jobs.json", backoff_seconds=10)
    report = _run(_inputs(), store, FakeExtractor(fail_on=lambda _i: True), ledger)
    assert report["failed"] == 2 and ledger.summary()["states"] == {"failed": 2}
    later = asyncio.run(run_extraction(_inputs(), extractor=FakeExtractor(), store=store,
                                       ledger=JobLedger(tmp_path / "jobs.json", backoff_seconds=10), now=200.0))
    assert later["done"] == 2


def test_evidence_out_of_range_is_failed_not_written(tmp_path):
    store = NoteStore(tmp_path)
    report = _run(_inputs(), store, FakeExtractor(out_of_range_on=lambda _i: True))
    assert report["failed"] == 2 and report["notes_rejected"] == 2
    assert store.list_notes(CONV) == []


def test_zero_output_is_skipped_with_reason(tmp_path):
    ledger = JobLedger(None, extractor_version="fake-1")
    _run(_inputs(), NoteStore(tmp_path), FakeExtractor(empty_on=lambda _i: True), ledger)
    assert ledger.summary()["skip_reasons"] == {"no_reusable_info": 2}


def test_budget_picks_oldest_and_returns_rest(tmp_path):
    fake = FakeExtractor()
    report = _run(_inputs(), NoteStore(tmp_path), fake, budget=Budget(max_segments=1))
    assert (report["picked"], report["deferred"], report["done"]) == (1, 1, 1)
    assert fake.calls == [_inputs()[0].segment_id]


def test_open_segment_is_never_extracted(tmp_path):
    report = _run(_inputs(now=2 * HOUR + 10), NoteStore(tmp_path), FakeExtractor())
    assert report["segments_closed"] == 1


def test_identity_extraction_then_gated_retrieval_hides_nothing_on_branch(tmp_path):
    store = NoteStore(tmp_path)
    _run(_inputs(), store, FakeExtractor(identity=True))
    texts = [n["text"] for n in store.list_notes(CONV)]
    assert "her name is Mochi" in texts and not any("SIBLING" in t for t in texts)


def test_relay_helpers():
    exchanges = [[{"content": "x" * 10}], [{"content": "y" * 10}]]
    assert carry_start([1], exchanges) == 1
    assert carry_start([0], exchanges) is None  # must reach the last exchange
    assert ordered_chains(_inputs()) == [[_inputs()[0]], [_inputs()[1]]]


def test_llm_extractor_maps_exchange_numbers_to_event_ids():
    item = _inputs()[1]
    seen = []

    def complete(messages):
        seen.append(messages)
        return '{"notes": [{"kind": "fact", "text": "the cat is Mochi", "exchanges": [0, 7]}]}'

    result = asyncio.run(LLMExtractor(complete).extract(item))
    assert result.notes[0].source_event_ids == ("e-u2", "e-a2")
    assert "[0]" in build_messages(item)[1]["content"] and seen
    assert len(group_exchanges(item.events)) == 1
