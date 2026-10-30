"""Note store: idempotent writes, evidence checks, version history."""

from __future__ import annotations

import pytest

from branchmem._util import json_load
from branchmem.extraction.contracts import NoteDraft
from branchmem.notes import NoteStore


@pytest.fixture
def store(tmp_path):
    return NoteStore(tmp_path / "notes", namespace="role_a")


def _draft(text="likes tea", evidence=("e1",), **kwargs):
    return NoteDraft(kind="fact", text=text, source_event_ids=evidence, **kwargs)


def _write(store, drafts, version="v1", sources=("e1", "e2")):
    return store.replace_segment("conv", "seg1", version, drafts, segment_source_event_ids=sources)


def test_rerun_no_duplicates_and_has_segment(store):
    first = _write(store, [_draft(), _draft("likes  tea")])
    assert (first.written, first.duplicate) == (1, 1)
    _write(store, [_draft()])
    assert len(store.list_notes("conv")) == 1
    assert store.has_segment("conv", "seg1", "v1")
    note = store.list_notes("conv")[0]
    assert note["source_event_ids"] == ["e1"] and note["visibility"] == "tree_local"


def test_version_change_not_done_and_history_kept(store):
    _write(store, [_draft()])
    assert not store.has_segment("conv", "seg1", "v2")
    _write(store, [_draft("likes green tea")], version="v2")
    assert [h["extractor_version"] for h in store.segment_history("conv", "seg1")] == ["v1"]
    assert store.list_notes("conv")[0]["text"] == "likes green tea"


def test_out_of_range_evidence_rejected(store):
    result = _write(store, [_draft(evidence=("e1", "e9")), _draft("ok", ("e2",))])
    assert (result.written, result.rejected) == (1, 1)


def test_all_rejected_is_not_done(store):
    _write(store, [_draft(evidence=("zzz",))])
    assert not store.has_segment("conv", "seg1", "v1")


def test_drop_segment_moves_notes_to_history(store):
    _write(store, [_draft()])
    assert store.drop_segment("conv", "seg1")
    assert store.list_notes("conv") == []
    assert store.segment_history("conv", "seg1")[-1]["dropped"] is True
    assert not store.drop_segment("conv", "missing")


def test_importance_zero_kept_and_bad_value_tolerated(store):
    _write(store, [_draft(importance=0.0), _draft("x", importance="bad")])
    values = sorted(n["importance"] for n in store.list_notes("conv"))
    assert values == [0.0, 0.5]


def test_sanitized_conversation_ids_do_not_collide(tmp_path):
    store = NoteStore(tmp_path)
    for cid in ("a/b", "a:b"):
        store.replace_segment(cid, "s", "v1", [_draft()], segment_source_event_ids=["e1"])
    assert sorted(store.list_conversations()) == ["a/b", "a:b"]


def test_extra_fields_are_copied_from_metadata(tmp_path):
    store = NoteStore(tmp_path, extra_fields=("status",))
    store.replace_segment("c", "s", "v1", [_draft(metadata={"status": "open", "other": 1})],
                          segment_source_event_ids=["e1"])
    note = store.list_notes("c")[0]
    assert note["status"] == "open" and "other" not in note
    assert json_load(store.path("c"))["conversation_id"] == "c"
