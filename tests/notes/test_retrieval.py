"""Gated hybrid retrieval: sibling-derived notes never reach the result list."""

from __future__ import annotations

import pytest

from branchmem.extraction.contracts import NoteDraft
from branchmem.notes import NoteRetriever, NoteStore, reset_note_retrieval_cache

CONV = "conv-main"


def _event(node, parent, text, ts):
    return {"event_id": f"e-{node}", "conversation_id": CONV, "role": "user", "content": text,
            "timestamp": ts, "client_node_id": node, "parent_node_id": parent}


# root -> a -> {chengdu, hangzhou}: the user edited "moving to Chengdu" into "Hangzhou"
EVENTS = [
    _event("root", "", "thinking about moving", 1),
    _event("a", "root", "my cat is called Mochi", 2),
    _event("cd", "a", "decided: moving to Chengdu", 3),
    _event("hz", "a", "decided: moving to Hangzhou", 4),
]


@pytest.fixture
def store(tmp_path):
    reset_note_retrieval_cache()
    store = NoteStore(tmp_path / "notes")
    drafts = [
        NoteDraft("fact", "the cat is called Mochi", ("e-a",)),
        NoteDraft("plan", "moving to Chengdu next month", ("e-cd",)),
        NoteDraft("plan", "moving to Hangzhou next month", ("e-hz",)),
        NoteDraft("episode", "talked about moving and the cat", ("e-a", "e-cd")),
    ]
    store.replace_segment(CONV, "seg", "v1", drafts, segment_source_event_ids=[e["event_id"] for e in EVENTS])
    return store


def _texts(hits):
    return [hit["text"] for hit in hits]


def test_note_hidden_when_any_source_is_on_sibling_branch(store):
    retriever = NoteRetriever(store, tree_reader=lambda _c: EVENTS)
    hits = retriever.search("moving next month", conversation_id=CONV, active_path_ids={"root", "a", "hz"})
    assert "moving to Hangzhou next month" in _texts(hits)
    assert not any("Chengdu" in text for text in _texts(hits))
    assert retriever.last_gate_reasons["not_on_anchor_path"] == 2  # the plan and the mixed episode


def test_switch_back_restores_the_old_branch(store):
    retriever = NoteRetriever(store, tree_reader=lambda _c: EVENTS)
    hits = retriever.search("moving", conversation_id=CONV, active_path_ids={"root", "a", "cd"})
    assert "moving to Chengdu next month" in _texts(hits)
    assert "moving to Hangzhou next month" not in _texts(hits)


def test_hits_carry_provenance_and_score_parts(store):
    hit = NoteRetriever(store, tree_reader=lambda _c: EVENTS).search(
        "cat", conversation_id=CONV, active_path_ids={"hz"})[0]
    assert hit["source_event_ids"] == ["e-a"] and hit["source_segment_id"] == "seg"
    assert hit["score"] == pytest.approx(hit["rrf_score"] + hit["importance_adjust"])


def test_corpus_refreshes_after_new_write(store):
    retriever = NoteRetriever(store, tree_reader=lambda _c: EVENTS)
    assert retriever.search("piano", conversation_id=CONV) == []
    store.replace_segment(CONV, "seg2", "v1", [NoteDraft("fact", "plays piano", ("e-a",))],
                          segment_source_event_ids=["e-a"])
    assert _texts(retriever.search("piano", conversation_id=CONV)) == ["plays piano"]


def test_vectors_are_cached_and_reranked(store):
    np = pytest.importorskip("numpy")
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return [[1.0, 0.0] if "cat" in t or "pet" in t else [0.0, 1.0] for t in texts]

    retriever = NoteRetriever(store, embedder=embed, tree_reader=lambda _c: EVENTS)
    hits = retriever.search("pet", conversation_id=CONV, active_path_ids={"hz"})
    assert hits[0]["text"] == "the cat is called Mochi"  # BM25 has no match; vectors find it
    retriever.search("pet", conversation_id=CONV, active_path_ids={"hz"})
    assert calls == [4, 1, 1]  # notes embedded once, then only the query
    assert (store.root / "_vectors").exists() and np is not None


def test_dense_failure_falls_back_to_bm25(store):
    def broken(_texts):
        raise RuntimeError("model down")

    hits = NoteRetriever(store, embedder=broken, tree_reader=lambda _c: EVENTS).search(
        "cat", conversation_id=CONV, active_path_ids={"hz"})
    assert _texts(hits)[0] == "the cat is called Mochi"
