"""Extract notes along the current branch, then retrieve them through the gate.

Uses the fake identity extractor (one note per message), so no model is called.

    python examples/03_extract_and_retrieve.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from branchmem.extraction import build_conversation_inputs, run_extraction
from branchmem.extraction.testing import FakeExtractor
from branchmem.notes import NoteRetriever, NoteStore

CONV = "trip"


def msg(node, parent, role, text, t):
    return {"event_id": node, "conversation_id": CONV, "role": role, "content": text,
            "timestamp": t, "client_node_id": node, "parent_node_id": parent}


# An agent forked from a checkpoint: the first plan (Kyoto) was abandoned for Osaka.
events = [
    msg("u1", "", "user", "Plan a 3-day trip in Japan, I hate early flights.", 0),
    msg("a1", "u1", "assistant", "Noted: no early flights.", 10),
    msg("k1", "a1", "assistant", "Plan A: three days in Kyoto.", 20),
    msg("o1", "a1", "assistant", "Plan B: three days in Osaka.", 30),  # fork from checkpoint a1
]

with tempfile.TemporaryDirectory() as tmp:
    store = NoteStore(Path(tmp) / "notes")
    reader = lambda _cid: events  # noqa: E731

    # Extraction only ever sees the current branch (cursor at o1), not the abandoned plan.
    items = build_conversation_inputs(events, conversation_id=CONV, now=10_000,
                                      tree_reader=reader, cursors=lambda _cid: ["o1"])
    report = asyncio.run(run_extraction(items, extractor=FakeExtractor(identity=True), store=store))
    print("extraction:", {k: report[k] for k in ("done", "skipped", "failed", "notes_written")})

    # Suppose another job had summarized the Kyoto branch before it was abandoned:
    store.replace_segment(CONV, "old-seg", "other-extractor",
                          [{"kind": "plan", "text": "Plan A: three days in Kyoto.", "source_event_ids": ["k1"]}],
                          segment_source_event_ids=["k1"])

    retriever = NoteRetriever(store, tree_reader=reader)
    for leaf in ("o1", "k1"):
        hits = retriever.search("three days plan flights", conversation_id=CONV, active_path_ids=[leaf], limit=5)
        print(f"leaf={leaf}:", [h["text"] for h in hits], retriever.last_gate_reasons)
