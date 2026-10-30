"""Siblings never see each other: an edited user message and the notes derived from it.

    python examples/01_sibling_isolation.py
"""

from __future__ import annotations

from branchmem.visibility import MODE_STRICT, DerivedGate, filter_raw_events

CONV = "demo"


def msg(node, parent, role, text, t):
    return {"event_id": node, "conversation_id": CONV, "role": role, "content": text,
            "timestamp": t, "client_node_id": node, "parent_node_id": parent}


# The user first said "Chengdu", then edited that message to "Hangzhou".
# Both versions were stored and both were already summarized.
events = [
    msg("u1", "", "user", "I have a cat called Mochi.", 1),
    msg("a1", "u1", "assistant", "Mochi is a nice name.", 2),
    msg("u2", "a1", "user", "I'm moving to Chengdu next month.", 3),
    msg("a2", "u2", "assistant", "Chengdu has great food.", 4),
    msg("u2b", "a1", "user", "I'm moving to Hangzhou next month.", 5),  # the edit: a sibling of u2
    msg("a2b", "u2b", "assistant", "Hangzhou is lovely in spring.", 6),
]
notes = [
    {"id": "n1", "conversation_id": CONV, "text": "has a cat named Mochi", "source_event_ids": ["u1"]},
    {"id": "n2", "conversation_id": CONV, "text": "moving to Chengdu", "source_event_ids": ["u2"]},
    {"id": "n3", "conversation_id": CONV, "text": "moving to Hangzhou", "source_event_ids": ["u2b"]},
]


def show(leaf):
    common = {"conversation_id": CONV, "active_path_ids": [leaf], "mode": MODE_STRICT,
              "tree_reader": lambda _cid: events}
    raw = filter_raw_events(events, **common)
    gate = DerivedGate(**common)
    print(f"current leaf = {leaf}")
    print("  raw   :", [e["content"] for e in raw.events])
    print("  notes :", [n["text"] for n in gate.filter(notes)])
    print("  hidden:", raw.reasons, dict(gate.reasons))


show("a2b")  # the edited branch: Chengdu is gone, Mochi (shared prefix) stays
show("a2")   # switch back: only the anchor moved; nothing was marked inactive
