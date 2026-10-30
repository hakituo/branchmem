"""A background job with no request path still respects the current branch.

The JSONL store notifies the tree registry and the cursor store on every write; a
"nightly" read with no client path uses the cursor as its anchor.

    python examples/02_store_and_cursors.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from branchmem import CursorStore, TreeRegistry, filter_raw_events
from branchmem.sources.jsonl_store import JsonlEventStore


def msg(node, parent, role, text, t):
    return {"conversation_id": "chat", "role": role, "content": text, "timestamp": t,
            "client_node_id": node, "parent_node_id": parent}


with tempfile.TemporaryDirectory() as tmp:
    store = JsonlEventStore(Path(tmp) / "events")
    registry = TreeRegistry(store)
    cursors = CursorStore(Path(tmp) / "cursors", registry=registry)
    store.add_listener(registry.on_append)
    store.add_listener(cursors.advance)

    store.append([msg("u1", "", "user", "Book a table for Friday.", 1)])
    store.append([msg("a1", "u1", "assistant", "Booked for Friday 7pm.", 2)])
    registry.tree("chat")
    store.append([msg("a1b", "u1", "assistant", "Booked for Saturday 8pm.", 3)])  # regenerated reply
    print("cursor after regenerate:", cursors.cursor_for("chat"))

    events = store.list_conversation_events("chat")
    nightly = filter_raw_events(events, registry=registry, cursors=cursors.candidates)
    print("nightly sees:", [e["content"] for e in nightly.events], nightly.reasons)

    store.append([msg("u2", "a1", "user", "Actually, Friday works best.", 4)])  # user went back
    print("cursor after switching back:", cursors.cursor_for("chat"))
    nightly = filter_raw_events(store.list_conversation_events("chat"), registry=registry,
                                cursors=cursors.candidates)
    print("nightly sees:", [e["content"] for e in nightly.events], nightly.reasons)
    print("audit log:", (Path(tmp) / "cursors" / "anchor_events.jsonl").read_text().count("\n"), "lines")
