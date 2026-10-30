"""The ``branchmem`` baseline: the reference library run end to end on a scenario.

Unlike ``source_gated`` (a ten-line set check), this goes through the real pipeline:
scenario events become stored payloads, the tree is built by ``TreeBuilder`` from
parent pointers, notes are created on arrival with identity extraction, and both the
raw read and the note read go through the library's gates in strict mode, with the
probe's anchor as the client-supplied active path.
"""

from __future__ import annotations

from typing import Any, Dict, List

from branchbench.schema import Probe, Scenario
from branchmem.visibility import MODE_STRICT, DerivedGate, filter_raw_events


def _payloads(scenario: Scenario, probe: Probe) -> List[Dict[str, Any]]:
    return [
        {
            "event_id": event.id,
            "conversation_id": scenario.id,
            "role": event.role,
            "content": event.text,
            "timestamp": float(event.t),
            "client_node_id": event.id,
            "parent_node_id": event.parent or "",
        }
        for event in scenario.arrived_before(probe.anchor)
    ]


def branchmem_context(scenario: Scenario, probe: Probe) -> str:
    payloads = _payloads(scenario, probe)
    reader = lambda _cid: payloads  # noqa: E731 - the store as seen at question time
    gate_args = {"conversation_id": scenario.id, "active_path_ids": [probe.anchor], "mode": MODE_STRICT}
    raw = filter_raw_events(payloads, tree_reader=reader, **gate_args).events
    notes = [
        {"id": f"note-{p['event_id']}", "conversation_id": scenario.id,
         "source_event_ids": [p["event_id"]], "text": p["content"]}
        for p in payloads
        if p["role"] in ("user", "assistant", "tool")
    ]
    visible = DerivedGate(tree_reader=reader, **gate_args).filter(notes)
    lines = [f"[{p['role']}] {p['content']}" for p in sorted(raw, key=lambda p: p["timestamp"])]
    return "\n".join(lines) + "\n[memory]\n" + "\n".join(note["text"] for note in visible)
