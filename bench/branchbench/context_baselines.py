"""Context-level baselines: no model calls, they only decide what the answer model sees.

They show the probes separate leaky from non-leaky ways of reading memory, and bound
later LLM baselines. "Memory items" use the simplest identity extraction: each
message yields one item when it arrives, citing itself as the source. Real summaries
merge several sources, which only makes leaks harder to stop, so this is an
optimistic lower bound.

- ``all_history``: all arrived messages in arrival order (a branch-blind linear log).
- ``raw_filtered_notes_unfiltered``: raw messages filtered to the current branch, but
  every memory item created on arrival is kept (filter the raw text, not derivatives).
- ``source_gated``: a memory item is visible only if all its sources are on the path.
- ``path_only``: only the raw messages on the current branch path (the ideal raw read).
- ``branchmem``: the reference library end to end (see ``library_baseline``).
"""

from __future__ import annotations

from typing import Callable

from branchbench.schema import Event, Probe, Scenario

ContextFn = Callable[[Scenario, Probe], str]


def _render(events: list[Event]) -> str:
    return "\n".join(f"[{event.role}] {event.text}" for event in events)


def _notes(scenario: Scenario, probe: Probe) -> list[tuple[frozenset[str], str]]:
    """Identity extraction: one item per message arrived before the question, citing itself."""

    return [
        (frozenset({event.id}), event.text)
        for event in scenario.arrived_before(probe.anchor)
        if event.role in ("user", "assistant", "tool")
    ]


def all_history(scenario: Scenario, probe: Probe) -> str:
    return _render(scenario.arrived_before(probe.anchor))


def path_only(scenario: Scenario, probe: Probe) -> str:
    return _render(scenario.path(probe.anchor))


def raw_filtered_notes_unfiltered(scenario: Scenario, probe: Probe) -> str:
    notes = "\n".join(text for _, text in _notes(scenario, probe))
    return path_only(scenario, probe) + "\n[memory]\n" + notes


def source_gated(scenario: Scenario, probe: Probe) -> str:
    visible = {event.id for event in scenario.path(probe.anchor)}
    notes = "\n".join(text for sources, text in _notes(scenario, probe) if sources <= visible)
    return path_only(scenario, probe) + "\n[memory]\n" + notes


BASELINES: dict[str, ContextFn] = {
    "all_history": all_history,
    "raw_filtered_notes_unfiltered": raw_filtered_notes_unfiltered,
    "source_gated": source_gated,
    "path_only": path_only,
}

try:  # the reference implementation is optional for the benchmark itself
    from branchbench.library_baseline import branchmem_context

    BASELINES["branchmem"] = branchmem_context
except ImportError:  # pragma: no cover - branchmem not importable
    pass


def contexts(scenarios: list[Scenario], fn: ContextFn) -> dict[tuple[str, str], str]:
    return {(s.id, p.id): fn(s, p) for s in scenarios for p in s.probes}
