"""Scenario format and tree helpers (standard library only).

A scenario is a message tree given in arrival order plus probes asked at an anchor.

- The order of ``events`` is the arrival order. The memory system receives every
  message in that order, including branches abandoned later (in a real app the memory
  system has already processed a message by the time it is abandoned).
- Each event points to its ``parent``; children of the same parent are siblings.
- A probe's ``anchor`` is the current leaf when the question is asked; only the anchor
  and its ancestors form the current branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BRANCH_OPS = ("edit_user", "regenerate", "fork_checkpoint", "switch_back", "update_in_branch")
FACT_TYPES = ("stable_fact", "preference", "plan", "agent_action", "update")
PROBE_KINDS = ("leak", "retain")
ROLES = ("user", "assistant", "tool")


@dataclass(frozen=True)
class Event:
    id: str
    parent: str | None
    role: str
    text: str
    t: int  # arrival time (seconds from scenario start)


@dataclass(frozen=True)
class Probe:
    id: str
    anchor: str
    question: str
    kind: str  # leak: does a sibling-only fact leak? retain: is a current-branch fact still there?
    must_include_any: tuple[str, ...]
    must_not_include: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    branch_op: str
    fact_type: str
    events: tuple[Event, ...]
    probes: tuple[Probe, ...]
    note: str = ""

    @property
    def by_id(self) -> dict[str, Event]:
        return {event.id: event for event in self.events}

    def path(self, anchor: str) -> list[Event]:
        """Ancestor path of the anchor (inclusive), ordered root -> anchor."""

        index = self.by_id
        chain: list[Event] = []
        current: str | None = anchor
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ValueError(f"{self.id}: parent chain has a cycle at {current}")
            seen.add(current)
            event = index[current]
            chain.append(event)
            current = event.parent
        return list(reversed(chain))

    def arrived_before(self, anchor: str) -> list[Event]:
        """Every event the memory system has received when the anchor arrives, in arrival order."""

        cutoff = self.by_id[anchor].t
        return [event for event in self.events if event.t <= cutoff]


def _probe(raw: dict[str, Any]) -> Probe:
    return Probe(
        id=raw["id"],
        anchor=raw["anchor"],
        question=raw["question"],
        kind=raw["kind"],
        must_include_any=tuple(raw.get("must_include_any", ())),
        must_not_include=tuple(raw.get("must_not_include", ())),
        note=raw.get("note", ""),
    )


def parse_scenario(raw: dict[str, Any]) -> Scenario:
    return Scenario(
        id=raw["id"],
        title=raw["title"],
        branch_op=raw["branch_op"],
        fact_type=raw["fact_type"],
        events=tuple(
            Event(e["id"], e.get("parent"), e["role"], e["text"], int(e["t"]))
            for e in raw["events"]
        ),
        probes=tuple(_probe(p) for p in raw["probes"]),
        note=raw.get("note", ""),
    )


def load_scenarios(paths: Iterable[Path]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in sorted(paths):
        scenarios.append(parse_scenario(json.loads(path.read_text(encoding="utf-8"))))
    return scenarios


def load_dir(directory: Path) -> list[Scenario]:
    return load_scenarios(directory.glob("*.json"))
