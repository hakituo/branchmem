"""Scenario validation: every probe must test what it claims to test.

Structure: unique IDs, parents arrive before children, monotonic time, valid enum
values, existing anchors.

Semantics (the important part):

- leak probes: every ``must_not_include`` word appears **only off the anchor path** and
  has arrived before the question (otherwise it is not a sibling leak but a broken
  probe); at least one ``must_include_any`` word is on the anchor path.
- retain probes: at least one ``must_include_any`` word is on the anchor path;
  ``must_not_include`` words, if any, are likewise off the path.
- every scenario has at least one real fork (a parent with two or more children).
"""

from __future__ import annotations

from collections import Counter

from branchbench.score import normalize
from branchbench.schema import BRANCH_OPS, FACT_TYPES, PROBE_KINDS, ROLES, Scenario


def _text(events) -> str:
    """Joined and normalized exactly like answers are at scoring time."""
    return "\n".join(normalize(event.text) for event in events)


def validate(scenario: Scenario) -> list[str]:
    errors: list[str] = []
    sid = scenario.id
    if scenario.branch_op not in BRANCH_OPS:
        errors.append(f"{sid}: invalid branch_op {scenario.branch_op!r}")
    if scenario.fact_type not in FACT_TYPES:
        errors.append(f"{sid}: invalid fact_type {scenario.fact_type!r}")
    ids = [event.id for event in scenario.events]
    for dup, count in Counter(ids).items():
        if count > 1:
            errors.append(f"{sid}: duplicate event ID {dup}")
    seen: set[str] = set()
    last_t = -1
    for event in scenario.events:
        if event.role not in ROLES:
            errors.append(f"{sid}/{event.id}: invalid role {event.role!r}")
        if event.parent is not None and event.parent not in seen:
            errors.append(f"{sid}/{event.id}: parent {event.parent} has not arrived first")
        if event.t < last_t:
            errors.append(f"{sid}/{event.id}: arrival time goes backwards")
        last_t = max(last_t, event.t)
        seen.add(event.id)
    children = Counter(event.parent for event in scenario.events if event.parent)
    if not any(count >= 2 for count in children.values()):
        errors.append(f"{sid}: no fork at all")
    probe_ids = Counter(probe.id for probe in scenario.probes)
    for dup, count in probe_ids.items():
        if count > 1:
            errors.append(f"{sid}: duplicate probe ID {dup}")
    if errors:
        return errors
    for probe in scenario.probes:
        errors.extend(_validate_probe(scenario, probe))
    return errors


def _validate_probe(scenario: Scenario, probe) -> list[str]:
    tag = f"{scenario.id}/{probe.id}"
    if probe.kind not in PROBE_KINDS:
        return [f"{tag}: invalid kind {probe.kind!r}"]
    if probe.anchor not in scenario.by_id:
        return [f"{tag}: anchor {probe.anchor} does not exist"]
    errors: list[str] = []
    path = scenario.path(probe.anchor)
    path_ids = {event.id for event in path}
    on_path = _text(path)
    off_path = _text(e for e in scenario.arrived_before(probe.anchor) if e.id not in path_ids)
    if not probe.must_include_any:
        errors.append(f"{tag}: must_include_any is empty")
    elif not any(normalize(word) in on_path for word in probe.must_include_any):
        errors.append(f"{tag}: no answer word is on the current branch path")
    for word in probe.must_not_include:
        word = normalize(word)
        if word in on_path:
            errors.append(f"{tag}: forbidden word {word!r} is on the current branch path")
        if word not in off_path:
            errors.append(f"{tag}: forbidden word {word!r} is not in a sibling that arrived before the question")
    if probe.kind == "leak" and not probe.must_not_include:
        errors.append(f"{tag}: a leak probe needs forbidden words")
    return errors


def validate_all(scenarios) -> list[str]:
    errors: list[str] = []
    ids = Counter(scenario.id for scenario in scenarios)
    for dup, count in ids.items():
        if count > 1:
            errors.append(f"duplicate scenario ID {dup}")
    for scenario in scenarios:
        errors.extend(validate(scenario))
    return errors
