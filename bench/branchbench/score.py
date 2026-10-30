"""Deterministic scoring: keywords only, no LLM judge, fully reproducible.

Both levels share one rule:

- **context level**: does the text the memory system prepares for the answer model
  contain a sibling-only fact (leaked), and the fact the current branch needs (hit)?
  No model is called.
- **answer level**: does the final answer contain a sibling fact (leaked), and is it
  right (correct = an answer word is present and nothing leaked)?
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from branchbench.schema import Probe, Scenario


@dataclass(frozen=True)
class ProbeResult:
    scenario_id: str
    probe_id: str
    kind: str
    branch_op: str
    fact_type: str
    leaked: bool
    hit: bool  # any answer word present
    correct: bool  # hit and not leaked

    def to_dict(self) -> dict:
        return asdict(self)


def judge(text: str, probe: Probe) -> tuple[bool, bool]:
    leaked = any(word in text for word in probe.must_not_include)
    hit = any(word in text for word in probe.must_include_any)
    return leaked, hit


def score_texts(scenarios: Iterable[Scenario], texts: dict[tuple[str, str], str]) -> list[ProbeResult]:
    """``texts`` keyed by (scenario ID, probe ID); a missing probe counts as empty text."""

    results: list[ProbeResult] = []
    for scenario in scenarios:
        for probe in scenario.probes:
            text = texts.get((scenario.id, probe.id), "")
            leaked, hit = judge(text, probe)
            results.append(
                ProbeResult(
                    scenario_id=scenario.id,
                    probe_id=probe.id,
                    kind=probe.kind,
                    branch_op=scenario.branch_op,
                    fact_type=scenario.fact_type,
                    leaked=leaked,
                    hit=hit,
                    correct=hit and not leaked,
                )
            )
    return results


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(results: list[ProbeResult]) -> dict:
    """Leak rate (leak probes only), hit and correct rates (all probes), overall and per branch op."""

    def block(items: list[ProbeResult]) -> dict:
        leak_items = [r for r in items if r.kind == "leak"]
        return {
            "probes": len(items),
            "leak_probes": len(leak_items),
            "leak_rate": _rate([r.leaked for r in leak_items]),
            "hit_rate": _rate([r.hit for r in items]),
            "correct_rate": _rate([r.correct for r in items]),
        }

    by_op: dict[str, list[ProbeResult]] = defaultdict(list)
    for result in results:
        by_op[result.branch_op].append(result)
    return {
        "overall": block(results),
        "by_branch_op": {op: block(items) for op, items in sorted(by_op.items())},
    }
