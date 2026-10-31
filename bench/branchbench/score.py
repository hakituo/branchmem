"""Deterministic scoring: keywords only, no LLM judge, fully reproducible.

Both levels share one rule:

- **context level**: does the text the memory system prepares for the answer model
  contain a sibling-only fact (leaked), and the fact the current branch needs (hit)?
  No model is called.
- **answer level**: does the final answer contain a sibling fact (leaked), and is it
  right (correct = an answer word is present and nothing leaked)?
"""

from __future__ import annotations

import re

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


_IGNORED = str.maketrans("", "", " \t\r\n\u3000*,，")
_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_NUMERAL = re.compile(r"[零一二两三四五六七八九十百千]+")


def _cn_to_int(run: str) -> int:
    """Chinese numeral run -> int, colloquial tails included (一千五 = 1500, 两百五 = 250)."""

    total, digit, last_unit = 0, 0, 1
    for char in run:
        if char in _DIGITS:
            digit = _DIGITS[char]
        else:
            unit = _UNITS[char]
            total += (digit or 1) * unit
            digit, last_unit = 0, unit
    if digit:
        total += digit * (last_unit // 10 if last_unit >= 100 else 1)
    return total


def normalize(text: str) -> str:
    """Formatting-only normalization before matching, applied to the text and to the
    keywords alike and identically for every system:

    - whitespace, Markdown emphasis and thousands separators are dropped
      ("晚上 9 点" -> "晚上9点", "**杭州**" -> "杭州", "4,000" -> "4000");
    - runs of Chinese numerals become Arabic digits ("三年级" -> "3年级", "十一点" -> "11点",
      "一千五" -> "1500"), so "九点" and "9点" are the same keyword.
    """

    plain = str(text or "").translate(_IGNORED)
    return _NUMERAL.sub(lambda m: str(_cn_to_int(m.group(0))), plain)


def judge(text: str, probe: Probe) -> tuple[bool, bool]:
    plain = normalize(text)
    leaked = any(normalize(word) in plain for word in probe.must_not_include)
    hit = any(normalize(word) in plain for word in probe.must_include_any)
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
