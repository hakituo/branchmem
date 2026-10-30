"""Infer regenerations in linear history that has no tree.

When a client regenerates a reply it often re-submits the same user input, and a
server that only appends sees several turns in a row with the same user text and
different replies. Rule:

- group events into turns by ``message_id`` / ``turn_id``, ordered by time;
- adjacent turns whose user text (whitespace removed) is identical and whose gap is
  at most ``max_gap_seconds`` are the same position;
- keep the last one; replies and thoughts of the earlier turns are marked superseded
  (``superseded_by`` points at the kept turn). The user message itself reuses one node
  and stays on the main line.

The result is **not topology**: it writes no node IDs and never takes part in strict
visibility. Use it only where dropping a duplicate is the safe choice, such as
de-duplicating extraction input. Never use it to hide content on the read path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

DEFAULT_MAX_GAP_SECONDS = 30 * 60
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class InferredTurn:
    turn_id: str
    started_at: float
    user_text: str
    event_ids: tuple


@dataclass(frozen=True)
class Superseded:
    turn_id: str
    superseded_by: str
    run_size: int


def _normalize(text: str) -> str:
    return _SPACE_RE.sub("", str(text or ""))


def group_turns(events: Iterable[Mapping[str, Any]]) -> List[InferredTurn]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for event in events:
        turn_id = str(event.get("message_id") or event.get("turn_id") or event.get("event_id") or "")
        if turn_id:
            buckets.setdefault(turn_id, []).append(event)
    turns: List[InferredTurn] = []
    for turn_id, items in buckets.items():
        users = [item for item in items if str(item.get("role") or "").lower() == "user"]
        turns.append(
            InferredTurn(
                turn_id=turn_id,
                started_at=min(float(item.get("timestamp") or 0.0) for item in items),
                user_text=_normalize(users[0].get("content")) if users else "",
                event_ids=tuple(str(item.get("event_id") or "") for item in items),
            )
        )
    turns.sort(key=lambda turn: (turn.started_at, turn.turn_id))
    return turns


def infer_regenerations(
    events: Sequence[Mapping[str, Any]],
    *,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> Dict[str, Superseded]:
    """``superseded turn_id -> Superseded``. A turn without user input (e.g. a proactive
    message) breaks a run."""
    turns = group_turns(events)
    marks: Dict[str, Superseded] = {}
    run: List[InferredTurn] = []

    def close() -> None:
        if len(run) > 1:
            keeper = run[-1].turn_id
            for turn in run[:-1]:
                marks[turn.turn_id] = Superseded(turn.turn_id, keeper, len(run))

    for turn in turns:
        same = (
            run
            and turn.user_text
            and turn.user_text == run[-1].user_text
            and turn.started_at - run[-1].started_at <= max_gap_seconds
        )
        if same:
            run.append(turn)
            continue
        close()
        run = [turn] if turn.user_text else []
    close()
    return marks


def superseded_event_ids(
    events: Sequence[Mapping[str, Any]], **kwargs: Any
) -> set:
    """Reply / thought events of superseded turns.

    The user message is the same tree node across regenerations and stays on the main
    line, so it is not marked; only that turn's replies are superseded.
    """
    marks = infer_regenerations(events, **kwargs)
    return {
        str(event.get("event_id") or "")
        for event in events
        if str(event.get("message_id") or event.get("turn_id") or "") in marks
        and str(event.get("role") or "").lower() != "user"
    }


def drop_superseded(events: Sequence[Mapping[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
    """De-duplicate extraction input by dropping turns inferred as regenerated away."""
    dropped = superseded_event_ids(events, **kwargs)
    return [dict(event) for event in events if str(event.get("event_id") or "") not in dropped]


__all__ = [
    "DEFAULT_MAX_GAP_SECONDS",
    "InferredTurn",
    "Superseded",
    "drop_superseded",
    "group_turns",
    "infer_regenerations",
    "superseded_event_ids",
]
