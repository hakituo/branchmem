"""Prepare one conversation's main line for segmentation.

Steps: branch gate (only the current branch survives) -> drop turns inferred as
regenerated away -> content gate -> merge repeated submissions. Pure functions; the
gate reads trees and cursors but writes nothing.

Background jobs have no request path, so the anchor comes from cursors only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..events import (
    TOPOLOGY_COMPLETE,
    TOPOLOGY_PARTIAL,
    TOPOLOGY_UNKNOWN,
    NormalizedEvent,
    normalize_payload,
)
from ..gating import is_content_eligible
from ..sources import inferred
from ..visibility import filter_raw_events
from ..visibility.anchor_resolution import CursorLookup

_SPACE_RE = re.compile(r"\s+")




@dataclass
class Mainline:
    """The prepared main line of one conversation; ``events`` and ``normalized`` align, in time order."""

    conversation_id: str
    events: list[Dict[str, Any]] = field(default_factory=list)
    normalized: List[NormalizedEvent] = field(default_factory=list)
    aliases: Dict[str, str] = field(default_factory=dict)
    gate_reasons: Dict[str, int] = field(default_factory=dict)
    last_event_at: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def topology_status(self) -> str:
        return conversation_topology_status(self.normalized)


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _turn_of(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(
        payload.get("turn_id") or payload.get("message_id") or metadata.get("message_id") or ""
    ).strip()


def _role(payload: Mapping[str, Any]) -> str:
    return str(payload.get("role") or "").strip().lower()


def is_eligible_payload(payload: Mapping[str, Any]) -> bool:
    """Whether a payload may feed extraction (same rule as ``gating``)."""
    return is_content_eligible(
        str(payload.get("content") or ""), event_type=str(payload.get("event_type") or "")
    )


def merge_repeated_submissions(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], Dict[str, str]]:
    """Merge *consecutive* re-submissions of the same user input, keeping the first.

    Returns (kept events, aliases ``dropped event_id -> kept event_id``). Later events of
    the dropped turn (usually the reply kept after regenerating) move to the kept turn,
    so one exchange is not split into a user-only and a reply-only turn. An assistant
    message in between breaks the run.
    """
    kept: list[Dict[str, Any]] = []
    aliases: Dict[str, str] = {}
    turn_remap: Dict[str, str] = {}
    last_user: tuple[Dict[str, Any], str] | None = None
    for raw in events:
        payload = dict(raw)
        turn_id = _turn_of(payload)
        if turn_id in turn_remap:
            payload["turn_id"] = turn_remap[turn_id]
        if _role(payload) == "user":
            text = _SPACE_RE.sub("", str(payload.get("content") or ""))
            if last_user is not None and text and text == last_user[1]:
                keeper = last_user[0]
                aliases[str(payload.get("event_id") or "")] = str(keeper.get("event_id") or "")
                keeper_turn = _turn_of(keeper)
                if turn_id and keeper_turn and turn_id != keeper_turn:
                    turn_remap[turn_id] = keeper_turn
                continue
            last_user = (payload, text)
        else:
            last_user = None
        kept.append(payload)
    aliases.pop("", None)
    return kept, aliases


def conversation_topology_status(events: Sequence[NormalizedEvent]) -> str:
    """Topology status of a whole main line: all complete / all unknown, else partial.

    The line is one root-to-leaf path, so a first event with a node and no parent is the
    root and counts as complete; a missing parent elsewhere is a real ``partial``.
    """
    statuses = {
        TOPOLOGY_COMPLETE
        if index == 0 and event.client_node_id and event.topology_status == TOPOLOGY_PARTIAL
        else event.topology_status
        for index, event in enumerate(events)
    }
    if not statuses or statuses == {TOPOLOGY_UNKNOWN}:
        return TOPOLOGY_UNKNOWN
    if statuses == {TOPOLOGY_COMPLETE}:
        return TOPOLOGY_COMPLETE
    return TOPOLOGY_PARTIAL


def prepare_mainline(
    raw_events: Sequence[Mapping[str, Any]],
    *,
    conversation_id: str,
    registry: Any = None,
    tree_reader: Any = None,
    cursors: Optional[CursorLookup] = None,
    mode: str = "compat",
) -> Mainline:
    raw = [dict(item) for item in raw_events if isinstance(item, Mapping)]
    raw.sort(key=lambda item: (_as_float(item.get("timestamp")), str(item.get("event_id") or "")))
    result = Mainline(conversation_id=conversation_id)
    result.last_event_at = max((_as_float(item.get("timestamp")) for item in raw), default=0.0)
    result.counts["raw"] = len(raw)
    if not raw:
        return result
    gated = filter_raw_events(
        raw,
        conversation_id=conversation_id,
        mode=mode,
        registry=registry,
        tree_reader=tree_reader,
        cursors=cursors,
    )
    result.gate_reasons = dict(gated.reasons)
    result.counts["after_gate"] = len(gated.events)
    deduped = inferred.drop_superseded(gated.events)
    result.counts["after_superseded"] = len(deduped)
    eligible = [item for item in deduped if is_eligible_payload(item)]
    result.counts["after_content_gate"] = len(eligible)
    merged, aliases = merge_repeated_submissions(eligible)
    result.counts["merged_repeats"] = len(aliases)
    result.events = merged
    result.aliases = aliases
    result.normalized = [normalize_payload(payload) for payload in merged]
    return result
