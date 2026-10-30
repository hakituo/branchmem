"""Segmenter v1: deterministic segments along one root-to-leaf path of one tree.

Cuts only at complete turn boundaries. Deliberately not done:

- no hard cut at a fixed hour or calendar day (a diary view is a separate projection);
- no cut just because a parent lies in the previous segment (after a size limit the
  next turn's parent is always there);
- never mixing sibling paths or several trees in one batch.

``segment_id`` depends on the segmenter version, the tree and the ordered source event
IDs, so recomputing a path yields the same IDs and derived products can be reused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..events import NormalizedEvent

SEGMENTER_VERSION = "1"

MAX_TURNS = "max_turns"
MAX_CHARS = "max_chars"
IDLE = "idle"
PATH_END = "path_end"


@dataclass
class Turn:
    turn_id: str
    events: List[NormalizedEvent] = field(default_factory=list)

    @property
    def started_at(self) -> float:
        return min((event.occurred_at for event in self.events), default=0.0)

    @property
    def ended_at(self) -> float:
        return max((event.occurred_at for event in self.events), default=0.0)

    @property
    def content_length(self) -> int:
        return sum(
            len(str(event.content or "").strip())
            for event in self.events
            if event.content_eligible
        )

    @property
    def eligible(self) -> bool:
        return any(event.content_eligible for event in self.events)


@dataclass
class Segment:
    segment_id: str
    tree_id: str
    continuity_id: str
    source_event_ids: List[str]
    turn_ids: List[str]
    started_at: float
    ended_at: float
    boundary_reason: str
    topology_status: str = "complete"
    char_count: int = 0
    #: segmenter v2: tail events of the previous segment, shown to the extractor as
    #: read-only context after an in-session topic cut (never counted as sources)
    context_event_ids: List[str] = field(default_factory=list)
    segmenter_version: str = SEGMENTER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "tree_id": self.tree_id,
            "continuity_id": self.continuity_id,
            "source_event_ids": list(self.source_event_ids),
            "turn_ids": list(self.turn_ids),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "boundary_reason": self.boundary_reason,
            "topology_status": self.topology_status,
            "char_count": self.char_count,
            "context_event_ids": list(self.context_event_ids),
            "segmenter_version": self.segmenter_version,
        }


def group_turns(events: Iterable[NormalizedEvent]) -> List[Turn]:
    """Group by ``turn_id``; without one, pair an assistant event with the user event before it."""
    ordered = sorted(events, key=lambda item: (item.occurred_at, item.event_id))
    turns: List[Turn] = []
    for event in ordered:
        turn_id = str(event.turn_id or "").strip()
        if turns and turn_id and turns[-1].turn_id == turn_id:
            turns[-1].events.append(event)
            continue
        if (
            turns
            and not turn_id
            and event.speaker_role == "assistant"
            and turns[-1].events
            and turns[-1].events[-1].speaker_role == "user"
            and not turns[-1].turn_id
        ):
            turns[-1].events.append(event)
            continue
        turns.append(Turn(turn_id=turn_id or f"anon:{event.event_id}", events=[event]))
    return turns


def compute_segment_id(
    tree_id: str, event_ids: Iterable[str], *, version: str = SEGMENTER_VERSION
) -> str:
    basis = "|".join([str(version), str(tree_id or ""), *[str(item) for item in event_ids]])
    return "seg_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def build_segments(
    events: Iterable[NormalizedEvent],
    *,
    tree_id: str = "",
    continuity_id: str = "",
    idle_seconds: float = 1800.0,
    max_turns: int = 40,
    max_chars: int = 8000,
    topology_status: str = "complete",
    include_ineligible: bool = False,
) -> List[Segment]:
    """Segment one path. Parameters are candidates; pick defaults from replay data and A/B tests."""
    turns = group_turns(events)
    if not include_ineligible:
        turns = [turn for turn in turns if turn.eligible]
    segments: List[Segment] = []
    current: List[Turn] = []
    current_chars = 0
    pending_reason = PATH_END

    def flush(reason: str) -> None:
        nonlocal current, current_chars
        if not current:
            return
        source_ids = [event.event_id for turn in current for event in turn.events]
        segments.append(
            Segment(
                segment_id=compute_segment_id(tree_id, source_ids),
                tree_id=tree_id,
                continuity_id=continuity_id,
                source_event_ids=source_ids,
                turn_ids=[turn.turn_id for turn in current],
                started_at=current[0].started_at,
                ended_at=current[-1].ended_at,
                boundary_reason=reason,
                topology_status=topology_status,
                char_count=current_chars,
            )
        )
        current = []
        current_chars = 0

    for turn in turns:
        if current:
            gap = turn.started_at - current[-1].ended_at
            if gap > idle_seconds:
                pending_reason = IDLE
            elif len(current) >= max_turns:
                pending_reason = MAX_TURNS
            elif current_chars >= max_chars:
                pending_reason = MAX_CHARS
            else:
                pending_reason = ""
            if pending_reason:
                # close *before* the new turn starts: segments always end on a turn boundary
                flush(pending_reason)
        current.append(turn)
        current_chars += turn.content_length
    flush(PATH_END)
    return segments


def summarize_segments(segments: List[Segment]) -> Dict[str, Any]:
    turn_counts = [len(segment.turn_ids) for segment in segments]
    char_counts = [segment.char_count for segment in segments]
    boundaries: Dict[str, int] = {}
    for segment in segments:
        boundaries[segment.boundary_reason] = boundaries.get(segment.boundary_reason, 0) + 1
    return {
        "segment_count": len(segments),
        "turn_counts": _distribution(turn_counts),
        "char_counts": _distribution(char_counts),
        "boundary_reasons": boundaries,
        "empty_segments": sum(1 for segment in segments if not segment.source_event_ids),
    }


def _distribution(values: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)

    def pick(ratio: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]

    return {"min": ordered[0], "p50": pick(0.5), "p90": pick(0.9), "max": ordered[-1]}


@dataclass
class SegmentDiff:
    """Difference of two segmentations: derive ``added``, invalidate ``invalidated``."""

    kept: List[str] = field(default_factory=list)
    invalidated: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.invalidated or self.added)


def diff_segments(previous: Iterable[Segment], current: Iterable[Segment]) -> SegmentDiff:
    """Incremental rebuild after late events or replay.

    Segments whose sources did not change keep their ID, so their products are reused.
    A late event only changes the segment it lands in (and later segments shifted by a
    size limit). Raw events are never rewritten.
    """
    old_ids = [segment.segment_id for segment in previous]
    new_ids = [segment.segment_id for segment in current]
    old_set, new_set = set(old_ids), set(new_ids)
    return SegmentDiff(
        kept=[item for item in new_ids if item in old_set],
        invalidated=[item for item in old_ids if item not in new_set],
        added=[item for item in new_ids if item not in old_set],
    )
