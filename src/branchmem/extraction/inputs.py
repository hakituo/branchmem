"""Turn one conversation's raw events into ``SegmentInput``s along the current branch.

Segmenter ``v1`` cuts by turns / characters / idle gaps; ``v2`` uses exchanges, idle
gaps and topic depth (needs a scorer; a cut ledger keeps replays stable). The last
segment stays open (``closed=False``) until the conversation has been idle for
``idle_seconds``.

``tree_id`` used for segment IDs is the conversation ID: root sets can change after a
back-fill or a graft, and that must not invalidate every segment of the conversation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

from ..segments.topic import TopicParams, build_topic_segments
from ..segments.topic_scoring import CutLedger
from ..segments.v1 import build_segments
from . import mainline
from .contracts import SegmentInput


@dataclass(frozen=True)
class SegmentParams:
    segmenter: str = "v1"
    idle_seconds: float = 1800.0
    # v1
    max_turns: int = 8
    max_chars: int = 4000
    # v2
    session_max_chars: int = 4000
    min_part_chars: int = 300


def _v2_segments(line: mainline.Mainline, conversation_id: str, continuity_id: str,
                 params: SegmentParams, scorer: Any, ledger: Optional[CutLedger], tail_closed: bool):
    result = build_topic_segments(
        line.normalized,
        tree_id=conversation_id,
        continuity_id=continuity_id,
        params=TopicParams(
            idle_seconds=params.idle_seconds,
            session_max_chars=params.session_max_chars,
            min_part_chars=params.min_part_chars,
        ),
        scorer=scorer,
        lookup=ledger.lookup if ledger is not None else None,
        topology_status=line.topology_status,
    )
    if ledger is not None:
        for digest, cuts, is_tail in result.sessions:
            if not is_tail or tail_closed:  # the last session may still grow
                ledger.record(digest, cuts)
    return result.segments, 0 if tail_closed else result.tail_segments


def build_conversation_inputs(
    raw_events: Sequence[Mapping[str, Any]],
    *,
    conversation_id: str,
    namespace: str = "default",
    params: Optional[SegmentParams] = None,
    now: Optional[float] = None,
    registry: Any = None,
    tree_reader: Any = None,
    cursors: Any = None,
    scorer: Any = None,
    ledger: Optional[CutLedger] = None,
) -> List[SegmentInput]:
    """All segments of one conversation, the open tail included (``closed`` says which)."""
    params = params or SegmentParams()
    stamp = time.time() if now is None else float(now)
    line = mainline.prepare_mainline(
        raw_events, conversation_id=conversation_id,
        registry=registry, tree_reader=tree_reader, cursors=cursors,
    )
    if not line.normalized:
        return []
    continuity_id = line.normalized[0].continuity_id
    tail_closed = stamp - line.last_event_at > params.idle_seconds
    open_count = 0 if tail_closed else 1
    if params.segmenter == "v2":
        segments, open_count = _v2_segments(
            line, conversation_id, continuity_id, params, scorer, ledger, tail_closed
        )
    else:
        segments = build_segments(
            line.normalized,
            tree_id=conversation_id,
            continuity_id=continuity_id,
            idle_seconds=params.idle_seconds,
            max_turns=params.max_turns,
            max_chars=params.max_chars,
            topology_status=line.topology_status,
        )
    by_id = {str(item.get("event_id") or ""): item for item in line.events}
    inputs: List[SegmentInput] = []
    for index, segment in enumerate(segments):
        source = set(segment.source_event_ids)
        inputs.append(
            SegmentInput(
                namespace=namespace,
                conversation_id=conversation_id,
                segment=segment,
                events=tuple(by_id[eid] for eid in segment.source_event_ids if eid in by_id),
                closed=index < len(segments) - open_count,
                aliases=tuple(sorted((d, k) for d, k in line.aliases.items() if k in source)),
                context_events=tuple(by_id[eid] for eid in segment.context_event_ids if eid in by_id),
                previous_segment_id=(
                    segments[index - 1].segment_id
                    if index and segments[index - 1].boundary_reason == "topic"
                    else ""
                ),
            )
        )
    return inputs
