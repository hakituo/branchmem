"""Relay splitting: when a long conversation is cut mid-topic, hand the open topic forward.

- Only a segment followed by more of the same session (``SegmentInput.continues``) may
  hand off: the extractor marks its last topic as open, and that topic must reach the
  segment's last exchange; the segment then does not summarize it.
- At most ``CARRY_MAX_CHARS`` characters are handed off; anything longer is summarized
  in place, so carries never snowball.
- The next segment prepends the handed-off exchanges and counts them as its own sources
  (``SegmentInput.allowed_sources``).
- The number of model calls is unchanged; segments of one session just run in order,
  sessions still run in parallel. Handoffs are stored in the note store
  (``carry_event_ids``), so they survive between runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contracts import SegmentInput

CARRY_MAX_CHARS = 3000


def with_carry(item: SegmentInput, store: Any, items_by_id: Mapping[str, SegmentInput]) -> SegmentInput:
    """Prepend exchanges handed over by the previous segment; unchanged when there are none."""
    previous = items_by_id.get(item.previous_segment_id) if item.previous_segment_id else None
    if previous is None:
        return item
    carry = set(store.get_carry(item.conversation_id, previous.segment_id))
    if not carry:
        return item
    carried = tuple(event for event in previous.events if str(event.get("event_id")) in carry)
    if not carried:
        return item
    carried_ids = tuple(str(event.get("event_id")) for event in carried)
    return dataclasses.replace(
        item,
        events=(*carried, *item.events),
        carried_event_ids=carried_ids,
        context_events=tuple(
            e for e in item.context_events if str(e.get("event_id")) not in set(carried_ids)
        ),
    )


def carry_start(
    open_ids: Iterable[int], exchanges: Sequence[Sequence[Mapping[str, Any]]]
) -> int | None:
    """First exchange index of the open topic to hand off; ``None`` if it does not reach the
    end or is too long."""
    ids = sorted(set(open_ids))
    last = len(exchanges) - 1
    if not ids or ids[-1] != last or ids[0] < 0:
        return None  # must reach the last exchange; a fully open segment is handed off whole
    chars = sum(len(str(e.get("content") or "")) for ex in exchanges[ids[0] :] for e in ex)
    return ids[0] if chars <= CARRY_MAX_CHARS else None


def ordered_chains(items: Sequence[SegmentInput]) -> list[list[SegmentInput]]:
    """Chain segments previous -> next: run a chain in order, chains in parallel."""
    present = {item.segment_id: item for item in items}
    chains: dict[str, list[SegmentInput]] = {}
    head_of: dict[str, str] = {}
    for item in sorted(items, key=lambda i: (i.segment.started_at, i.segment_id)):
        head = head_of.get(item.previous_segment_id) if item.previous_segment_id in present else None
        head = head or item.segment_id
        head_of[item.segment_id] = head
        chains.setdefault(head, []).append(item)
    return list(chains.values())


__all__ = ["CARRY_MAX_CHARS", "carry_start", "ordered_chains", "with_carry"]
