"""Shared contracts of per-segment extraction.

An extractor turns one ``SegmentInput`` (events of one segment on one branch path)
into ``NoteDraft``s, each citing the events it is based on. The pipeline checks that
evidence stays inside the segment before anything is stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..segments.v1 import Segment

SKIP_NO_REUSABLE_INFO = "no_reusable_info"
SKIP_CONTENT_REJECTED = "content_rejected"
#: the whole segment is an open topic handed to the next segment (processed, no output)
SKIP_CARRIED_FORWARD = "carried_forward"
#: zero output without a reason: the runner fills this in, never an unexplained empty result
SKIP_UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class SegmentInput:
    """One segment to extract.

    ``events`` are the segment's payloads, already restricted to the current branch,
    de-duplicated and content-gated, in time order. Only ``closed`` segments may be
    extracted.
    """

    namespace: str
    conversation_id: str
    segment: Segment
    events: tuple
    closed: bool
    #: merged repeated user submissions ``(dropped event_id, kept event_id)``
    aliases: tuple = ()
    #: segmenter v2: previous segment's tail, read-only context (never a source)
    context_events: tuple = ()
    #: relay: the previous segment of the same session (only after a topic cut)
    previous_segment_id: str = ""
    #: relay: exchanges handed over by the previous segment (prepended; they are sources)
    carried_event_ids: tuple = ()

    @property
    def segment_id(self) -> str:
        return self.segment.segment_id

    @property
    def continues(self) -> bool:
        """The same session continues after this segment (it ended at a topic cut)."""
        return self.segment.boundary_reason == "topic"

    @property
    def allowed_sources(self) -> tuple:
        """What evidence may cite: the segment's sources plus relayed exchanges."""
        return (*self.segment.source_event_ids, *self.carried_event_ids)

    @property
    def char_count(self) -> int:
        return int(self.segment.char_count or 0)


@dataclass(frozen=True)
class NoteDraft:
    kind: str  # "episode" | "fact" | "preference" | "plan" | ...
    text: str
    source_event_ids: tuple  # must be a subset of the segment's allowed sources
    importance: float = 0.5
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    notes: tuple  # tuple[NoteDraft, ...]; may be empty
    skip_reason: str = ""  # required when there is no output
    extractor_version: str = ""
    #: relay: exchanges of an open last topic handed to the next segment
    carry_event_ids: tuple = ()


@runtime_checkable
class Extractor(Protocol):
    version: str

    async def extract(self, item: SegmentInput) -> ExtractionResult: ...
