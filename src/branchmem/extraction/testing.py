"""A fake extractor for tests and demos; calls no model."""

from __future__ import annotations

from typing import Callable, List, Optional

from .contracts import SKIP_NO_REUSABLE_INFO, ExtractionResult, NoteDraft, SegmentInput

Predicate = Optional[Callable[[SegmentInput], bool]]


class FakeExtractor:
    """One episode per segment citing its first event; can fail, return nothing, or
    cite out-of-range evidence. ``calls`` records extracted segment IDs.

    ``identity=True`` writes one note per event instead, with the event's own text:
    the most leak-prone extraction possible, handy for visibility demos.
    """

    def __init__(
        self,
        version: str = "fake-1",
        *,
        identity: bool = False,
        fail_on: Predicate = None,
        empty_on: Predicate = None,
        out_of_range_on: Predicate = None,
    ) -> None:
        self.version = version
        self.identity = identity
        self.calls: List[str] = []
        self._fail_on = fail_on
        self._empty_on = empty_on
        self._out_of_range_on = out_of_range_on

    async def extract(self, item: SegmentInput) -> ExtractionResult:
        self.calls.append(item.segment_id)
        if self._fail_on and self._fail_on(item):
            raise ValueError("fake_parse_error")
        if self._empty_on and self._empty_on(item):
            return ExtractionResult(notes=(), skip_reason=SKIP_NO_REUSABLE_INFO, extractor_version=self.version)
        if self.identity:
            drafts = tuple(
                NoteDraft(kind="fact", text=str(e.get("content") or ""), source_event_ids=(str(e["event_id"]),))
                for e in item.events
                if str(e.get("content") or "").strip()
            )
            return ExtractionResult(notes=drafts, extractor_version=self.version)
        evidence = (item.segment.source_event_ids[0],)
        if self._out_of_range_on and self._out_of_range_on(item):
            evidence = ("evt-not-in-segment",)
        draft = NoteDraft(
            kind="episode",
            text=f"segment {item.segment_id} with {len(item.segment.turn_ids)} turns",
            source_event_ids=evidence,
        )
        return ExtractionResult(notes=(draft,), extractor_version=self.version)
