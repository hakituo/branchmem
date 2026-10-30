"""Run one budgeted pass of per-segment extraction.

Every closed segment ends up ``done``, ``skipped`` (with a reason), ``failed``
(retryable, with an error type) or still ``pending``; nothing is silently lost.

- Oldest first, by ``(segment start, segment_id)``; budgets on segment count, characters
  and wall clock. The first segment is always allowed, so a huge segment cannot starve.
- ``done`` only after the notes are written and visible in the store.
- Drafts whose evidence leaves the segment are dropped; if none remain the job fails.
- Segments of one session run in order (relay); different sessions are independent.
- Only error *types* are logged: messages may contain conversation text.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from . import relay
from .contracts import SKIP_UNSPECIFIED, Extractor, SegmentInput
from .ledger import JobLedger

DONE, SKIPPED, FAILED = "done", "skipped", "failed"


@dataclass(frozen=True)
class Budget:
    max_segments: int = 0  # 0 = unlimited
    max_chars: int = 0
    max_seconds: float = 0.0


def apply_budget(pending: Sequence[SegmentInput], budget: Budget) -> List[SegmentInput]:
    picked: List[SegmentInput] = []
    used = 0
    for item in pending:
        if budget.max_segments and len(picked) >= budget.max_segments:
            break
        if budget.max_chars and picked and used + item.char_count > budget.max_chars:
            break
        picked.append(item)
        used += item.char_count
    return picked


def accepted_versions(extractor: Any) -> Set[str]:
    """The current version plus older versions the extractor still accepts as done."""
    return {str(extractor.version), *(str(v) for v in getattr(extractor, "accepted_versions", ()))}


def _done_segments(store: Any, conversation_ids: Iterable[str], versions: Set[str]) -> Dict[str, Set[str]]:
    return {
        cid: {
            str(note.get("source_segment_id") or "")
            for note in store.list_notes(cid)
            if str(note.get("extractor_version") or "") in versions
        }
        for cid in conversation_ids
    }


def _evidence_ok(draft: Any, allowed: Set[str]) -> bool:
    evidence = {str(item) for item in (getattr(draft, "source_event_ids", ()) or ())}
    return bool(evidence) and evidence <= allowed


async def _process(item, extractor, store, ledger, report, now, items_by_id) -> str:
    sid = item.segment_id
    item = relay.with_carry(item, store, items_by_id)
    try:
        result = await extractor.extract(item)
    except Exception as exc:  # noqa: BLE001 - parse / request failures are retryable
        ledger.mark_failed(sid, type(exc).__name__, now=now)
        return FAILED
    drafts = tuple(result.notes or ())
    carry = tuple(getattr(result, "carry_event_ids", ()) or ())
    if not drafts:
        if carry:  # the whole segment was handed to the next one
            await asyncio.to_thread(
                store.replace_segment, item.conversation_id, sid, extractor.version, (),
                segment_source_event_ids=item.allowed_sources, carry_event_ids=carry,
            )
        ledger.mark_skipped(sid, result.skip_reason or SKIP_UNSPECIFIED, now=now)
        return SKIPPED
    allowed = set(item.allowed_sources)
    valid = tuple(draft for draft in drafts if _evidence_ok(draft, allowed))
    report["notes_rejected"] += len(drafts) - len(valid)
    if not valid:
        ledger.mark_failed(sid, "evidence_out_of_range", now=now)
        return FAILED
    written = await asyncio.to_thread(
        store.replace_segment, item.conversation_id, sid, extractor.version, valid,
        segment_source_event_ids=item.allowed_sources,
        occurred_from=item.segment.started_at, occurred_to=item.segment.ended_at,
        carry_event_ids=carry,
    )
    report["notes_written"] += int(getattr(written, "written", 0) or 0)
    if await asyncio.to_thread(store.has_segment, item.conversation_id, sid, extractor.version):
        ledger.mark_done(sid)
        return DONE
    ledger.mark_failed(sid, "write_not_visible", now=now)
    return FAILED


async def run_extraction(
    segments: Sequence[SegmentInput],
    *,
    extractor: Extractor,
    store: Any,
    ledger: Optional[JobLedger] = None,
    budget: Budget = Budget(),
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract pending closed segments; returns a report with counts only."""
    stamp = time.time() if now is None else float(now)
    ledger = ledger or JobLedger(None, extractor_version=extractor.version)
    closed = [item for item in segments if item.closed]
    ledger.forget_other_versions()
    ledger.forget_missing(item.segment_id for item in closed)
    done = _done_segments(store, sorted({i.conversation_id for i in closed}), accepted_versions(extractor))
    pending = sorted(
        (i for i in closed if i.segment_id not in done.get(i.conversation_id, set())
         and ledger.is_eligible(i.segment_id, now=stamp)),
        key=lambda i: (i.segment.started_at, i.segment_id),
    )
    picked = apply_budget(pending, budget)
    report: Dict[str, Any] = {
        "extractor_version": extractor.version, "segments_closed": len(closed),
        "pending": len(pending), "picked": len(picked), "deferred": len(pending) - len(picked),
        DONE: 0, SKIPPED: 0, FAILED: 0, "notes_written": 0, "notes_rejected": 0,
    }
    ids = [item.segment_id for item in picked]
    ledger.lease(ids, now=stamp)
    ledger.save()
    deadline = time.monotonic() + budget.max_seconds if budget.max_seconds > 0 else None
    items_by_id: Mapping[str, SegmentInput] = {item.segment_id: item for item in closed}
    pending_ids = {item.segment_id for item in pending}
    outcomes: Dict[str, str] = {}
    try:
        for item in picked:
            if deadline is not None and time.monotonic() >= deadline:
                break
            previous = item.previous_segment_id
            if previous in pending_ids and outcomes.get(previous) not in (DONE, SKIPPED):
                continue  # relay: wait until the previous segment of the session is processed
            outcome = await _process(item, extractor, store, ledger, report, stamp, items_by_id)
            outcomes[item.segment_id] = outcome
            report[outcome] += 1
    finally:
        untouched = [sid for sid in ids if sid not in outcomes]
        ledger.release(untouched)
        ledger.save()
        report["released"] = len(untouched)
        report["ledger"] = ledger.summary()
    return report
