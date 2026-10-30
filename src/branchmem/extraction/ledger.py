"""Extraction job ledger: ``pending -> leased -> done | skipped | failed``.

There is no "last 24 hours only" eligibility window: under such a rule a job that
failed or waited too long silently loses eligibility forever.

Only non-done states are stored. "done" is proven by the product itself (the note
store has current notes for that segment and version), so there is no second truth.
No entry means pending.

- ``leased``: taken by the current run; an expired lease returns to pending after a
  crash, so nothing gets stuck;
- ``failed``: retried with exponential backoff; after ``max_attempts`` it is no longer
  retried automatically and is listed as ``exhausted`` in ``summary()`` for a human;
- ``skipped``: explicitly not applicable, with a reason.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .._util import atomic_json_dump, json_load

LEDGER_VERSION = 1

STATE_LEASED = "leased"
STATE_FAILED = "failed"
STATE_SKIPPED = "skipped"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEASE_SECONDS = 2 * 3600
# for a nightly run: retry the next night, then skip one night, and so on
DEFAULT_BACKOFF_SECONDS = 20 * 3600


class JobLedger:
    """Jobs of one store; ``path=None`` keeps it in memory only."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        extractor_version: str = "",
    ) -> None:
        self.path = Path(path) if path else None
        self.max_attempts = max(1, int(max_attempts))
        self.lease_seconds = float(lease_seconds)
        self.backoff_seconds = float(backoff_seconds)
        self.extractor_version = str(extractor_version or "")
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = self._load()

    # ---- eligibility -------------------------------------------------------
    def is_eligible(self, job_id: str, *, now: Optional[float] = None) -> bool:
        stamp = time.time() if now is None else float(now)
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            return True
        state = job.get("state")
        if state == STATE_LEASED:
            return float(job.get("lease_expires_at") or 0.0) <= stamp
        if state == STATE_FAILED:
            if int(job.get("attempt_count") or 0) >= self.max_attempts:
                return False
            return float(job.get("retry_after") or 0.0) <= stamp
        return False  # skipped: explicitly not applicable, never taken automatically

    # ---- transitions -------------------------------------------------------
    def lease(self, job_ids: Iterable[str], *, now: Optional[float] = None) -> None:
        stamp = time.time() if now is None else float(now)
        with self._lock:
            for job_id in job_ids:
                job = self._jobs.setdefault(str(job_id), {})
                job.update(
                    state=STATE_LEASED,
                    leased_at=stamp,
                    lease_expires_at=stamp + self.lease_seconds,
                    extractor_version=self.extractor_version,
                )

    def mark_done(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(str(job_id), None)

    def mark_failed(self, job_id: str, error: str, *, now: Optional[float] = None) -> None:
        stamp = time.time() if now is None else float(now)
        with self._lock:
            job = self._jobs.setdefault(str(job_id), {})
            attempts = int(job.get("attempt_count") or 0) + 1
            job.update(
                state=STATE_FAILED,
                attempt_count=attempts,
                last_error=str(error or "")[:300],
                failed_at=stamp,
                retry_after=stamp + self.backoff_seconds * (2 ** (attempts - 1)),
                extractor_version=self.extractor_version,
            )
            job.pop("lease_expires_at", None)

    def mark_skipped(self, job_id: str, reason: str, *, now: Optional[float] = None) -> None:
        stamp = time.time() if now is None else float(now)
        with self._lock:
            self._jobs[str(job_id)] = {
                "state": STATE_SKIPPED,
                "skip_reason": str(reason or ""),
                "skipped_at": stamp,
                "extractor_version": self.extractor_version,
            }

    def release(self, job_ids: Iterable[str]) -> None:
        """Jobs not reached within this run's budget go back to pending, same order next run."""
        with self._lock:
            for job_id in job_ids:
                job = self._jobs.get(str(job_id))
                if job is None or job.get("state") != STATE_LEASED:
                    continue
                if int(job.get("attempt_count") or 0) > 0:
                    job["state"] = STATE_FAILED
                    job.pop("lease_expires_at", None)
                else:
                    self._jobs.pop(str(job_id), None)

    def forget_missing(self, live_ids: Iterable[str]) -> int:
        """Drop entries whose job no longer exists, so the ledger does not grow without bound."""
        live = {str(item) for item in live_ids}
        with self._lock:
            stale = [job_id for job_id in self._jobs if job_id not in live]
            for job_id in stale:
                self._jobs.pop(job_id, None)
        return len(stale)

    def forget_other_versions(self) -> int:
        """After an extractor version change, old skipped / failed entries are re-queued."""
        with self._lock:
            stale = [
                job_id
                for job_id, job in self._jobs.items()
                if job.get("state") != STATE_LEASED
                and str(job.get("extractor_version") or "") != self.extractor_version
            ]
            for job_id in stale:
                self._jobs.pop(job_id, None)
        return len(stale)

    # ---- observability -----------------------------------------------------
    def state_of(self, job_id: str) -> str:
        with self._lock:
            job = self._jobs.get(str(job_id))
        return str(job.get("state")) if job else "pending"

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        skip_reasons: Dict[str, int] = {}
        exhausted = 0
        with self._lock:
            for job in self._jobs.values():
                state = str(job.get("state") or "")
                counts[state] = counts.get(state, 0) + 1
                if state == STATE_SKIPPED:
                    reason = str(job.get("skip_reason") or "")
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                if state == STATE_FAILED and int(job.get("attempt_count") or 0) >= self.max_attempts:
                    exhausted += 1
        return {"states": counts, "skip_reasons": skip_reasons, "exhausted": exhausted}

    # ---- persistence -------------------------------------------------------
    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = {
                "version": LEDGER_VERSION,
                "updated_at": time.time(),
                "jobs": {key: dict(value) for key, value in sorted(self._jobs.items())},
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(payload, self.path)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return {}
        raw = json_load(self.path, default={})
        jobs = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(jobs, dict):
            return {}
        return {str(key): dict(value) for key, value in jobs.items() if isinstance(value, dict)}


