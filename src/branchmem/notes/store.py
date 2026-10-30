"""Derived note store: one JSON file per conversation, replaced atomically per segment.

- ``note_id = hash(source_segment_id + extractor_version + kind + normalized text)``,
  so re-running never creates equivalent duplicates.
- Rewriting a segment moves the old version into that segment's ``history``; only one
  current version is kept, and old versions stay traceable.
- Evidence is checked before writing: ``source_event_ids`` must be a subset of the
  segment's sources. Drafts out of range are dropped and counted.
- ``has_segment`` only counts a segment done for that extractor version **and** with at
  least one current note; zero output is not "done".
- Every note keeps provenance (``source_event_ids``, ``source_segment_id``) and the
  bitemporal fields; this is what the branch gate reads, never a lineage label.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .._util import atomic_json_dump, json_load
from ..events import VISIBILITY_TREE_LOCAL

SCHEMA_VERSION = 1
_lock = threading.RLock()


@dataclass(frozen=True)
class ReplaceResult:
    written: int = 0
    rejected: int = 0  # dropped for out-of-range evidence or missing fields
    duplicate: int = 0  # merged with an equivalent draft in the same batch


def _norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def make_note_id(segment_id: str, extractor_version: str, kind: str, text: str) -> str:
    raw = "\x1f".join([str(segment_id), str(extractor_version), str(kind), _norm_text(text)])
    return "note_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def notes_file_name(conversation_id: str) -> str:
    raw = str(conversation_id or "")
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw)[:120] or "default"
    if safe != raw:  # sanitized names get a hash of the original to avoid collisions
        safe += "-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe}.json"


def _field(draft: Any, name: str, default: Any = None) -> Any:
    if isinstance(draft, Mapping):
        return draft.get(name, default)
    return getattr(draft, name, default)


def _draft_time(draft: Any, key: str, default: Optional[float]) -> Optional[float]:
    metadata = _field(draft, "metadata", None)
    value = metadata.get(key) if isinstance(metadata, Mapping) else None
    try:
        return float(value) if value else default
    except (TypeError, ValueError):
        return default


class NoteStore:
    """Notes of one namespace under ``root``.

    ``extra_fields`` lists draft ``metadata`` keys copied onto the stored note (for
    application-specific signals).
    """

    def __init__(
        self,
        root: Path,
        *,
        namespace: str = "default",
        perspective_role: Optional[str] = None,
        extra_fields: Iterable[str] = (),
    ) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.perspective_role = perspective_role or namespace
        self.extra_fields = tuple(extra_fields)

    def path(self, conversation_id: str) -> Path:
        return self.root / notes_file_name(conversation_id)

    # ---- reads -------------------------------------------------------------
    def _load(self, conversation_id: str) -> Dict[str, Any]:
        data = json_load(self.path(conversation_id), default=None)
        if not isinstance(data, dict) or not isinstance(data.get("segments"), dict):
            return {"version": SCHEMA_VERSION, "conversation_id": conversation_id, "segments": {}}
        return data

    def list_conversations(self) -> List[str]:
        """Conversation IDs with a notes file (original values, not file names)."""
        found: List[str] = []
        for file_path in sorted(self.root.glob("*.json")) if self.root.exists() else ():
            data = json_load(file_path, default=None)
            if isinstance(data, dict) and data.get("conversation_id"):
                found.append(str(data["conversation_id"]))
        return found

    def stamp(self, conversation_id: str) -> float:
        try:
            return self.path(conversation_id).stat().st_mtime
        except OSError:
            return -1.0

    def has_segment(self, conversation_id: str, segment_id: str, extractor_version: str) -> bool:
        entry = self._load(conversation_id)["segments"].get(segment_id)
        return bool(entry and entry.get("extractor_version") == extractor_version and entry.get("notes"))

    def list_notes(self, conversation_id: str) -> List[Dict[str, Any]]:
        notes: List[Dict[str, Any]] = []
        for entry in self._load(conversation_id)["segments"].values():
            notes.extend(dict(n) for n in entry.get("notes") or [])
        return notes

    def segment_entries(self, conversation_id: str) -> Dict[str, Dict[str, Any]]:
        return dict(self._load(conversation_id)["segments"])

    def get_carry(self, conversation_id: str, segment_id: str) -> List[str]:
        """Exchanges this segment handed to the next one (event IDs)."""
        entry = self._load(conversation_id)["segments"].get(segment_id) or {}
        return [str(e) for e in entry.get("carry_event_ids") or []]

    def segment_history(self, conversation_id: str, segment_id: str) -> List[Dict[str, Any]]:
        entry = self._load(conversation_id)["segments"].get(segment_id) or {}
        return [dict(h) for h in entry.get("history") or []]

    # ---- writes ------------------------------------------------------------
    def replace_segment(
        self,
        conversation_id: str,
        segment_id: str,
        extractor_version: str,
        drafts: Iterable[Any],
        *,
        segment_source_event_ids: Iterable[str],
        occurred_from: Optional[float] = None,
        occurred_to: Optional[float] = None,
        carry_event_ids: Iterable[str] = (),
    ) -> ReplaceResult:
        """Replace the segment's current notes with ``drafts``; evidence must stay in range."""
        allowed = {str(e) for e in segment_source_event_ids}
        now = time.time()
        notes: Dict[str, Dict[str, Any]] = {}
        rejected = duplicate = 0
        for draft in drafts:
            kind = str(_field(draft, "kind", "") or "")
            text = _norm_text(_field(draft, "text", ""))
            evidence = tuple(str(e) for e in (_field(draft, "source_event_ids", ()) or ()))
            if not kind or not text or not evidence or not set(evidence) <= allowed:
                rejected += 1
                continue
            try:
                importance = float(_field(draft, "importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            note_id = make_note_id(segment_id, extractor_version, kind, text)
            if note_id in notes:
                duplicate += 1
                continue
            metadata = _field(draft, "metadata", None)
            metadata = metadata if isinstance(metadata, Mapping) else {}
            notes[note_id] = {
                "note_id": note_id,
                "namespace": self.namespace,
                "perspective_role": self.perspective_role,
                "conversation_id": conversation_id,
                "visibility": VISIBILITY_TREE_LOCAL,
                "kind": kind,
                "text": text,
                "source_event_ids": list(evidence),
                "source_segment_id": segment_id,
                "extractor_version": extractor_version,
                "recorded_from": now,
                "recorded_to": None,
                # when the underlying conversation happened (not the same as valid_*)
                "occurred_from": _draft_time(draft, "occurred_from", occurred_from),
                "occurred_to": _draft_time(draft, "occurred_to", occurred_to),
                "valid_from": None,
                "valid_to": _draft_time(draft, "valid_to", None),
                "superseded_by": None,
                "importance": importance,
                **{key: metadata[key] for key in self.extra_fields if key in metadata},
            }
        with _lock:
            data = self._load(conversation_id)
            old = data["segments"].get(segment_id)
            history = list((old or {}).get("history") or [])
            if old and old.get("notes"):
                history.append(
                    {"extractor_version": old.get("extractor_version"), "replaced_at": now, "notes": old["notes"]}
                )
            data["segments"][segment_id] = {
                "extractor_version": extractor_version,
                "recorded_at": now,
                "notes": list(notes.values()),
                "history": history,
                "source_event_ids": sorted(allowed),
                **({"carry_event_ids": list(carry_event_ids)} if carry_event_ids else {}),
            }
            atomic_json_dump(data, self.path(conversation_id))
        return ReplaceResult(written=len(notes), rejected=rejected, duplicate=duplicate)

    def patch_notes(self, conversation_id: str, patch: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> int:
        """Update metadata of current notes in place (never ``note_id`` or ``text``)."""
        changed = 0
        with _lock:
            data = self._load(conversation_id)
            for entry in data["segments"].values():
                for note in entry.get("notes") or []:
                    fields = {k: v for k, v in (patch(dict(note)) or {}).items() if k not in {"note_id", "text"}}
                    if any(note.get(k) != v for k, v in fields.items()):
                        note.update(fields)
                        changed += 1
            if changed:
                atomic_json_dump(data, self.path(conversation_id))
        return changed

    def drop_segment(self, conversation_id: str, segment_id: str) -> bool:
        """A late event invalidated the segment: notes move to history, segment not done."""
        with _lock:
            data = self._load(conversation_id)
            entry = data["segments"].get(segment_id)
            if not entry:
                return False
            if entry.get("notes"):
                entry.setdefault("history", []).append(
                    {
                        "extractor_version": entry.get("extractor_version"),
                        "replaced_at": time.time(),
                        "notes": entry["notes"],
                        "dropped": True,
                    }
                )
            entry["notes"] = []
            atomic_json_dump(data, self.path(conversation_id))
        return True
