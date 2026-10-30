"""Optional layer: per-segment extraction along the current branch, with provenance.

Raw events -> main line of the current branch -> deterministic segments -> extractor
-> notes that cite their source events -> ``notes.NoteStore``.
"""

from .contracts import ExtractionResult, Extractor, NoteDraft, SegmentInput
from .inputs import SegmentParams, build_conversation_inputs
from .ledger import JobLedger
from .runner import Budget, run_extraction

__all__ = [
    "Budget",
    "ExtractionResult",
    "Extractor",
    "JobLedger",
    "NoteDraft",
    "SegmentInput",
    "SegmentParams",
    "build_conversation_inputs",
    "run_extraction",
]
