"""Visibility: the read context, the shared predicate and the gates built on it."""

from .anchor_resolution import resolve_gate_anchors
from .context import (
    MODE_COMPAT,
    MODE_STRICT,
    MemoryReadContext,
    build_read_context,
    event_visibility_reason,
    filter_visible_events,
    filter_visible_notes,
    note_visibility_reason,
)
from .derived_gate import DerivedGate, filter_derived
from .raw_gate import COMPAT_PASS_REASONS, GateOutcome, filter_raw_events, load_trees

__all__ = [
    "COMPAT_PASS_REASONS",
    "MODE_COMPAT",
    "MODE_STRICT",
    "DerivedGate",
    "GateOutcome",
    "MemoryReadContext",
    "build_read_context",
    "event_visibility_reason",
    "filter_derived",
    "filter_raw_events",
    "filter_visible_events",
    "filter_visible_notes",
    "load_trees",
    "note_visibility_reason",
    "resolve_gate_anchors",
]
