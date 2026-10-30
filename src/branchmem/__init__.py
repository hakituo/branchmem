"""branchmem: branch-aware visibility for derived memory.

Conversation and agent histories are trees (regenerate, edit, fork from a checkpoint).
A derived memory (summary, fact, profile entry, graph edge) is visible only when all
of its source events lie on the current branch's ancestor path.
"""

from .anchors import AnchorStore, bound_anchors, get_request_anchors, set_request_anchors
from .cursors import CursorStore
from .events import NormalizedEvent, enrich_payload, normalize_event, normalize_payload
from .registry import TreeRegistry
from .tree import TreeBuilder, TreeBuildResult, build_tree
from .visibility import (
    MODE_COMPAT,
    MODE_STRICT,
    DerivedGate,
    build_read_context,
    event_visibility_reason,
    filter_derived,
    filter_raw_events,
    note_visibility_reason,
)

__version__ = "0.1.0"

__all__ = [
    "MODE_COMPAT",
    "MODE_STRICT",
    "AnchorStore",
    "CursorStore",
    "DerivedGate",
    "NormalizedEvent",
    "TreeBuildResult",
    "TreeBuilder",
    "TreeRegistry",
    "bound_anchors",
    "build_read_context",
    "build_tree",
    "enrich_payload",
    "event_visibility_reason",
    "filter_derived",
    "filter_raw_events",
    "get_request_anchors",
    "normalize_event",
    "normalize_payload",
    "note_visibility_reason",
    "set_request_anchors",
]
