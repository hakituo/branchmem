"""Deterministic segmentation along one branch path (v1 by turns, v2 by topic)."""

from .topic import SEGMENTER_V2, TopicParams, TopicSegmentation, build_topic_segments
from .v1 import SEGMENTER_VERSION, Segment, SegmentDiff, build_segments, diff_segments

__all__ = [
    "SEGMENTER_V2",
    "SEGMENTER_VERSION",
    "Segment",
    "SegmentDiff",
    "TopicParams",
    "TopicSegmentation",
    "build_segments",
    "build_topic_segments",
    "diff_segments",
]
