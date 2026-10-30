"""Content gating: which text may feed extraction, independent of topology.

Blank text, media placeholders, system / scheduler events and inner monologue can be
``content_eligible=False``, but the event and its parent pointer are always kept.
Filtering what gets summarized must never cut a branch apart.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Optional

#: Structural event types: kept in the tree, never used as extraction input.
TOPOLOGY_ONLY_EVENT_TYPES = frozenset({"chat_thought", "system_prompt", "scheduled_task"})

#: Pure media placeholders such as "[image]", "[图片]", "[IMG:1]".
_MEDIA_PLACEHOLDER = re.compile(
    r"^\s*(?:\[(?:图片|照片|视频|语音|表情|文件|贴纸|IMG|BM|VOICE|FILE|STICKER|image|photo|video|voice|file|sticker)"
    r"(?:[:：][^\]]{0,32})?\]\s*)+$",
    re.IGNORECASE,
)

#: Optional application hook: return True to exclude a text (e.g. debug markers).
ContentFilter = Callable[[str], bool]
_extra_filter: Optional[ContentFilter] = None


def set_content_filter(func: Optional[ContentFilter]) -> None:
    """Install an extra exclusion rule; ``None`` removes it."""
    global _extra_filter
    _extra_filter = func


def content_hash(content: str) -> str:
    normalized = " ".join(str(content or "").strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def is_media_placeholder(content: str) -> bool:
    return bool(_MEDIA_PLACEHOLDER.match(str(content or "").strip()))


def content_gate_reason(content: str, *, event_type: str = "") -> Optional[str]:
    """Why a text may not feed extraction; ``None`` when it may."""
    raw = str(content or "").strip()
    if not raw:
        return "empty"
    if str(event_type or "").strip().lower() in TOPOLOGY_ONLY_EVENT_TYPES:
        return "topology_only_event_type"
    if _extra_filter is not None and _extra_filter(raw):
        return "custom_filter"
    if is_media_placeholder(raw):
        return "media_placeholder"
    return None


def is_content_eligible(content: str, *, event_type: str = "") -> bool:
    return content_gate_reason(content, event_type=event_type) is None
