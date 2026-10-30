"""Namespace and continuity resolution.

``namespace`` is the sharing / ACL boundary ("whose memory is this") and is checked
before any branch filtering. ``continuity`` says which conversations belong to the
same narrative. Both are application policy, so they are pluggable:

- an explicit field on the event always wins;
- otherwise a resolver supplied by the caller is used;
- otherwise the safe default applies: the namespace is ``storage_scope`` (or
  ``"default"``), and every conversation is its own continuity. Sharing across
  conversations has to be declared, never guessed.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

from ._util import clean_text

DEFAULT_NAMESPACE = "default"

#: ``(conversation_id, storage_scope) -> namespace``
NamespaceResolver = Callable[[str, str], str]
#: ``(conversation_id, namespace) -> continuity_id``
ContinuityResolver = Callable[[str, str], str]


def resolve_namespace(
    conversation_id: str,
    storage_scope: str = "",
    *,
    resolver: Optional[NamespaceResolver] = None,
) -> str:
    if clean_text(storage_scope):
        return clean_text(storage_scope).lower()
    if resolver is not None:
        resolved = clean_text(resolver(clean_text(conversation_id), ""))
        if resolved:
            return resolved.lower()
    return DEFAULT_NAMESPACE


def resolve_continuity_id(
    conversation_id: str,
    namespace: str = "",
    *,
    resolver: Optional[ContinuityResolver] = None,
) -> str:
    cid = clean_text(conversation_id)
    if not cid:
        return "continuity:unknown"
    if resolver is not None:
        resolved = clean_text(resolver(cid, clean_text(namespace)))
        if resolved:
            return resolved
    return f"continuity:conversation:{cid}"


def mapping_resolver(groups: Dict[str, Iterable[str]]) -> ContinuityResolver:
    """Build a continuity resolver from an explicit ``continuity -> conversations`` map."""
    lookup = {
        clean_text(conversation): clean_text(continuity)
        for continuity, conversations in groups.items()
        for conversation in conversations
    }
    return lambda conversation_id, _namespace: lookup.get(conversation_id, "")


def continuity_mapping(
    conversation_ids: Iterable[str], *, resolver: Optional[ContinuityResolver] = None
) -> Dict[str, List[str]]:
    """Export ``continuity -> conversations`` so a human can review the grouping."""
    mapping: Dict[str, List[str]] = {}
    for conversation_id in conversation_ids:
        cid = clean_text(conversation_id)
        if cid:
            mapping.setdefault(resolve_continuity_id(cid, resolver=resolver), []).append(cid)
    return {key: sorted(values) for key, values in sorted(mapping.items())}
