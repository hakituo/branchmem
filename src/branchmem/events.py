"""Normalized event fields.

- ``event_id`` identifies a stored record (e.g. a random UUID at append time). It is
  not an idempotency key.
- ``idempotency_key`` deduplicates re-deliveries. It is scoped by conversation, actor,
  speaker and the client node ID.
- ``turn_id`` is an interaction unit. User, assistant and thought events may share a
  turn but each has its own node.
- ``content_eligible`` only says whether the text may feed extraction (see
  ``gating``). It never affects whether the event stays in the tree.

Everything here is a pure function: no files, no locks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ._util import as_float, clean_text
from .gating import content_hash, is_content_eligible
from .identity import (
    ContinuityResolver,
    NamespaceResolver,
    resolve_continuity_id,
    resolve_namespace,
)

SCHEMA_VERSION = "3.0"

TOPOLOGY_COMPLETE = "complete"
TOPOLOGY_PARTIAL = "partial"
TOPOLOGY_UNKNOWN = "unknown"

VISIBILITY_TREE_LOCAL = "tree_local"
VISIBILITY_CONTINUITY_LOCAL = "continuity_local"
VISIBILITY_NAMESPACE_SHARED = "namespace_shared"
VISIBILITY_LEGACY_UNKNOWN = "legacy_unknown"


@dataclass(frozen=True)
class NormalizedEvent:
    """A normalized view of one raw event. It carries the field contract, not a copy store."""

    event_id: str
    idempotency_key: str
    schema_version: str
    namespace: str
    perspective_role: str
    conversation_id: str
    tree_id: str
    continuity_id: str
    actor_id: str
    speaker_role: str
    turn_id: str
    event_type: str
    client_node_id: str
    parent_node_id: str
    variant_of_node_id: str
    content: str
    occurred_at: float
    recorded_at: float
    platform: str
    source_ref: Dict[str, Any] = field(default_factory=dict)
    topology_status: str = TOPOLOGY_UNKNOWN
    content_eligible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["source_ref"] = dict(self.source_ref)
        return data


def build_idempotency_key(
    *,
    conversation_id: str,
    actor_id: str,
    speaker_role: str,
    event_type: str,
    client_node_id: str = "",
    content: str = "",
    occurred_at: float = 0.0,
) -> str:
    """Re-delivery key.

    With a client node ID only stable identity is used (a re-delivered node may get a
    new timestamp). Without one, fall back to content hash plus second-level time.
    """
    parts = [
        SCHEMA_VERSION,
        clean_text(conversation_id),
        clean_text(actor_id),
        clean_text(speaker_role).lower(),
        clean_text(event_type).lower(),
    ]
    if clean_text(client_node_id):
        parts.append(f"node:{clean_text(client_node_id)}")
    else:
        parts.append(f"content:{content_hash(content)}")
        parts.append(f"ts:{int(as_float(occurred_at))}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def normalize_event(
    payload: Mapping[str, Any],
    *,
    storage_scope: str = "",
    perspective_role: str = "",
    recorded_at: float = 0.0,
    tree_id: str = "",
    namespace_resolver: Optional[NamespaceResolver] = None,
    continuity_resolver: Optional[ContinuityResolver] = None,
) -> NormalizedEvent:
    """Normalize one raw event (a stored JSONL line or a payload about to be written).

    Accepted aliases: ``metadata.client_message_id`` for ``client_node_id``,
    ``metadata.parent_id`` for ``parent_node_id``, ``metadata.variant_of`` for
    ``variant_of_node_id``, ``timestamp`` for ``occurred_at``.
    """
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    conversation_id = clean_text(payload.get("conversation_id")) or "default"
    role = clean_text(payload.get("role")).lower() or "system"
    event_type = clean_text(payload.get("event_type")) or "message"
    content = str(payload.get("content") or "")
    occurred_at = as_float(payload.get("timestamp") or payload.get("occurred_at"))

    client_node_id = clean_text(payload.get("client_node_id") or metadata.get("client_message_id"))
    parent_node_id = clean_text(payload.get("parent_node_id") or metadata.get("parent_id"))
    variant_of = clean_text(payload.get("variant_of_node_id") or metadata.get("variant_of"))
    turn_id = clean_text(
        payload.get("turn_id") or payload.get("message_id") or metadata.get("message_id")
    )
    platform = clean_text(payload.get("platform") or metadata.get("platform"))
    namespace = clean_text(payload.get("namespace")) or resolve_namespace(
        conversation_id, storage_scope, resolver=namespace_resolver
    )
    perspective = clean_text(payload.get("perspective_role")) or perspective_role or namespace

    owner_role = perspective if role in {"assistant", "system"} else "user"
    actor_id = clean_text(payload.get("actor_id")) or f"{namespace}:{owner_role}"

    if not client_node_id:
        topology_status = TOPOLOGY_UNKNOWN
    elif not parent_node_id:
        topology_status = TOPOLOGY_PARTIAL
    else:
        topology_status = TOPOLOGY_COMPLETE

    event_id = clean_text(payload.get("event_id"))
    idempotency_key = clean_text(payload.get("idempotency_key")) or build_idempotency_key(
        conversation_id=conversation_id,
        actor_id=actor_id,
        speaker_role=role,
        event_type=event_type,
        client_node_id=client_node_id,
        content=content,
        occurred_at=occurred_at,
    )

    return NormalizedEvent(
        event_id=event_id,
        idempotency_key=idempotency_key,
        schema_version=SCHEMA_VERSION,
        namespace=namespace,
        perspective_role=perspective,
        conversation_id=conversation_id,
        tree_id=clean_text(tree_id),
        continuity_id=clean_text(payload.get("continuity_id"))
        or resolve_continuity_id(conversation_id, namespace, resolver=continuity_resolver),
        actor_id=actor_id,
        speaker_role=role,
        turn_id=turn_id,
        event_type=event_type,
        client_node_id=client_node_id,
        parent_node_id=parent_node_id,
        variant_of_node_id=variant_of,
        content=content,
        occurred_at=occurred_at,
        recorded_at=as_float(payload.get("recorded_at")) or recorded_at,
        platform=platform,
        source_ref={"kind": "event", "event_id": event_id, "conversation_id": conversation_id},
        topology_status=topology_status,
        content_eligible=is_content_eligible(content, event_type=event_type),
    )


def normalize_payload(payload: Mapping[str, Any], **kwargs: Any) -> NormalizedEvent:
    """The normalization used by both read and write paths (``timestamp`` as recorded_at)."""
    return normalize_event(
        payload,
        storage_scope=str(payload.get("storage_scope") or ""),
        recorded_at=as_float(payload.get("timestamp")),
        **kwargs,
    )


_ENRICH_FIELDS = (
    "schema_version",
    "idempotency_key",
    "namespace",
    "continuity_id",
    "actor_id",
    "turn_id",
    "client_node_id",
    "parent_node_id",
    "variant_of_node_id",
    "occurred_at",
    "recorded_at",
    "platform",
    "topology_status",
    "content_eligible",
)


def enrich_payload(payload: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Normalized fields to merge into a payload before it is stored (originals untouched)."""
    event = normalize_event(payload, **kwargs)
    return {name: getattr(event, name) for name in _ENRICH_FIELDS}
