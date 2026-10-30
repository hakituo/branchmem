# Invariants, contracts and reason codes

## Terms

| term | meaning |
|---|---|
| namespace | sharing / ACL boundary ("whose memory"); checked before any branch logic |
| tree | one conversation's message tree, built from `parent` / `variant_of` pointers |
| continuity | conversations that form one narrative; must be declared, never inferred |
| anchor | the current node of a tree; the tree's visible set is the anchor and its ancestors |
| `anchors_by_tree` | the anchors of every tree a read may touch |
| segment | consecutive turns on **one** root-to-leaf path of **one** tree, used for extraction |
| note | a derived item (episode, fact, preference, plan, profile entry, edge) citing source events |

## Invariants

- **I1 Isolation before recall.** namespace, perspective and branch are hard filters
  applied before any ranking; probabilistic recall never weakens them.
- **I2 Raw history is never blamed.** Failures of summaries, indexes or profiles never
  touch raw events; derived data is versioned and rebuildable.
- **I3 Nothing is silently dropped.** Every input ends as done, explicitly skipped,
  explicitly failed or pending, with a reason.
- **I4 Deterministic topology.** The same event set yields the same tree, segments and
  source mapping in any arrival order.
- **I5 Siblings never see each other.** Each tree sees only its anchor and ancestors.
  Switching branches changes only that tree's anchor, never history.
- **I6 Sharing is explicit.** Sharing across conversations is not merging across
  branches; `namespace_shared` is a separate, auditable visibility class.

## Visibility predicate

```text
a_t = anchors_by_tree[t]
Visible(ctx) = union over allowed trees t of ({a_t} ∪ ancestors(a_t))
raw event e visible  <=> namespace, perspective, as-of pass, and e's node ∈ Visible(ctx)
derived note n visible <=> namespace, perspective, as-of (and continuity) pass, and
                           every source event of n is visible by the rule above
```

The filters run in a fixed order: namespace, perspective, continuity, ancestor path,
as-of. A failure at any step keeps the item out of every candidate list (vector, BM25,
graph seeds, every graph hop, final answer evidence).

| visibility | applies to | path rule |
|---|---|---|
| `tree_local` | raw and notes | sources on the anchor path of their tree |
| `continuity_local` | raw and notes | each source on its own tree's path, same continuity |
| `namespace_shared` | notes only | the note is exempt from tree / continuity; its sources are not |
| `legacy_unknown` | old raw / notes | readable only in compat mode, counted |

### Topology status

| status | raw read | extraction | shared promotion |
|---|---|---|---|
| `complete` | strict path | allowed on the same path | explicit rules only |
| `partial` | strict where resolvable; affected path fails closed | blocked until repaired | blocked |
| `unknown` | compat mode only, counted | not in strict pipelines | blocked |

## Anchor resolution (per tree)

1. explicit request anchors (`anchors=` or the `bound_anchors` context variable);
2. for the request's own conversation, the deepest node of the client-supplied
   active path (`active_path_ids`);
3. a persisted cursor, but only if its parent chain reaches a real root.

No anchor -> `anchor_missing` (hidden in strict mode).

## Tree building

- An event without `client_node_id` is `unknown`; it never becomes a new root.
- `variant_of` points to the **old sibling**. A variant with a parent attaches at once
  (the old version may never arrive); without a parent it waits and inherits the old
  version's parent. A conflicting parent is reported and the tree becomes `partial`.
- A child that arrives first waits in `pending` and is replayed in a stable order.
- Deduplication by `event_id` and by `idempotency_key` (conversation + actor + speaker
  + event type + client node ID, or content hash + second when there is no node).
- Dangling parents are repaired only on a copy of the result:
  `segment_alias` (`X-1`, `X-2` reply bubbles map to the assistant node `X`) and
  `cursor_graft` (an orphan root recorded under the cursor at write time). A real
  parent arriving later always wins.

## Cursors

- `advance()` after each write moves the conversation's cursor to the newest node and
  appends one audit line (`turn_written`, `branch_switch`, `orphan_grafted`,
  `anchor_held`).
- A new node whose chain ends at a node still waiting for its parent is not a branch
  switch; the orphan root is grafted under the old cursor, or the cursor is held.

## Segmentation

- v1: group by `turn_id`, cut before a new turn on idle gap, `max_turns` or
  `max_chars`. Never at a fixed clock hour, never because a parent is in the previous
  segment.
- v2: units are exchanges (consecutive user messages + following replies). Idle gaps
  are the only hard cut. A session over `session_max_chars` is split recursively at the
  exchange boundary with the highest topic-depth score, each side at least
  `min_part_chars`; scores are rounded to 3 decimals with stable tie-breaks. Segments
  after an in-session cut carry the previous tail as read-only context (never sources).
  Cuts of closed sessions are recorded in a ledger keyed by
  `sha256(segmenter version + tree + ordered event IDs)`.
- `segment_id = hash(segmenter version + tree + ordered source event IDs)`;
  `diff_segments` tells which derived products to invalidate after late events.

## Note contract

```text
note_id, namespace, perspective_role, conversation_id, visibility
kind, text, source_event_ids, source_segment_id, extractor_version
recorded_from, recorded_to, occurred_from, occurred_to, valid_from, valid_to
superseded_by, importance
```

`note_id = hash(segment + extractor version + kind + normalized text)`. Evidence must
be a subset of the segment's sources. Rewriting a segment moves the old version into
its history.

## Reason codes

| code | meaning |
|---|---|
| `namespace_mismatch` / `perspective_mismatch` / `continuity_mismatch` | wrong owner or narrative |
| `as_of_excluded` | recorded after the read's as-of time |
| `legacy_unknown` | no client node: topology unknown |
| `tree_unresolvable` | the event cannot be placed in a tree |
| `tree_not_allowed` | the tree is not in the read's allowed set |
| `anchor_missing` | the tree has no anchor |
| `topology_incomplete` | the node is still waiting for its parent |
| `not_on_anchor_path` | a sibling branch: always hidden |
| `source_<code>` | a note hidden because of one of its sources |
| `source_unresolvable` | a note with no resolvable sources |
| `conversation_unknown` | a derived record without a conversation |
| `gate_error` | topology evaluation failed (compat passes, strict hides) |

In compat mode, passes are counted as `compat_pass:<code>`.
