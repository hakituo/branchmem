# branchmem

Branch-aware visibility for derived memory in tree-shaped conversation and agent histories.

[中文说明](README.zh.md)

## The problem

Chat and agent histories are trees, not logs. A user regenerates a reply, edits an
earlier message, or an agent framework forks from a checkpoint, and every time the
old continuation becomes a sibling branch. Filtering the *raw* messages to the current
branch is easy. Long-term memory makes it hard: summaries, facts, profile entries and
graph edges were derived while the abandoned branch was still current, and they stay
in the store. Without a rule tying them back to the tree, the abandoned branch leaks
into later answers ("you said you were moving to Chengdu", after the user edited that
message into "Hangzhou").

The bundled benchmark shows this in its simplest form: a reader that filters raw
messages but keeps memory items created on arrival leaks on every leak probe.

| reader (context level, 5 scenarios, 10 probes) | leak rate | hit rate |
|---|---|---|
| `all_history`: every arrived message, in arrival order | 100% | 100% |
| `raw_filtered_notes_unfiltered`: raw filtered, derived memory kept | 100% | 100% |
| `source_gated`: memory visible only if all its sources are on the path | 0% | 100% |
| `path_only`: only raw messages on the current path | 0% | 100% |
| `branchmem`: this library end to end, strict mode | 0% | 100% |

## Core rule

A derived memory is visible only when **every one of its source events lies on the
ancestor path of the current anchor** of the tree those sources belong to.

Everything else follows from taking that rule seriously:

- **Siblings never see each other.** A branch sees its own anchor and the anchor's
  ancestors, nothing else. Shared prefixes are visible from every branch; sibling
  content and anything derived from it are not.
- **Judge by provenance, not by labels.** No `lineage_id`, `branch_id` or `inactive`
  flag decides visibility. Labels go stale (a summary of the shared prefix was tagged
  with whichever branch was current when it was written); source event IDs do not.
  One source on a sibling hides the whole note.
- **Switching back only moves the anchor.** Nothing is marked inactive, deleted or
  rewritten when the user switches branches. Going back to an old branch restores its
  memories automatically, because they were never changed.
- **A missing anchor fails closed.** A tree without an anchor returns nothing in strict
  mode; it never defaults to "the whole tree".
- **Topology is deterministic.** The same event set gives the same tree, the same
  segments and the same IDs in any arrival order (children before parents, duplicate
  deliveries, late events).
- **Content gating never cuts the tree.** Media placeholders, thoughts and system
  events can be excluded from extraction, but they stay in the tree so parent chains
  remain connected.
- **Shared memory is explicit.** `namespace_shared` exempts only the note itself;
  expanding its sources re-checks the raw path.
- **Nothing is silently lost.** Every extraction job ends as done, skipped (with a
  reason), failed (retryable) or pending.

[docs/invariants.md](docs/invariants.md) states the predicate, the data contracts
and the reason codes precisely.

## Install

Python 3.10+. The core library has no dependencies.

```bash
pip install -e .            # core + benchmark
pip install -e ".[notes]"   # + numpy for vector retrieval
pip install -e ".[zh]"      # + jieba for Chinese BM25 tokenization
pip install -e ".[test]"    # + pytest
```

Without installing, `PYTHONPATH=src:bench` works too.

## Quick start

```python
from branchmem.visibility import MODE_STRICT, DerivedGate, filter_raw_events

def msg(node, parent, text):
    return {"event_id": node, "conversation_id": "c", "role": "user", "content": text,
            "client_node_id": node, "parent_node_id": parent}

events = [msg("u1", "", "I have a cat called Mochi."),
          msg("u2", "u1", "Moving to Chengdu."),
          msg("u2b", "u1", "Moving to Hangzhou.")]          # the edit: a sibling of u2
notes = [{"id": "n1", "conversation_id": "c", "source_event_ids": ["u1"], "text": "cat: Mochi"},
         {"id": "n2", "conversation_id": "c", "source_event_ids": ["u2"], "text": "moving to Chengdu"}]

ctx = dict(conversation_id="c", active_path_ids=["u2b"], mode=MODE_STRICT,
           tree_reader=lambda _cid: events)
filter_raw_events(events, **ctx).events      # u1, u2b
DerivedGate(**ctx).filter(notes)             # n1 only; n2 is derived from the sibling
```

The anchor can come from the request (the client's active path, or explicit
`anchors_by_tree`), or from a persisted per-conversation cursor for background work
that has no request path (nightly extraction, proactive messages). See
[examples/](examples/):

| example | shows |
|---|---|
| `01_sibling_isolation.py` | an edited message, its notes, switching back |
| `02_store_and_cursors.py` | JSONL store -> resident trees -> cursors; a background read without a request path |
| `03_extract_and_retrieve.py` | extraction along the current branch, then gated BM25 retrieval |

## Layout

```text
src/branchmem/
  events.py  gating.py  identity.py     normalized events, content gate, namespace / continuity
  tree.py  dangling.py  registry.py     deterministic tree, dangling-parent repair, resident trees
  anchors.py  cursors.py                per-tree anchors, per-conversation cursors, orphan guard
  visibility/                           read context + predicate, raw gate, derived-record gate
  segments/                             segmenter v1 (turns), v2 (exchanges + topic depth), cut ledger
  sources/                              JSONL reference store, linear sources, overlays, regeneration inference
  notes/        (optional)              provenance-carrying note store, gated BM25 + vector retrieval
  extraction/   (optional)              per-segment extraction contracts, job ledger, runner, example LLM extractor
bench/
  branchbench/                          scenario format, validator, keyword scorer, context baselines
  scenarios/                            synthetic branching dialogues
tests/                                  core, notes, extraction, bench
examples/
```

The library never calls a model. The example LLM extractor takes your own
`complete(messages)` function.

## Strict and compat modes

`strict` hides anything that cannot be proven to be on the current branch. `compat`
exists for stores whose history predates topology: missing-information reasons
(`legacy_unknown`, `tree_unresolvable`, `anchor_missing`, `topology_incomplete`) are
let through **and counted**, but once an anchor is known a sibling is still hidden.
Counts from compat reads are not evidence of zero leakage; report them separately.

## Tests and benchmark

```bash
python -m pytest -q
python -m branchbench validate
python -m branchbench context-report
```

The validator checks that each probe tests what it claims: forbidden words exist only
off the anchor path and arrived before the question; answer words are on the path;
every scenario has a real fork. Scoring is keyword-based and fully reproducible, at
two levels: context (what the memory system hands to the answer model) and answer.

### Parity with the original implementation

This code was extracted from the memory system of a private application and cleaned
up for release. A differential check against the original modules, run on 400 random
trees with shuffled arrival, variants, missing parents, reply bubbles, duplicates and
legacy events, gave identical results for trees, ~107k visibility decisions, ~4.7k
read-gate calls, ~24k note decisions, segmentation (v1 and v2) and regeneration
inference.

### Results in the original deployment (aggregate only)

In the private deployment (data not released), historical client trees were matched
back to five conversations (about 23k events) with an overlay. In simulated requests,
about 10k items from abandoned branches were hidden, with 0 sibling leaks and 0
main-line items hidden by mistake. Inferring regenerations from repeated identical user
input alone reached 95.3% precision but only 53.5% recall against the real trees, which
is why the library uses inference for extraction de-duplication only, never to hide
content.

## Status

- [x] core library: normalization, deterministic trees, anchors and cursors,
      visibility predicate and gates, segmentation
- [x] optional note store, gated retrieval, extraction pipeline
- [x] benchmark format, validator, scorer, context-level baselines, 5 scenarios
- [ ] about 30 scenarios / 100 probes (templates + human review)
- [ ] answer-level baselines (e.g. Mem0-style ADD/UPDATE/DELETE, global expiry,
      lineage-label filtering), which need LLM calls
- [ ] invalidation (supersede / retract) as gated derived records

## License

Not decided yet.
