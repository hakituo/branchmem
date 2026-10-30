"""Note retrieval: BM25 + vector RRF fusion, then the branch gate, then results with provenance.

Visibility: a note is returned only when **all** of its source events are on the
current branch of their tree (``DerivedGate``). One source on a sibling branch hides
the whole note. The gate runs after ranking but before the limit, so hidden notes
never take a slot and never reach the answer model.

Ranking components are reported separately (``rrf_score``, ``importance_adjust``)
instead of being folded into one unexplained score.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..visibility import MODE_COMPAT, DerivedGate
from .bm25 import BM25
from .store import NoteStore

#: RRF constant and candidates per route
RRF_K = 60
CANDIDATES_PER_ROUTE = 50
#: Additive importance nudge: weight x (importance - 0.5), importance clipped to [0, 1].
#: With RRF_K=60 this can reorder notes of similar relevance but never lifts a note found
#: by one route above a note found by both.
IMPORTANCE_WEIGHT = 0.001
NEUTRAL_IMPORTANCE = 0.5

_LOG = logging.getLogger("branchmem.notes")


@dataclass
class _Corpus:
    stamps: Dict[str, float] = field(default_factory=dict)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    bm25: Optional[BM25] = None


_cache: Dict[str, _Corpus] = {}
_cache_lock = threading.Lock()


def reset_note_retrieval_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _load_corpus(store: NoteStore) -> _Corpus:
    conversations = store.list_conversations()
    stamps = {cid: store.stamp(cid) for cid in conversations}
    key = str(store.root)
    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached.stamps == stamps:
            return cached
    notes = [n for cid in conversations for n in store.list_notes(cid) if str(n.get("text") or "").strip()]
    corpus = _Corpus(stamps=stamps, notes=notes, bm25=BM25([n["text"] for n in notes]))
    with _cache_lock:
        _cache[key] = corpus
    return corpus


def importance_adjustment(note: Mapping[str, Any]) -> float:
    try:
        importance = float(note.get("importance", NEUTRAL_IMPORTANCE))
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(importance):
        return 0.0
    return IMPORTANCE_WEIGHT * (min(max(importance, 0.0), 1.0) - NEUTRAL_IMPORTANCE)


class NoteRetriever:
    """``embedder`` is optional; without it retrieval is BM25 only and numpy is not needed.

    The gate settings (``registry`` / ``tree_reader`` / ``cursors`` / ``mode``) are the
    same as for ``DerivedGate``.
    """

    def __init__(
        self,
        store: NoteStore,
        *,
        embedder: Any = None,
        mode: str = MODE_COMPAT,
        registry: Any = None,
        tree_reader: Any = None,
        cursors: Any = None,
    ) -> None:
        self.store = store
        self._embedder = embedder
        self._gate_kwargs = {"mode": mode, "registry": registry, "tree_reader": tree_reader, "cursors": cursors}
        self.last_gate_reasons: Dict[str, int] = {}

    def _dense_ranking(self, corpus: _Corpus, query: str) -> List[int]:
        import numpy as np

        from . import vectors as vector_cache

        vectors: Dict[str, Any] = {}
        by_conversation: Dict[str, List[Dict[str, Any]]] = {}
        for note in corpus.notes:
            by_conversation.setdefault(str(note.get("conversation_id") or ""), []).append(note)
        budget = vector_cache.MAX_NEW_PER_CALL  # one budget shared by all conversations
        for cid, notes in by_conversation.items():
            found, computed = vector_cache.load_vectors(self.store, cid, notes, self._embedder, max_new=budget)
            vectors.update(found)
            budget = max(budget - computed, 0)
        query_vec = np.asarray(self._embedder([query])[0], dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec) or 1.0
        scored = [
            (float(vectors[note["note_id"]] @ query_vec), index)
            for index, note in enumerate(corpus.notes)
            if note["note_id"] in vectors
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [index for _score, index in scored[:CANDIDATES_PER_ROUTE]]

    def rank(self, query: str) -> Tuple[_Corpus, Dict[int, float], List[int]]:
        """Ungated ranking: ``(corpus, rrf scores, order)``."""
        corpus = _load_corpus(self.store)
        if not corpus.notes or not str(query or "").strip():
            return corpus, {}, []
        rankings = [corpus.bm25.rank(query, CANDIDATES_PER_ROUTE)] if corpus.bm25 else []
        if self._embedder is not None:
            try:
                rankings.append(self._dense_ranking(corpus, query))
            except Exception as exc:  # noqa: BLE001 - fall back to BM25 only
                _LOG.warning("vector ranking failed, using BM25 only: %s", type(exc).__name__)
        fused: Dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)
        adjusted = {i: fused[i] + importance_adjustment(corpus.notes[i]) for i in fused}
        order = sorted(adjusted, key=lambda i: (-adjusted[i], corpus.notes[i]["note_id"]))
        return corpus, fused, order

    def search(
        self,
        query: str,
        *,
        conversation_id: str = "",
        active_path_ids: Iterable[str] = (),
        anchors: Optional[Mapping[str, str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        corpus, fused, order = self.rank(query)
        gate = DerivedGate(
            conversation_id=conversation_id,
            active_path_ids=active_path_ids,
            anchors=anchors,
            **self._gate_kwargs,
        )
        hits = []
        for index in order:
            note = corpus.notes[index]
            if not gate.is_visible(note):
                continue
            score = fused[index] + importance_adjustment(note)
            hits.append(_hit(note, fused[index], score))
            if len(hits) >= limit:
                break
        self.last_gate_reasons = dict(sorted(gate.reasons.items()))
        return hits


def _hit(note: Mapping[str, Any], rrf: float, score: float) -> Dict[str, Any]:
    return {
        "note_id": note["note_id"],
        "kind": note.get("kind"),
        "text": note.get("text"),
        "conversation_id": note.get("conversation_id"),
        "occurred_from": note.get("occurred_from"),
        "occurred_to": note.get("occurred_to"),
        "source_event_ids": list(note.get("source_event_ids") or []),
        "source_segment_id": note.get("source_segment_id"),
        "importance": note.get("importance"),
        "valid_to": note.get("valid_to"),
        "score": round(score, 6),
        "rrf_score": round(rrf, 6),
        "importance_adjust": round(score - rrf, 6),
    }


__all__ = ["IMPORTANCE_WEIGHT", "NoteRetriever", "importance_adjustment", "reset_note_retrieval_cache"]
