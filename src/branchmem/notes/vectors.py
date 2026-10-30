"""Side cache of note vectors: ``<notes root>/_vectors/<conversation>.npz``.

Computing vectors for thousands of notes inside one query is too slow, so vectors are
filled after writes and stored; a query only embeds the question. Missing vectors are
also filled at query time, up to a cap per call; the rest of that query uses BM25 only.
Deleting the cache recomputes everything. Requires numpy (``pip install branchmem[notes]``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .store import NoteStore

Embedder = Callable[[List[str]], Sequence[Sequence[float]]]

#: at most this many note vectors are computed during one query
MAX_NEW_PER_CALL = 64
_lock = threading.Lock()


def vectors_path(store: NoteStore, conversation_id: str) -> Path:
    return store.root / "_vectors" / (store.path(conversation_id).stem + ".npz")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _read(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            return {str(i): row.astype(np.float32) for i, row in zip(data["ids"], data["vectors"])}
    except Exception:  # noqa: BLE001 - a broken cache is treated as empty
        return {}


def _write(path: Path, vectors: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, ids=np.array(list(vectors), dtype=str),
             vectors=np.stack(list(vectors.values())).astype(np.float16))
    tmp.replace(path)


def load_vectors(
    store: NoteStore,
    conversation_id: str,
    notes: Sequence[Dict[str, Any]],
    embedder: Embedder,
    *,
    max_new: int = MAX_NEW_PER_CALL,
) -> Tuple[Dict[str, np.ndarray], int]:
    """``(note_id -> unit vector, number computed now)``; stale vectors are pruned."""
    path = vectors_path(store, conversation_id)
    with _lock:
        cached = _read(path)
        live = {str(note["note_id"]): str(note.get("text") or "") for note in notes}
        missing = [note_id for note_id in live if note_id not in cached][: max(max_new, 0)]
        stale = set(cached) - set(live)
        if missing:
            fresh = _normalize(np.asarray(embedder([live[i] for i in missing]), dtype=np.float32))
            cached.update(zip(missing, fresh))
        for note_id in stale:
            cached.pop(note_id, None)
        if (missing or stale) and cached:
            _write(path, cached)
    vectors = {note_id: _normalize(vec[None, :])[0] for note_id, vec in cached.items() if note_id in live}
    return vectors, len(missing)
