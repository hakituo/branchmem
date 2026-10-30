"""Optional layer: a provenance-carrying note store and gated hybrid retrieval.

BM25-only retrieval needs nothing extra; vector ranking needs numpy and an embedder.
"""

from .retrieval import NoteRetriever, reset_note_retrieval_cache
from .store import NoteStore, ReplaceResult, make_note_id

__all__ = ["NoteRetriever", "NoteStore", "ReplaceResult", "make_note_id", "reset_note_retrieval_cache"]
