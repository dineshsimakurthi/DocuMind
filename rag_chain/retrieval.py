"""Hybrid retrieval — **the only retrieval mode** in this project now:

    dense (Qdrant cosine search) + BM25 (exact-token lexical search)
        -> fused via EnsembleRetriever (LangChain's native Reciprocal Rank Fusion)

No reranking stage — this is intentionally "hybrid RAG only," not
"hybrid + rerank." BM25 catches exact tokens (error codes, IDs, SKUs) that
sentence-transformer embeddings are comparatively weak at; dense catches
paraphrases/semantic matches BM25 can't. Fusing both by rank position
rather than raw score avoids having to reconcile cosine similarity and
BM25 scores on incompatible scales.

The BM25 index is in-memory and rebuilt lazily from the vector store's
payloads whenever `invalidate()` has been called (after any ingest/delete)
— fine at this project's scale; a persistent sparse index would be the
scale-up path for a much larger corpus.
"""
from __future__ import annotations

import threading

from . import config
from .store import get_dense_retriever, scroll_all_documents

_lock = threading.Lock()
_bm25_retriever = None
_dirty = True


def invalidate() -> None:
    """Call after any ingest/delete — BM25 is rebuilt lazily on next use."""
    global _dirty
    with _lock:
        _dirty = True


def _get_bm25_retriever():
    from langchain_community.retrievers import BM25Retriever

    global _bm25_retriever, _dirty
    with _lock:
        if _dirty or _bm25_retriever is None:
            docs = scroll_all_documents()
            _bm25_retriever = BM25Retriever.from_documents(docs) if docs else None
            if _bm25_retriever:
                _bm25_retriever.k = config.BM25_TOP_K
            _dirty = False
    return _bm25_retriever


class _TopK:
    """Thin wrapper giving `EnsembleRetriever` a plain `.invoke(query) ->
    list[Document]` truncated to the requested final `k`, and letting it
    still pipe into `format_docs` via `|` like a normal Runnable."""

    def __init__(self, retriever, k: int, reranker=None):
        self._retriever = retriever
        self._k = k
        self._reranker = reranker

    def invoke(self, query: str):
        docs = self._retriever.invoke(query)
        if self._reranker is not None and docs:
            docs = self._reranker.compress_documents(docs, query)
        return docs[: self._k]

    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self.invoke) | other


def get_hybrid_retriever(k: int = config.TOP_K):
    """dense + BM25 -> EnsembleRetriever (RRF fusion). Falls back to
    dense-only if there's no BM25 index yet (e.g. nothing ingested)."""
    from langchain.retrievers import EnsembleRetriever

    dense = get_dense_retriever(k=config.DENSE_TOP_K)
    bm25 = _get_bm25_retriever()
    if bm25 is None:
        return _TopK(dense, k)

    ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=config.HYBRID_WEIGHTS, c=config.RRF_C)
    reranker = None
    try:
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-base",
                model_kwargs={"device": "cpu"},
            ),
            top_n=k,
        )
    except Exception:  # noqa: BLE001
        reranker = None

    return _TopK(ensemble, k, reranker=reranker)
