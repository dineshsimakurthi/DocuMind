"""Embeddings + vector store — cached so the model only loads **once per
process**, and persisted to disk so a restart doesn't lose the index.

  * `HuggingFaceBgeEmbeddings` — real sentence-transformers model (PyTorch).
    Using `HuggingFaceBgeEmbeddings` specifically (not the newer
    `langchain_huggingface.HuggingFaceEmbeddings`) because BGE models are
    asymmetric — queries want an instruction prefix, passages don't — and
    this is the LangChain class that actually exposes a `query_instruction`
    field for that; the newer class doesn't have one (strict pydantic model,
    `extra="forbid"`, would raise if you passed it).
  * `QdrantVectorStore(path=...)` — embedded mode, in-process, on-disk.
    No server, no Docker; the index survives app restarts, so (combined with
    `ingest.py`'s already-ingested registry) you only pay the embedding cost
    once per file.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from . import config

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    return HuggingFaceBgeEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction=_QUERY_INSTRUCTION,
    )


@lru_cache
def _client():
    from qdrant_client import QdrantClient

    try:
        return QdrantClient(path=str(config.QDRANT_PATH))
    except RuntimeError as exc:
        msg = str(exc)
        if "already accessed by another instance of Qdrant client" in msg:
            raise RuntimeError(
                "Qdrant local storage is already open in another process. "
                "Close any other Streamlit app, eval script, or Python process using the same project, "
                "then restart. If needed, remove data/qdrant_local and re-ingest your documents."
            ) from exc
        raise


def _ensure_collection() -> None:
    from qdrant_client.http import models as qm

    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME in existing:
        return
    dim = len(get_embeddings().embed_query("dimension probe"))
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


@lru_cache
def get_vectorstore():
    from langchain_qdrant import QdrantVectorStore

    _ensure_collection()
    return QdrantVectorStore(client=_client(), collection_name=config.COLLECTION_NAME, embedding=get_embeddings())


def add_documents(docs: list[Document]) -> None:
    if docs:
        get_vectorstore().add_documents(docs)


def collection_count() -> int:
    client = _client()
    if config.COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        return 0
    return client.count(config.COLLECTION_NAME, exact=True).count


def delete_by_doc_id(doc_id: str) -> None:
    from qdrant_client.http import models as qm

    client = _client()
    if config.COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        return
    client.delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="metadata.doc_id", match=qm.MatchValue(value=doc_id))])
        ),
    )


def scroll_all_documents() -> list[Document]:
    """Fetch every stored chunk back out as a `Document` — this is what
    `retrieval.py` uses to (re)build the BM25 index, since BM25 needs the
    full corpus in memory rather than a vector index."""
    client = _client()
    if config.COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        return []
    out: list[Document] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            out.append(Document(page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {})))
        if next_offset is None:
            break
    return out


def get_dense_retriever(k: int = config.TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
