"""Load + split — the fast path.

Two things make this fast compared to a Docling-based loader:

  1. **Lightweight loaders.** `PyPDFLoader`/`Docx2txtLoader`/`TextLoader` do
     plain text extraction (via `pypdf`/`docx2txt`) — no layout-analysis or
     OCR models to download and run. Docling is more accurate for complex
     PDFs (tables, multi-column layout) but that accuracy costs real cold-
     start time; this trades some of that accuracy for speed.
  2. **Skip already-ingested files.** `ingested.json` is a tiny persisted
     registry (content-hash -> filename) checked *before* loading/embedding
     a file at all — re-uploading the same file, or just restarting the
     app, never re-does the expensive embedding step for files it's already
     indexed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document

from . import config

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


def _load_registry() -> dict:
    if config.REGISTRY_PATH.exists():
        try:
            return json.loads(config.REGISTRY_PATH.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    config.REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def already_ingested(doc_id: str) -> bool:
    return doc_id in _load_registry()


def mark_ingested(doc_id: str, filename: str, chunks: int) -> None:
    registry = _load_registry()
    registry[doc_id] = {"filename": filename, "chunks": chunks}
    _save_registry(registry)


def registry() -> dict:
    return _load_registry()


def remove_from_registry(doc_id: str) -> None:
    registry_data = _load_registry()
    registry_data.pop(doc_id, None)
    _save_registry(registry_data)


def load_file(path: Path, filename: str) -> list[Document]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()  # one Document per page, real page numbers
    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(path)).load()
    if ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


def split_documents(docs: list[Document]) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    return splitter.split_documents(docs)
