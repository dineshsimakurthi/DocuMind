"""Minimal settings — just what this simplified pipeline needs."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
QDRANT_PATH = DATA_DIR / "qdrant_local"      # embedded Qdrant — no server, no Docker
COLLECTION_NAME = "documind_chunks"
REGISTRY_PATH = DATA_DIR / "ingested.json"   # tracks which files are already indexed

# Embeddings — real sentence-transformers model via HuggingFace (not fastembed).
# Use a smaller model for faster startup and lower CPU memory; this is the
# best trade-off for a lightweight local app. It is slightly less powerful than
# BGE-small, but much faster to load and respond.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Chunking — smaller, tighter chunks are better for manuals and policy-ish
# documents because they keep each retrieval unit semantically focused and
# reduce cross-topic contamination. This improves answer grounding.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Retrieval — keep the hybrid search precise enough to surface only the most
# relevant evidence without stuffing the prompt with unrelated chunks.
TOP_K = 5
DENSE_TOP_K = 5
BM25_TOP_K = 5
RRF_C = 50
HYBRID_WEIGHTS = [0.7, 0.3]  # [dense, bm25]

# LLM — split across two providers/API keys on purpose:
#   generation (the actual chat feature, chain.py)  -> Groq   / GROQ_API_KEY
#   evaluation (RAGAS judge + testset gen, eval/*)   -> Gemini / GOOGLE_API_KEY
# Both draw from independent free-tier quotas, so running a full RAGAS eval
# (which can be dozens of LLM calls) never competes with, or exhausts, the
# quota your actual chat feature needs. See chain.py's get_llm() and
# eval/ragas_eval.py's _get_eval_llm() for where each is used.
GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_EVAL_MODEL = "gemini-3.5-flash-lite".strip()

for d in (DATA_DIR, UPLOAD_DIR, QDRANT_PATH):
    d.mkdir(parents=True, exist_ok=True)
