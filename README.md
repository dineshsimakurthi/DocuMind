# DocuMind (lean) — Hybrid RAG with LangChain

A small, hybrid-retrieval RAG project built around one chain pattern:

```
(dense + BM25 → EnsembleRetriever) | format_docs | prompt_template | llm | StrOutputParser
```

Dense (sentence-transformers embeddings + Qdrant) and BM25 (exact-token lexical search) are fused via LangChain's `EnsembleRetriever` — its native Reciprocal Rank Fusion. **No reranking stage** — this is "hybrid RAG only," not "hybrid + rerank," kept deliberately simple. See `rag_chain/retrieval.py` and `rag_chain/chain.py`.

## ⚠️ Version pinning matters here

LangChain shipped a **v1.x major release** that restructured `langchain.retrievers` (where `EnsembleRetriever` lives) and bumped `langchain-core` to `>=1.0`. Installing `langchain`/`langchain-core` unpinned will pull in the new major version and break the imports in this project. `requirements.txt` pins to the last verified-compatible `0.3.x` set:

```
langchain==0.3.27
langchain-core==0.3.86
langchain-community==0.3.27
langchain-text-splitters==0.3.11
langchain-qdrant<1.0
langchain-groq<1.0
```

If you want to move to LangChain v1.x later, `EnsembleRetriever` still exists there — just under a different import path — worth checking their migration guide before bumping.

## The stack

| Stage | Component | File |
|---|---|---|
| Loader | `PyPDFLoader` / `Docx2txtLoader` / `TextLoader` | `rag_chain/ingest.py` |
| Splitter | `RecursiveCharacterTextSplitter` | `rag_chain/ingest.py` |
| Embeddings | `HuggingFaceBgeEmbeddings` (real sentence-transformers model) | `rag_chain/store.py` |
| Vector DB | `QdrantVectorStore`, embedded/local — no Docker | `rag_chain/store.py` |
| Lexical | `BM25Retriever` | `rag_chain/retrieval.py` |
| Fusion | `EnsembleRetriever` (native RRF) | `rag_chain/retrieval.py` |
| Generation | `ChatPromptTemplate` → `ChatGroq` → `StrOutputParser` | `rag_chain/chain.py` |

## Why `HuggingFaceBgeEmbeddings` specifically

BGE models are asymmetric — queries want an instruction prefix, passages don't. `HuggingFaceBgeEmbeddings` (from `langchain_community`) is the LangChain class that actually exposes a `query_instruction` field for this. The newer `langchain_huggingface.HuggingFaceEmbeddings` class does **not** have that field — it's a strict pydantic model (`extra="forbid"`) that would raise a validation error if you tried. Verified directly against both installed classes before writing this.

## Why hybrid, and why no reranker

- **Dense** (sentence-transformer cosine similarity) catches semantic/paraphrase matches.
- **BM25** catches exact tokens dense embeddings are comparatively weak at — error codes, SKUs, IDs.
- **`EnsembleRetriever`** fuses both by *rank position* (Reciprocal Rank Fusion), not raw score — cosine similarity and BM25 scores live on incompatible scales, so RRF sidesteps needing to reconcile them.
- No cross-encoder reranking stage on top — that's the accuracy/latency trade this version makes to stay simple. Add `ContextualCompressionRetriever` + `CrossEncoderReranker` wrapping the ensemble retriever if you want that precision pass back.

## The chain itself

```python
retriever = get_hybrid_retriever(k=4)       # dense + BM25 -> EnsembleRetriever
prompt = ChatPromptTemplate.from_template(_TEMPLATE)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

- `retriever | format_docs` — the fused, ranked `Document` list gets piped straight into `format_docs`, which joins their text into one context block.
- `RunnablePassthrough()` — passes the raw question straight through unchanged into the prompt's `{question}` slot.
- `StrOutputParser()` — unwraps the LLM's message object into a plain string (and streams cleanly, chunk by chunk).

## Keeping loading fast despite going back to sentence-transformers

Real sentence-transformers models are slower to cold-start than the fastembed/ONNX alternative — that's the honest trade-off of using an actual HuggingFace/sentence-transformers embedding model as requested here. Two things still keep re-runs fast:

- **`data/ingested.json` registry** — a file's content hash is checked *before* loading or embedding it at all; already-indexed files are skipped entirely on re-upload or app restart.
- **`@lru_cache` on `get_embeddings()`/`get_vectorstore()`** — the model loads once per process, not once per Streamlit rerun.
- **BM25 rebuilds lazily**, only when `retrieval.invalidate()` has actually been called (after an ingest/delete) — not on every question asked.

If cold-start time still matters more than accuracy for your use case, `BAAI/bge-small-en-v1.5` (the current default in `config.py`) is already the smaller end of BGE models; `sentence-transformers/all-MiniLM-L6-v2` is even smaller if you want to trade further.

## Conversational memory

Chat history is wired in via LangChain's `RunnableWithMessageHistory` (`rag_chain/chain.py` + `rag_chain/memory.py`):

```python
return RunnableWithMessageHistory(
    base_chain, get_session_history,
    input_messages_key="question", history_messages_key="history",
)
```

It wraps the base chain so history is automatically fetched before each call and appended after — keyed by a `session_id` generated once per browser session in `app.py`. Verified end-to-end with a fake LLM that reports how many prior messages it received: 0 on turn 1, 2 on turn 2, 4 on turn 3 — confirming history actually grows turn-over-turn rather than just being decorative.

**Trade-off worth knowing:** this makes the *LLM's answer* aware of prior turns, but it does **not** rewrite the search query for retrieval. A pure pronoun follow-up like "what about the second one?" still hits the hybrid retriever as that literal string — the retriever doesn't know what "the second one" refers to. Fixing that needs a query-condensing step (rewrite the follow-up into a standalone question before retrieval, as the fuller version of this project does) — one extra LLM call per follow-up turn, not added here to keep this version simple.

**Heads-up (not a current bug):** `RunnableWithMessageHistory` is flagged as pending-deprecated in LangChain, in favor of LangGraph's built-in persistence/checkpointing. It works correctly today — this is worth knowing before the project sits unmaintained for a long stretch, not a reason to avoid it now.

**Also in-memory only:** `_store` in `memory.py` is a plain process-local dict — history is lost on restart, and it doesn't share across multiple app instances. Swap `InMemoryChatMessageHistory` for a persisted backend (e.g. `SQLChatMessageHistory`) if that matters for your use case.

## Evaluation (RAGAS)

`eval/ragas_eval.py` scores your live RAG chain on the 4 things that actually matter for a RAG system, using [RAGAS](https://docs.ragas.io) — which is itself LLM-as-judge, just with pre-built, tested judge prompts instead of hand-rolled ones:

| What you asked for | RAGAS metric(s) | Needs a reference answer? |
|---|---|---|
| Faithfulness / groundedness | `Faithfulness` | No |
| Answer relevance | `ResponseRelevancy` | No |
| Retrieval relevance | `LLMContextPrecisionWithoutReference` + `LLMContextRecall` | Precision: no. Recall: **yes** |
| Correctness | `FactualCorrectness` + `SemanticSimilarity` | **Yes** |

**How each one actually works, briefly:**
- **Faithfulness**: the evaluator LLM breaks your answer into individual claims, then checks each claim against the retrieved context — score = supported claims ÷ total claims.
- **Answer relevance**: the evaluator LLM generates questions *from* your answer, embeds them, and compares similarity to the original question — an accurate-but-off-topic answer scores low here even if faithful.
- **Retrieval relevance (precision)**: judges whether retrieved chunks are relevant and ranked near the top. **(recall)**: needs a reference answer — checks whether the retrieved context contains everything needed to produce it, catching "the retriever missed something" that precision alone can't.
- **Correctness**: factual overlap + semantic similarity against a written reference answer.

### ⚠️ Another version conflict, found the same way as the LangChain one

Installing `ragas` unpinned pulls in `langchain-core>=1.0` and silently upgrades your whole LangChain stack — breaking `EnsembleRetriever` exactly like before. **`ragas==0.2.15`** is the version verified (by actually installing and testing it) to work alongside the `langchain==0.3.27` pin above. It also hard-imports `langchain_openai` in its embeddings module even though we never call OpenAI — that import just needs to exist, so `langchain-openai<1.0` is in requirements for that reason alone.

### Run it

```bash
python -m eval.ragas_eval
```

Add real questions + reference answers to `eval/qa_dataset.json` first — the shipped ones are placeholders matching a generic product-manual example, not your actual corpus. Results print per-question and as averages, and get saved to `eval/ragas_results.csv`.

**Cost/rate-limit note**: each metric is 1–3+ LLM calls per question under the hood — six metrics across even a small dataset is a real number of calls. That's exactly why the evaluator LLM (`eval/ragas_eval.py`'s `_get_eval_llm()`) runs on **Gemini** (`GOOGLE_API_KEY`), a completely separate provider and API key from the chain's own answer-generation LLM (**Groq**, `GROQ_API_KEY`, `chain.py`'s `get_llm()`). Running a full eval can never exhaust the quota your actual chat feature needs, because they're not sharing one — see both `.env.example` entries. `.with_retry()` + `raise_exceptions=False` handle transient failures within the eval run itself; a failed cell just shows up as `NaN` instead of killing the whole run.

## Trade-offs versus the fuller version

- **No reranking stage** — hybrid retrieval only, as requested. Add `CrossEncoderReranker` back in if you need the extra precision pass.
- **No citation markers in the answer** — sources are shown separately (doc name + page) below the answer, not parsed out of the model's response.
- **No LLM fallback chain** — a single `ChatGroq` call. Add `.with_retry().with_fallbacks([...])` back in `chain.py`'s `get_llm()` for resilience against rate limits.
- **In-memory BM25, rebuilt from Qdrant's payloads** — fine at this project's scale; a persistent sparse index is the scale-up path for a much larger corpus.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
streamlit run app.py
```
