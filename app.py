"""DocuMind (lean) — Streamlit frontend for the simplified, fast-loading RAG chain."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import uuid

import streamlit as st

from rag_chain import config, ingest
from rag_chain.chain import ask_stream, get_sources, groundedness_metrics
from rag_chain.memory import clear_session
from rag_chain.retrieval import invalidate as invalidate_bm25
from rag_chain.store import add_documents, collection_count, delete_by_doc_id


def _require_groq_key() -> bool:
    """Return True if a Groq key is configured, otherwise show a clear warning."""
    if os.getenv("GROQ_API_KEY"):
        return True

    st.warning(
        "Missing `GROQ_API_KEY`. Add it to a local `.env` file using the template in `.env.example` "
        "or export it in your shell before starting Streamlit."
    )
    return False


st.set_page_config(page_title="DocuMind (lean)", page_icon="📄", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "session_id" not in st.session_state:
    # One id per browser session — RunnableWithMessageHistory uses this to
    # look up (and auto-update) this session's chat history.
    st.session_state.session_id = str(uuid.uuid4())


def ingest_file(filename: str, raw: bytes) -> str:
    """Returns a short status string. Skips embedding entirely if this exact
    file content was already indexed (see rag_chain/ingest.py's registry)."""
    doc_id = ingest.file_hash(raw)
    if ingest.already_ingested(doc_id):
        return f"Already indexed — skipped re-embedding {filename}."

    ext = Path(filename).suffix.lower()
    dest = config.UPLOAD_DIR / f"{doc_id}{ext}"
    dest.write_bytes(raw)

    docs = ingest.load_file(dest, filename)
    for d in docs:
        d.metadata["doc_id"] = doc_id
        d.metadata["doc_name"] = filename
    chunks = ingest.split_documents(docs)

    add_documents(chunks)
    invalidate_bm25()
    ingest.mark_ingested(doc_id, filename, len(chunks))
    return f"Indexed {filename} — {len(chunks)} chunks."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 DocuMind (lean)")
    st.caption("Hybrid RAG: (dense + BM25 → EnsembleRetriever) → format_docs → prompt → llm → StrOutputParser")
    try:
        st.caption(f"Indexed chunks: **{collection_count()}**")
    except RuntimeError as exc:
        st.error(str(exc))

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}`")
    if st.button("🧹 Clear chat memory"):
        st.session_state.messages = []
        clear_session(st.session_state.session_id)
        st.rerun()
    uploads = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads:
        for f in uploads:
            with st.spinner(f"Ingesting {f.name}…"):
                try:
                    st.success(ingest_file(f.name, f.getvalue()))
                except Exception as e:  # noqa: BLE001
                    st.error(f"{f.name}: {e}")
        st.session_state.uploader_key += 1  # reset the uploader so it stops
        st.rerun()                          # resubmitting the same files

    st.subheader("Indexed documents")
    reg = ingest.registry()
    if not reg:
        st.caption("No documents yet.")
    for doc_id, info in reg.items():
        cols = st.columns([4, 1])
        cols[0].write(f"📄 {info['filename']} ({info['chunks']} chunks)")
        if cols[1].button("🗑", key=f"del-{doc_id}"):
            delete_by_doc_id(doc_id)
            invalidate_bm25()
            ingest.remove_from_registry(doc_id)
            st.rerun()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
st.header("Ask your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents…")
if question:
    if not _require_groq_key():
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        if not ingest.registry():
            st.warning("Upload a document first.")
        else:
            answer_box = st.empty()
            buffer = ""
            try:
                for token in ask_stream(question, session_id=st.session_state.session_id):
                    buffer += token
                    answer_box.markdown(buffer + "▌")
                answer_box.markdown(buffer)
            except Exception as exc:  # noqa: BLE001
                buffer = (
                    "I couldn't generate an answer from the LLM. "
                    "Please check your Groq API key, model availability, and environment."
                )
                answer_box.markdown(buffer)
                st.caption(f"LLM error: {exc}")

            sources = get_sources(question)
            metrics = groundedness_metrics(buffer, sources)

            st.markdown("### Answer quality / grounding")
            cols = st.columns(3)
            cols[0].metric("Groundedness", f"{metrics['groundedness']:.1f}%")
            cols[1].metric("Source coverage", f"{metrics['coverage']:.1f}%")
            cols[2].metric("Docs used", metrics['source_count'])

            if metrics["unsupported_terms"]:
                st.caption(
                    "Potential unsupported terms: " + ", ".join(metrics["unsupported_terms"][:5])
                )
            else:
                st.caption("This answer is strongly grounded in the retrieved document context.")

            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for d in sources:
                        page = d.metadata.get("page")
                        loc = f"p.{page + 1}" if isinstance(page, int) else ""
                        st.caption(f"{d.metadata.get('doc_name', 'unknown')} {loc}")
            else:
                st.caption("No source documents were retrieved for this question.")

    st.session_state.messages.append({"role": "assistant", "content": buffer})
