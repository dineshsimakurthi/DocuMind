"""The RAG chain, now with conversational memory via `RunnableWithMessageHistory`:

    (dense + BM25 -> EnsembleRetriever) | format_docs | prompt | llm | StrOutputParser

wrapped so that chat history is automatically read before each call and
appended after it, keyed by `session_id`.

**Trade-off worth knowing**: `RunnableWithMessageHistory` makes the LLM aware
of prior turns — it can reference earlier context in its *answer*. It does
NOT rewrite the search query for retrieval. A pure pronoun follow-up like
"what about the second one?" still hits the retriever as literally that
string. Adding a condense-the-follow-up-into-a-standalone-query step (as the
fuller version of this project does) would fix that, at the cost of one
extra LLM call per follow-up turn — not added here to keep this version simple.
"""
from __future__ import annotations

import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from . import config
from .memory import get_session_history
from .retrieval import get_hybrid_retriever

_SYSTEM = """You are answering only from the retrieved document context.

Mandatory rules:
1. Use only facts explicitly present in the context.
2. If the answer is not in the context, reply: 'I don't know based on the provided documents.'
3. Do not add unsupported assumptions or generic facts that are not in the context.
4. Give a detailed but clear explanation for a user. Write 5-10 sentences, or a short structured explanation with bullet points if the topic is multi-part.
5. Start with a direct answer sentence, then explain the concept in plain language.
6. Include a simple real-world example that helps a beginner understand the idea.
7. Prefer the exact wording from the source when possible, and do not replace technical terms with vague synonyms.
8. If the topic has steps or causes/effects, explain them in order.
9. Do not invent examples. The example must be a simple illustration of the same idea from the document.

Context:
{context}"""


def format_docs(docs) -> str:
    """retriever -> format_docs: turns the retrieved Documents into one
    plain-text context block for the prompt."""
    return "\n\n".join(d.page_content for d in docs)


def get_llm():
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Add it to a local `.env` file using the template in `.env.example` "
            "or export it in your shell before starting the app."
        )

    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=0.4,
        max_tokens=1200,
        api_key=api_key,
    )


def _build_base_chain(k: int):
    """The actual (context | question | history) -> prompt -> llm -> parser
    chain — this is what gets wrapped with message history below."""
    retriever = get_hybrid_retriever(k=k)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), MessagesPlaceholder("history"), ("human", "{question}")]
    )
    llm = get_llm()

    get_question = RunnableLambda(lambda x: x["question"])
    # Explicit RunnableLambda chaining (rather than `itemgetter | retriever`)
    # since our hybrid retriever is a thin wrapper, not a full BaseRetriever —
    # RunnableLambda(retriever.invoke) coerces its plain callable cleanly.
    context_chain = get_question | RunnableLambda(retriever.invoke) | RunnableLambda(format_docs)

    return (
        {
            "context": context_chain,
            "question": get_question,
            "history": RunnableLambda(lambda x: x.get("history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_chain(k: int = config.TOP_K):
    """The base chain wrapped with automatic chat-history read/append,
    keyed by `session_id` (passed via `config={"configurable": {...}}`)."""
    return RunnableWithMessageHistory(
        _build_base_chain(k),
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def _run_config(session_id: str) -> dict:
    return {"configurable": {"session_id": session_id}}


def ask(question: str, session_id: str = "default", k: int = config.TOP_K) -> str:
    try:
        return build_chain(k=k).invoke({"question": question}, config=_run_config(session_id))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LLM request failed. Check your Groq API key and model availability. "
            f"Original error: {exc}"
        ) from exc


def ask_stream(question: str, session_id: str = "default", k: int = config.TOP_K):
    """Same chain, streamed token-by-token (StrOutputParser streams cleanly
    since it just passes each chunk's text through)."""
    try:
        yield from build_chain(k=k).stream({"question": question}, config=_run_config(session_id))
    except Exception as exc:  # noqa: BLE001
        yield (
            "I couldn't generate an answer from the LLM. "
            "Please check your Groq API key, model availability, and that you are running the project in the correct virtual environment. "
            f"Details: {exc}"
        )


def get_sources(question: str, k: int = config.TOP_K) -> list:
    """Separate call to show *what* was retrieved, since the chain above
    only returns the final answer string. Uses the raw question — see the
    module docstring's trade-off note on why this isn't history-condensed."""
    return get_hybrid_retriever(k=k).invoke(question)


def groundedness_metrics(answer: str, docs: list) -> dict:
    """Estimate how grounded an answer is in the retrieved docs.

    This is a lightweight guardrail for the app: it checks how much of the
    answer vocabulary overlaps the retrieved context and shows a percentage.
    A low score suggests the model may be hallucinating beyond the docs.
    """
    if not docs:
        return {"groundedness": 0.0, "coverage": 0.0, "unsupported_terms": [], "source_count": 0}

    context = " ".join(d.page_content for d in docs).lower()
    answer_lower = (answer or "").lower()

    def tokens(text: str) -> set[str]:
        return {
            t for t in re.findall(r"[a-z0-9]+", text)
            if len(t) > 2 and t not in _STOPWORDS
        }

    answer_tokens = tokens(answer_lower)
    context_tokens = tokens(context)

    if not answer_tokens:
        return {"groundedness": 0.0, "coverage": 0.0, "unsupported_terms": [], "source_count": len(docs)}

    overlap = answer_tokens & context_tokens
    coverage = len(overlap) / len(answer_tokens)
    unsupported = sorted(answer_tokens - context_tokens)

    return {
        "groundedness": round(max(0.0, min(1.0, coverage)) * 100, 1),
        "coverage": round(max(0.0, min(1.0, coverage)) * 100, 1),
        "unsupported_terms": unsupported[:10],
        "source_count": len(docs),
    }


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "with", "from", "into", "onto", "for", "of", "on", "in",
    "to", "by", "be", "is", "are", "was", "were", "it", "its", "as", "at",
    "about", "after", "before", "because", "while", "when", "where", "how",
    "why", "what", "who", "which", "your", "they", "them", "their", "you",
    "we", "our", "us", "very", "more", "most", "some", "such", "not", "no",
    "can", "could", "should", "would", "will", "may", "make", "made", "many",
    "much", "also", "just", "only", "into", "over", "under", "through", "between",
    "among", "does", "do", "did", "have", "has", "had", "using", "used",
    "use", "there", "here", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "same", "like", "each", "other", "another", "moreover"
}
