"""RAGAS evaluation — runs your live RAG chain against `eval/qa_dataset.json`
and scores it on the 4 things you asked for:

    answer relevance      -> ResponseRelevancy          (reference-free)
    retrieval relevance   -> LLMContextPrecisionWithoutReference (reference-free)
                              + LLMContextRecall          (reference-based)
    faithfulness/grounded -> Faithfulness                (reference-free)
    correctness           -> FactualCorrectness + SemanticSimilarity (reference-based)

"Reference-free" metrics need only (question, answer, retrieved context) —
they run on live output from your actual `RagChain`, no labeled data needed.
"Reference-based" metrics additionally need a written reference answer, which
is what `eval/qa_dataset.json` supplies per question.

Usage (after ingesting a corpus via the Streamlit app):
    python -m eval.ragas_eval

**Cost/rate-limit note**: each metric is 1-3+ LLM calls per question under
the hood (e.g. faithfulness = decompose claims, then verify each). Six
metrics across even a small dataset is genuinely a lot of calls — that's
exactly why the evaluator LLM below is deliberately a *different provider*
from the chain's own answer-generation LLM (see `_get_eval_llm` docstring).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    FactualCorrectness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
    SemanticSimilarity,
)

from rag_chain import config
from rag_chain.chain import ask, get_sources
from rag_chain.store import get_embeddings

DATASET_PATH = Path(__file__).parent / "qa_dataset.json"
RESULTS_PATH = Path(__file__).parent / "ragas_results.csv"
SUMMARY_PATH = Path(__file__).parent / "ragas_summary.csv"


def _get_eval_llm():
    """A dedicated evaluator LLM on a **different provider** (Gemini) from
    the chain's own `get_llm()` (Groq) — so a RAGAS run's many judge calls
    draw from a genuinely separate free-tier quota/API key, and can never
    exhaust the quota your actual chat feature needs.

    Requires GOOGLE_API_KEY in .env (see .env.example). Get a free key at
    https://aistudio.google.com/apikey — separate from your Groq key.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit(
            "No GOOGLE_API_KEY found. The evaluator LLM uses Gemini specifically "
            "so it doesn't share Groq's quota with your chat feature — add "
            "GOOGLE_API_KEY to .env (see .env.example)."
        )

    llm = ChatGoogleGenerativeAI(model=config.GEMINI_EVAL_MODEL, temperature=0.0).with_retry(
        stop_after_attempt=5, wait_exponential_jitter=True
    )
    return LangchainLLMWrapper(llm)


def build_dataset() -> EvaluationDataset:
    if not DATASET_PATH.exists():
        sys.exit(f"No eval set found at {DATASET_PATH}. Add questions to eval/qa_dataset.json first.")
    items = json.loads(DATASET_PATH.read_text())

    samples = []
    for i, item in enumerate(items):
        question = item["question"]
        reference = item.get("reference_answer")

        # Fresh session per question — evaluating single-turn Q&A, not a
        # conversation, so we don't want RunnableWithMessageHistory's memory
        # bleeding context from one eval question into the next.
        session_id = f"ragas-eval-{i}"
        answer = ask(question, session_id=session_id)  # uses Groq (chain.py's get_llm)
        retrieved_docs = get_sources(question)

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=[d.page_content for d in retrieved_docs],
                reference=reference,
            )
        )
    return EvaluationDataset.from_list([s.model_dump() for s in samples])


def _metric_column(df, metric_name: str) -> str | None:
    """Return the real DataFrame column name for a metric, regardless of small
    naming differences across RAGAS versions."""
    candidates = [metric_name]
    if metric_name == "factual_correctness":
        candidates.extend(["factual_correctness(mode=f1)", "factual_correctness(mode=f1) ", "factual_correctness(mode=f1) "])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        if column.startswith(metric_name):
            return column
    return None


def _metric_summary_rows(df):
    """Return a compact summary list of metric -> average score."""
    summary = []
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_without_reference",
        "context_recall",
        "factual_correctness",
        "semantic_similarity",
    ]:
        column = _metric_column(df, metric)
        if column is None:
            continue
        summary.append({"metric": metric, "average_score": float(df[column].mean(skipna=True))})
    return summary


def run() -> None:
    dataset = build_dataset()
    llm = _get_eval_llm()  # uses Gemini — separate quota from the chain above
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
        LLMContextRecall(llm=llm),
        FactualCorrectness(llm=llm),
        SemanticSimilarity(embeddings=embeddings),
    ]

    print(f"Evaluating {len(dataset)} questions across {len(metrics)} metrics (using Gemini as judge)...")
    result = evaluate(
        dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)

    summary_rows = _metric_summary_rows(df)
    summary_df = __import__('pandas').DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n=== Per-question scores ===")
    print(df.to_string(index=False))

    print("\n=== Average metric scores ===")
    if summary_rows:
        for row in summary_rows:
            print(f"  {row['metric']}: {row['average_score']:.3f}")
    else:
        print("  No metric columns were returned by the evaluation run.")

    print("\n=== Summary table ===")
    if summary_rows:
        print(summary_df.to_string(index=False, formatters={"average_score": lambda x: f"{x:.3f}"}))
    else:
        print("  No summary rows available.")

    print(f"\nFull results saved to {RESULTS_PATH}")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    run()
