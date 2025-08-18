"""
RAGAS evaluation (Chat Evaluation only, Groq-only).
- Ambil GROQ_API_KEY dari st.secrets (fallback ke os.getenv)
- Pakai LlamaIndex Groq LLM via RAGAS
- Return pandas.DataFrame (atau dict {"error": "..."} saat gagal)
"""

from typing import List, Dict, Optional
import os

import streamlit as st
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, context_precision, context_recall
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.llms.groq import Groq

from src.embedding import load_sentence_transformer


# ---------- helpers ----------
def _get_groq_api_key() -> Optional[str]:
    """Ambil Groq API key dari st.secrets lalu fallback ke environment."""
    # 1) st.secrets
    try:
        v = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
        if v:
            return str(v)
    except Exception:
        pass
    # 2) environment
    v = os.getenv("GROQ_API_KEY")
    if not v:
        st.error("GROQ_API_KEY tidak ditemukan di .streamlit/secrets.toml atau .env")
        return None
    return v


def _llm(temperature: float = 0.1) -> Optional[LlamaIndexLLMWrapper]:
    """Buat LLM evaluator untuk RAGAS."""
    api = _get_groq_api_key()
    if not api:
        return None
    groq = Groq(
        api_key=api,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=temperature,
    )
    return LlamaIndexLLMWrapper(groq)


def _truncate(text: str, max_chars: int = 1200) -> str:
    """Potong teks agar hemat token."""
    return text if len(text) <= max_chars else text[:max_chars] + " ..."


# ---------- prepare evaluation data ----------
def prepare_evaluation_data(
    chat_history: List[Dict[str, str]], contexts: List[str]
) -> Optional[Dataset]:
    """
    Ambil pasangan QA terakhir dari chat_history + contexts (subset yang diberikan).
    contexts: list of strings (chunk teks) yang dipakai saat menjawab
    """
    if len(chat_history) < 2:
        return None

    # cari pasangan Q (user) -> A (assistant) terakhir
    last_q = None
    last_a = None
    for msg in reversed(chat_history):
        if msg["role"] == "assistant" and last_a is None:
            last_a = msg["content"]
        elif msg["role"] == "user" and last_q is None:
            last_q = msg["content"]
        if last_q and last_a:
            break

    if not last_q or not last_a:
        return None
    if ("No relevant information found" in last_a) or ("Please upload a PDF" in last_a):
        return None

    if not contexts:
        return None

    # gunakan maksimal 3 konteks pertama agar stabil & hemat token
    top_k = min(3, len(contexts))
    ctx = [_truncate(c) for c in contexts[:top_k]]

    data = {
        "question": [last_q],
        "answer": [last_a],
        "contexts": [ctx],
        # pakai jawaban sebagai approx GT (tanpa test dataset)
        "ground_truths": [last_a],
        "reference": [" ".join(ctx)],
    }
    return Dataset.from_dict(data)


# ---------- main API (dipakai app.py) ----------
def evaluate_rag_system(
    chat_history: List[Dict[str, str]], contexts: List[str]
) -> pd.DataFrame | Dict[str, str]:
    """
    Evaluasi QA terakhir dengan konteks yang dipakai saat menjawab.
    - chat_history: riwayat chat [{"role": "user"/"assistant", "content": "..."}]
    - contexts: list chunk yang dipakai saat menjawab (mis. st.session_state.last_contexts)
    """
    ds = prepare_evaluation_data(chat_history, contexts)
    if ds is None:
        return {
            "error": "Not enough data for evaluation. Pastikan ada minimal satu pasang tanya–jawab dan konteks tersedia."
        }

    llm = _llm(temperature=0.1)
    if llm is None:
        return {"error": "Groq LLM unavailable (cek GROQ_API_KEY)."}

    try:
        res = evaluate(
            ds,
            metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm),
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=load_sentence_transformer(),
        )
        return res.to_pandas()  # DataFrame
    except Exception as e:
        st.error(f"RAGAS evaluation error: {e}")
        return {"error": f"{e}"}
