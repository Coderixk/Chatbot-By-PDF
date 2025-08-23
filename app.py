# app.py — Chat Evaluation only

from dotenv import load_dotenv
import src.streamlit_patch as streamlit_patch  # noqa: F401
import streamlit as st
import nltk
import os
import base64
import pandas as pd

# -------------------- Bootstrap --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Berhenti rapi jika key belum ada
if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY tidak ditemukan.\n\n"
        "Buat file `.env` di root proyek dan isi:\n\n"
        "GROQ_API_KEY=your_groq_api_key_here\n\n"
        "Lalu jalankan ulang aplikasi."
    )
    st.stop()

# Sesuai permintaanmu: biarkan 'punkt_tab' apa adanya
nltk.download("punkt_tab")

# -------------------- App imports --------------------
from src.utils import (
    get_file_hash,
    extract_text_from_pdf,
    chunk_text,
    save_faiss_data,
    load_faiss_data,
)
from src.embedding import create_faiss_index, retrieve_relevant_chunks
from src.ollama import generate_response
from src.ragas_eval import evaluate_rag_system  # ⬅️ hanya ini yang dipakai

# -------------------- Page config --------------------
st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")


# -------------------- Session State --------------------
def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = ""
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = ""
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 3
    if "query_submitted" not in st.session_state:
        st.session_state.query_submitted = False
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "approach" not in st.session_state:
        st.session_state.approach = "zero-shot"
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []


# -------------------- Core logic --------------------
def process_pdf(uploaded_file):
    """Process a PDF file and create/load embeddings."""
    file_content = uploaded_file.getvalue()
    file_hash = get_file_hash(file_content)

    # hanya proses jika file baru
    if st.session_state.file_hash != file_hash:
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_hash = file_hash
        st.session_state.processing_complete = False
        st.session_state.chat_history = []  # reset chat

        # coba load cache FAISS
        chunks, embeddings, index = load_faiss_data(file_hash)

        if chunks is None:
            status_text = st.empty()
            progress_bar = st.progress(0)

            status_text.text("Extracting text from PDF...")
            progress_bar.progress(10)
            pdf_text = extract_text_from_pdf(uploaded_file)
            progress_bar.progress(30)

            status_text.text("Chunking text...")
            chunks = chunk_text(pdf_text)
            progress_bar.progress(60)

            status_text.text("Creating FAISS index...")
            embeddings, index = create_faiss_index(chunks)
            progress_bar.progress(85)

            save_faiss_data(chunks, embeddings, index, file_hash)
            progress_bar.progress(100)
            status_text.text("")
            st.success(f"Processed and indexed {len(chunks)} text chunks")
        else:
            st.success(f"Loaded {len(chunks)} indexed text chunks from cache")

        st.session_state.chunks = chunks
        st.session_state.faiss_index = index
        st.session_state.processing_complete = True


def handle_user_query(user_query: str, approach: str):
    """Process a user query and generate a response."""
    # cegah double-submit
    if st.session_state.query_submitted:
        st.session_state.query_submitted = False
        return

    # simpan pertanyaan
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    if st.session_state.chunks and st.session_state.faiss_index is not None:
        with st.spinner("Searching document..."):
            relevant_chunks = retrieve_relevant_chunks(
                user_query,
                st.session_state.chunks,
                st.session_state.faiss_index,
                top_k=st.session_state.num_chunks,
            )
            # ⬇️ simpan konteks yang benar-benar dipakai
            st.session_state.last_contexts = relevant_chunks

        if relevant_chunks:
            response = generate_response(
                user_query,
                relevant_chunks,
                model_name=st.session_state.ollama_model,
                approach=approach,
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
        else:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "No relevant information found. Try rephrasing your question.",
                }
            )
    else:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "Please upload a PDF document first."}
        )

    st.session_state.query_submitted = True
    st.rerun()


def display_pdf(file_b64: str):
    st.markdown(
        f"""
        <iframe src="data:application/pdf;base64,{file_b64}"
                width="100%" height="600" type="application/pdf"></iframe>
        """,
        unsafe_allow_html=True,
    )


# -------------------- Sidebar (Chat Evaluation only) --------------------
def render_sidebar():
    st.sidebar.subheader("⚙️Settings")

    # Kunci ke zero-shot (tanpa opsi)
    st.session_state.approach = "zero-shot"

    # jumlah chunk untuk retrieval
    num_chunks = st.sidebar.slider(
        "Jumlah Chunk yang ingin di ambil", 1, 10, st.session_state.num_chunks
    )
    if num_chunks != st.session_state.num_chunks:
        st.session_state.num_chunks = num_chunks

    st.sidebar.info(
        "Aplikasi ini menggunakan Groq API dengan model Llama 4."
    )

    # --------- Chat Evaluation ----------
    st.sidebar.subheader("🧠RAGAS Evaluation (Chat)")

    run_eval = st.sidebar.button(
        "Run RAGAS Evaluation",
        disabled=not st.session_state.processing_complete
        or len(st.session_state.chat_history) < 2,
        help="Butuh minimal satu pasang tanya–jawab di chat.",
    )

    if run_eval:
        # gunakan konteks yang dipakai saat menjawab; fallback ke semua chunks
        contexts = st.session_state.last_contexts or st.session_state.chunks

        # (opsional) tampilkan konteks yang dievaluasi untuk debugging
        with st.sidebar.expander("Konteks yang dievaluasi"):
            for i, c in enumerate(contexts, 1):
                st.write(f"--- Context #{i} ---")
                st.write(c[:500] + ("..." if len(c) > 500 else ""))

        results = evaluate_rag_system(st.session_state.chat_history, contexts)
        st.session_state.evaluation_results = results
        st.sidebar.success("Evaluation complete!")

    # tampilkan hasil
    res = st.session_state.get("evaluation_results")
    if res is not None:
        st.sidebar.subheader("Evaluation Results")
        if isinstance(res, dict) and "error" in res:
            st.sidebar.error(res["error"])
        else:
            try:
                df = res if hasattr(res, "columns") else pd.DataFrame(res)
                metric_cols = [
                    c for c in df.columns
                    if any(k in c.lower() for k in ["faith", "answer", "precision", "recall"])
                ]
                if metric_cols:
                    means = df[metric_cols].mean(numeric_only=True)
                    for c in metric_cols:
                        v = means[c]
                        st.sidebar.metric(
                            c.replace("_", " ").title(),
                            f"{v*100:.2f}%" if isinstance(v, (int, float)) and 0 <= v <= 1 else f"{v:.3f}",
                        )
                with st.sidebar.expander("Detail per-sample"):
                    st.sidebar.dataframe(df)
            except Exception:
                st.sidebar.write(res)



# -------------------- Main UI --------------------
def main():
    init_session_state()

    st.title("🤖 Chatbot PDF")
    render_sidebar()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📰 Unggah File PDF")
        uploaded_file = st.file_uploader("Pilih File:", type="pdf")

        if uploaded_file is not None:
            process_pdf(uploaded_file)

            if st.session_state.file_name:
                if st.session_state.processing_complete:
                    st.success("File PDF sudah di proses dan siap untuk ditanyakan!")
                else:
                    st.warning("PDF sedang di proes. mohon tunggu")

            if st.session_state.processing_complete:
                base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                display_pdf(base64_pdf)

    with col2:
        st.subheader("Tanya jawab Dengan AI")

        # riwayat chat
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f"**You:** {chat['content']}")
                else:
                    st.markdown(f"**Assistant:** {chat['content']}")

        # form input
        with st.form("query_form", clear_on_submit=True):
            user_query = st.text_input(
                "Berikan Pertanyaan seputar isi PDF:",
                disabled=not st.session_state.processing_complete,
            )
            submit = st.form_submit_button("Submit")
            if submit and user_query and st.session_state.processing_complete:
                handle_user_query(user_query, st.session_state.approach)


if __name__ == "__main__":
    main()
