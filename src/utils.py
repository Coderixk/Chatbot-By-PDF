import os
import re
import hashlib
import fitz
import nltk
import pickle
from nltk.tokenize import sent_tokenize
import streamlit as st

# Download NLTK resources if not already downloaded
try:
    nltk.data.path.append("/usr/share/nltk_data")
except LookupError:
    nltk.download("punkt")

# Create temp directory if it doesn't exist
TEMP_DIR = "temp_chunks"
os.makedirs(TEMP_DIR, exist_ok=True)


def get_file_hash(file_content):
    """Generate a hash for the file content."""
    return hashlib.md5(file_content).hexdigest()


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split by sentences to maintain context
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed chunk_size, add the current chunk to chunks
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap for context
            current_chunk = (
                current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            )

        current_chunk += " " + sentence

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def save_faiss_data(chunks, embeddings, index, file_hash):
    """Save chunks, embeddings, and FAISS index to disk."""
    import faiss

    data_path = os.path.join(TEMP_DIR, f"{file_hash}_data.pkl")
    index_path = os.path.join(TEMP_DIR, f"{file_hash}_index.faiss")

    # Save chunks and embeddings
    with open(data_path, "wb") as f:
        pickle.dump((chunks, embeddings), f)

    # Save FAISS index
    faiss.write_index(index, index_path)


def load_faiss_data(file_hash):
    """Load chunks, embeddings, and FAISS index from disk."""
    data_path = os.path.join(TEMP_DIR, f"{file_hash}_data.pkl")
    index_path = os.path.join(TEMP_DIR, f"{file_hash}_index.faiss")

    if os.path.exists(data_path) and os.path.exists(index_path):
        # Load chunks and embeddings
        with open(data_path, "rb") as f:
            chunks, embeddings = pickle.load(f)

        # Load FAISS index
        import faiss

        index = faiss.read_index(index_path)

        return chunks, embeddings, index

    return None, None, None
