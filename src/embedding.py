"""
Embedding and FAISS-related functions for the PDF Chatbot application.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st


class SentenceTransformerAdapter:
    """
    Adapter class for SentenceTransformer to make it compatible with RAGAS.
    RAGAS expects an embedding model with embed_query method, but SentenceTransformer uses encode.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, **kwargs):
        """Original encode method that passes through to the underlying model."""
        return self.model.encode(texts, **kwargs)

    def embed_query(self, text, **kwargs):
        """
        Adapter method for RAGAS compatibility.
        RAGAS expects embed_query, which we map to encode.
        """
        # For single text input, ensure it's in a list
        if isinstance(text, str):
            result = self.model.encode([text], **kwargs)
            return result[0]  # Return the first (and only) embedding
        else:
            return self.model.encode(text, **kwargs)

    def embed_documents(self, documents, **kwargs):
        """
        Another adapter method for RAGAS compatibility.
        Some frameworks expect embed_documents for document embeddings.
        """
        return self.model.encode(documents, **kwargs)


@st.cache_resource
def load_sentence_transformer():
    """Load and cache the sentence transformer model."""
    return SentenceTransformerAdapter("all-MiniLM-L6-v2")


def create_faiss_index(chunks):
    """Create a FAISS index from text chunks."""
    # Get the model
    model = load_sentence_transformer()

    # Generate embeddings for all chunks
    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.asarray(embeddings).astype("float32"))

    return embeddings, index


def retrieve_relevant_chunks(query, chunks, index, top_k=3):
    """Retrieve the most relevant chunks for a query using FAISS."""
    if not chunks or index is None:
        return []

    # Get the model
    model = load_sentence_transformer()

    # Generate embedding for the query
    query_embedding = model.encode([query])

    # Search the index for similar chunks
    distances, indices = index.search(
        np.asarray(query_embedding).astype("float32"), top_k
    )

    # Get the relevant chunks
    relevant_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]

    return relevant_chunks
