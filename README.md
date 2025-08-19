# Chatbot-By-PDF
Project Skripsi

A Streamlit application that allows users to chat with their PDF documents using Groq API for generating responses.

## Features

- Upload and process PDF documents
- Extract text and create embeddings for semantic search
- Use FAISS for efficient similarity search
- Generate responses using Groq API's language models
- PDF viewer for uploaded documents

## Project Structure

```
.
├── app.py                # Main Streamlit application
├── embedding.py          # Embedding and FAISS-related functions
├── ollama.py             # Groq API integration
├── utils.py              # Utility functions for PDF processing
├── streamlit_patch.py    # Patch for Streamlit to fix PyTorch compatibility
├── Dockerfile            # Dockerfile for building the application
├── docker-compose.yml    # Docker Compose configuration
├── pyproject.toml        # Project dependencies
```

## Requirements

- Python 3.10+
- Streamlit
- PyMuPDF (for PDF processing)
- NLTK (for text processing)
- FAISS (for similarity search)
- Sentence Transformers (for embeddings)
- Groq API key (for generating responses)


