A Streamlit application that allows users to chat with their PDF documents using Groq API for generating responses.

## Features

- Upload and process PDF documents
- Extract text and create embeddings for semantic search
- Use FAISS for efficient similarity search
- Generate responses using Groq API's language models
- PDF viewer for uploaded documents
- Docker support for easy deployment

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

## Installation

### Using Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed
2. Clone this repository
3. Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

4. Run the application using Docker Compose:

```bash
docker-compose up -d
```

### Manual Installation

1. Install Python 3.10+
2. Get a Groq API key from https://console.groq.com/
3. Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

4. Install the Python dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/) :

```bash
uv sync
```

5. Run the application:

```bash
uv run streamlit run app.py
```

## Usage

1. Open the application in your browser (http://localhost:8501)
2. Upload a PDF document
3. Wait for the document to be processed and indexed
4. The PDF will be displayed in the left panel for easy reference
5. Ask questions about the content of the document in the right panel
6. The application will retrieve relevant chunks from the document and generate a response using Groq API

## Configuration

You can configure the following settings in the application:

- **Model**: Select the language model to use for generating responses
- **Number of Chunks**: Adjust the number of chunks to retrieve for each query

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required for generating responses)
- `OPENAI_API_KEY`: Your OpenAI API key (required for RAGAS evaluation)
