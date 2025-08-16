import src.streamlit_patch as streamlit_patch  # noqa: F401
import streamlit as st
import nltk

nltk.download("punkt_tab")

# Import our modules
from src.utils import (
    get_file_hash,
    extract_text_from_pdf,
    chunk_text,
    save_faiss_data,
    load_faiss_data,
)
from src.embedding import create_faiss_index, retrieve_relevant_chunks
from src.ollama import generate_response
from src.ragas_eval import (
    evaluate_rag_system,
    generate_test_dataset,
    evaluate_with_test_dataset,
)


# Set page configuration
st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")


# Initialize session state variables
def init_session_state():
    """Initialize session state variables."""
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
        st.session_state.ollama_model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Default to meta-llama/llama-4-scout-17b-16e-instruct
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 3
    if "query_submitted" not in st.session_state:
        st.session_state.query_submitted = False
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "test_dataset" not in st.session_state:
        st.session_state.test_dataset = None
    if "test_dataset_results" not in st.session_state:
        st.session_state.test_dataset_results = None


def process_pdf(uploaded_file):
    """Process a PDF file and create embeddings."""
    # Get file content and hash
    file_content = uploaded_file.getvalue()
    file_hash = get_file_hash(file_content)

    # Check if a new file is uploaded or the same file is re-uploaded
    if st.session_state.file_hash != file_hash:
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_hash = file_hash
        st.session_state.processing_complete = False

        # Reset chat history
        st.session_state.chat_history = []

        # Try to load existing FAISS index and chunks
        chunks, embeddings, index = load_faiss_data(file_hash)

        # If data doesn't exist, process the PDF
        if chunks is None:
            status_text = st.empty()
            progress_bar = st.progress(0)

            status_text.text("Extracting text from PDF...")
            progress_bar.progress(10)

            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            progress_bar.progress(30)

            status_text.text("Chunking text...")
            progress_bar.progress(50)

            # Chunk the text
            chunks = chunk_text(pdf_text)
            progress_bar.progress(70)

            status_text.text("Creating FAISS index...")
            progress_bar.progress(80)

            # Create FAISS index
            embeddings, index = create_faiss_index(chunks)
            progress_bar.progress(90)

            # Save FAISS index and chunks
            save_faiss_data(chunks, embeddings, index, file_hash)
            progress_bar.progress(100)

            status_text.text("")
            st.success(f"Processed and indexed {len(chunks)} text chunks")
        else:
            st.success(f"Loaded {len(chunks)} indexed text chunks from cache")

        # Update session state
        st.session_state.chunks = chunks
        st.session_state.faiss_index = index
        st.session_state.processing_complete = True


def handle_user_query(user_query, approach):
    """Process a user query and generate a response."""
    # Skip if this query was already submitted
    if st.session_state.query_submitted:
        st.session_state.query_submitted = False
        return

    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Process the query
    if st.session_state.chunks and st.session_state.faiss_index is not None:
        with st.spinner("Searching document..."):
            # Retrieve relevant chunks using FAISS with the user-selected number of chunks
            relevant_chunks = retrieve_relevant_chunks(
                user_query,
                st.session_state.chunks,
                st.session_state.faiss_index,
                top_k=st.session_state.num_chunks,
            )

            if relevant_chunks:
                # Generate response
                response = generate_response(
                    user_query,
                    relevant_chunks,
                    model_name=st.session_state.ollama_model,
                    approach=approach,
                )

                # Add response to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            else:
                no_info_response = (
                    "No relevant information found. Try rephrasing your question."
                )
                # Add response to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": no_info_response}
                )
    else:
        no_pdf_response = "Please upload a PDF document first."
        # Add response to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": no_pdf_response}
        )

    # Mark this query as submitted to prevent multiple processing
    st.session_state.query_submitted = True
    # Force UI update once to show the new message
    st.rerun()


def display_pdf(file):
    """Display a PDF file in the Streamlit app."""
    # Create a binary stream of the PDF file
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{file}" width="100%" height="600" type="application/pdf"></iframe>
    """
    # Display the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)


def run_ragas_evaluation():
    """Run RAGAS evaluation on the current chat history and chunks."""
    if not st.session_state.chat_history or len(st.session_state.chat_history) < 2:
        st.sidebar.error("Need at least one question-answer pair for evaluation.")
        return

    if not st.session_state.chunks:
        st.sidebar.error("No document chunks available for evaluation.")
        return

    with st.spinner("Running RAGAS evaluation..."):
        # Run evaluation
        results = evaluate_rag_system(
            st.session_state.chat_history, st.session_state.chunks
        )

        # Store results in session state
        st.session_state.evaluation_results = results

        # Show success message
        if "error" not in results:
            st.sidebar.success("Evaluation complete!")
        else:
            st.sidebar.error(results["error"])


def generate_ragas_test_dataset(testset_size):
    """Generate a test dataset using RAGAS."""
    if not st.session_state.chunks:
        st.sidebar.error("No document chunks available for test dataset generation.")
        return

    with st.spinner("Generating test dataset..."):
        # Generate test dataset
        dataset = generate_test_dataset(
            st.session_state.chunks, testset_size=testset_size
        )

        # Store dataset in session state
        st.session_state.test_dataset = dataset

        # Show success message
        if dataset is not None:
            st.sidebar.success(f"Generated {testset_size} test questions!")
        else:
            st.sidebar.error("Failed to generate test dataset. Check logs for details.")


def process_test_dataset():
    """Process the test dataset and evaluate the system."""
    if st.session_state.test_dataset is None:
        st.sidebar.error("No test dataset available. Generate one first.")
        return

    with st.spinner("Evaluating system with test dataset..."):
        # Define a function that takes a question and returns an answer and contexts
        def rag_function(question):
            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(
                question,
                st.session_state.chunks,
                st.session_state.faiss_index,
                top_k=st.session_state.num_chunks,
            )

            # Generate response
            response = generate_response(
                question,
                relevant_chunks,
                model_name=st.session_state.ollama_model,
                approach="zero-shot",  
            )

            return response, relevant_chunks

        # Evaluate the system
        results = evaluate_with_test_dataset(
            st.session_state.test_dataset, rag_function
        )

        # Store results in session state
        st.session_state.evaluation_results = results

        # Show success message
        if "error" not in results:
            st.sidebar.success("Evaluation complete!")
        else:
            st.sidebar.error(results["error"])


def render_sidebar():
    """Render the sidebar with settings."""
    # Advanced settings
    st.sidebar.subheader("Settings")

    # Number of chunks to retrieve
    num_chunks = st.sidebar.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=st.session_state.num_chunks,
        help="More chunks provide more context but may slow down response generation",
    )

    # Update session state with selected number of chunks
    if num_chunks != st.session_state.num_chunks:
        st.session_state.num_chunks = num_chunks

    st.sidebar.info(f"""
    This application uses Groq API with the meta-llama/llama-4-scout-17b-16e-instruct model for generating responses.
    """)

    # RAGAS Evaluation section
    st.sidebar.subheader("RAGAS Evaluation")

    # Add explanation about RAGAS
    st.sidebar.info("""
    RAGAS is a framework for evaluating Retrieval-Augmented Generation (RAG) systems.
    It provides metrics to assess the quality of retrieval and generation components.
    
    Note: RAGAS evaluation uses the Groq API with the meta-llama/llama-4-scout-17b-16e-instruct model.
    """)

    # Create tabs for different evaluation methods
    eval_tabs = st.sidebar.tabs(["Chat Evaluation", "Test Dataset"])

    # Tab 1: Evaluate using chat history
    with eval_tabs[0]:
        st.write("Evaluate using chat history")

        # Add button to run evaluation
        eval_button = st.button(
            "Run RAGAS Evaluation",
            disabled=not st.session_state.processing_complete
            or len(st.session_state.chat_history) < 2,
            help="Requires at least one question-answer pair in the chat history",
        )

        if eval_button:
            run_ragas_evaluation()

        # Display evaluation results if available
        if st.session_state.evaluation_results:
            st.subheader("Evaluation Results")

            results = st.session_state.evaluation_results

            if "error" in results:
                st.error(results["error"])
            else:
                # Create a table to display the results
                for metric, score in results.items():
                    st.metric(label=metric.replace("_", " ").title(), value=score)

    # Tab 2: Generate and evaluate using test dataset
    with eval_tabs[1]:
        st.write("Generate and evaluate using test dataset")

        # Add input for test dataset size
        testset_size = st.number_input(
            "Number of test questions to generate",
            min_value=1,
            max_value=50,
            value=10,
            help="More questions provide better evaluation but take longer to generate and process",
        )

        # Add button to generate test dataset
        gen_dataset_button = st.button(
            "Generate Test Dataset",
            disabled=not st.session_state.processing_complete,
            help="Generates test questions based on the document content",
        )

        if gen_dataset_button:
            generate_ragas_test_dataset(testset_size)

        # Add button to evaluate using test dataset
        eval_dataset_button = st.button(
            "Evaluate with Test Dataset",
            disabled=st.session_state.test_dataset is None,
            help="Evaluates the system using the generated test dataset",
        )

        if eval_dataset_button:
            process_test_dataset()

        # Display test dataset if available
        if st.session_state.test_dataset is not None:
            with st.expander("View Test Dataset"):
                # Convert to pandas DataFrame for display
                test_df = st.session_state.test_dataset.to_pandas()
                # Display only the questions
                st.dataframe(test_df[["question"]])

        # Display test dataset evaluation results if available
        if st.session_state.evaluation_results is not None:
            st.subheader("Test Dataset Evaluation Results")
            evaluation_df = st.session_state.evaluation_results
            st.dataframe(evaluation_df)

    # Add explanation of metrics
    with st.sidebar.expander("Metrics Explanation"):
        st.write("""
        - **Faithfulness**: Measures if the generated answer is factually consistent with the retrieved context
        - **Answer Relevancy**: Measures if the answer is relevant to the question
        - **Context Relevancy**: Measures if the retrieved context is relevant to the question
        - **Context Precision**: Measures the precision of the retrieved context
        - **Context Recall**: Measures the recall of the retrieved context
        - **Harmfulness**: Measures if the generated answer contains harmful content
        """)


def main():
    """Main application function."""
    # Initialize session state
    init_session_state()

    # Main application interface
    st.title("PDF Q&A Assistant")

    # Render sidebar
    render_sidebar()

    # Two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            process_pdf(uploaded_file)

            # Approach selection
            # approach = st.radio("Select method approach:", ("zero-shot", "few-shot"))

            # Status indicator
            if st.session_state.file_name:
                if st.session_state.processing_complete:
                    st.success("PDF indexed and ready for questions!")
                else:
                    st.warning("PDF is still being processed. Please wait.")

            # Display PDF viewer when a file is uploaded and processed
            if st.session_state.processing_complete:
                import base64

                base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                display_pdf(base64_pdf)

    with col2:
        st.subheader("Chat with your PDF")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f"**You:** {chat['content']}")
                else:
                    st.markdown(f"**Assistant:** {chat['content']}")

        # User input form to prevent resubmission on refresh
        with st.form("query_form", clear_on_submit=True):
            # Query input - disable if processing is not complete
            user_query = st.text_input(
                "Ask a question about your PDF:",
                disabled=not st.session_state.processing_complete,
            )
            submit_button = st.form_submit_button("Submit")

            if submit_button and user_query and st.session_state.processing_complete:
                handle_user_query(user_query, approach)


if __name__ == "__main__":
    main()
