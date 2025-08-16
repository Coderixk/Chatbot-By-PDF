import os
import requests
import streamlit as st


# List of available Groq models
AVAILABLE_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


def get_groq_api_key():
    """Get the Groq API key from environment variable."""
    api_key = st.secrets.GROQ_API_KEY
    if not api_key:
        st.warning("GROQ_API_KEY environment variable not set. Using demo mode.")
    return api_key


def generate_response(
    query,
    relevant_chunks,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    approach="zero-shot",
):
    """Generate a response using Groq API."""
    combined_context = " ".join(relevant_chunks)

    # Create prompt based on the approach
    if approach == "zero-shot":
        # Zero-shot prompt
        prompt = f"""Based on the following context, answer the question concisely:
        
        Context: {combined_context}
        
        Question: {query}
        
        Answer:"""
    else:  # few-shot
        # Few-shot prompt with examples
        prompt = f"""Based on the following context, answer the question concisely. Here are some examples:

        Example 1:
        Context: The Declaration of Independence was signed in 1776. It was primarily written by Thomas Jefferson.
        Question: When was the Declaration of Independence signed?
        Answer: The Declaration of Independence was signed in 1776.

        Example 2:
        Context: Python is a high-level programming language known for its readability and simplicity. It was created by Guido van Rossum and released in 1991.
        Question: Who created Python?
        Answer: Python was created by Guido van Rossum.

        Example 3:
        Context: The Great Barrier Reef is the world's largest coral reef system. It is located off the coast of Queensland, Australia.
        Question: Where is the Great Barrier Reef located?
        Answer: The Great Barrier Reef is located off the coast of Queensland, Australia.

        Now, please answer the following question based on the provided context:
        
        Context: {combined_context}
        
        Question: {query}
        
        Answer:"""

    try:
        # Get Groq API key
        api_key = get_groq_api_key()

        if not api_key:
            return "Error: GROQ_API_KEY environment variable not set. Please set it to use the Groq API."

        # Call Groq API with the selected model
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "Sorry, I couldn't generate a response.")
            )
        else:
            st.error(f"Error calling Groq API: {response.status_code}")
            return f"Error: Could not get a response from the language model. Status code: {response.status_code}. Details: {response.text}"

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Groq API: {e}")
        return "Error: Could not connect to Groq API. Please check your internet connection."
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return f"An unexpected error occurred: {str(e)}"


def get_ollama_models():
    """Get list of available Groq models."""
    # For Groq, we'll just return the predefined list of models
    return AVAILABLE_MODELS
