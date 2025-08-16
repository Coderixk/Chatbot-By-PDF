"""
RAGAS evaluation module for the PDF Chatbot application.
"""

import os
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate
from ragas.testset import TestsetGenerator
from ragas.llms import LlamaIndexLLMWrapper
from langchain.docstore.document import Document
from llama_index.llms.groq import Groq

# Import our adapter instead of directly using SentenceTransformer
from src.embedding import load_sentence_transformer


def prepare_evaluation_data(
    chat_history: List[Dict[str, str]], chunks: List[str]
) -> Optional[Dataset]:
    """
    Prepare evaluation data from chat history and retrieved chunks.

    Args:
        chat_history: List of chat messages with 'role' and 'content'
        chunks: List of text chunks from the document

    Returns:
        Dataset object with questions, answers, contexts, and ground_truths
    """
    # Extract question-answer pairs from chat history
    questions = []
    answers = []
    contexts_list = []
    references = []

    # We need at least one question-answer pair
    if len(chat_history) < 2:
        return None

    # Process chat history to extract QA pairs
    for i in range(0, len(chat_history) - 1, 2):
        # Check if we have a user question followed by an assistant answer
        if (
            i + 1 < len(chat_history)
            and chat_history[i]["role"] == "user"
            and chat_history[i + 1]["role"] == "assistant"
        ):
            question = chat_history[i]["content"]
            answer = chat_history[i + 1]["content"]

            # Skip pairs where the answer indicates no information was found
            if (
                "No relevant information found" in answer
                or "Please upload a PDF document first" in answer
            ):
                continue

            questions.append(question)
            answers.append(answer)
            # Ensure chunks are properly formatted as a list of strings for RAGAS
            contexts_list.append(chunks)

            # For reference, join all chunks into a single string
            references.append(" ".join(chunks))

    # If no valid QA pairs were found, return None
    if not questions:
        return None

    # Create a dataset with the extracted data
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truths": answers.copy(),  # Using answers as approximate ground truths
        "reference": references,  # Join chunks into a single string for each QA pair
    }

    return Dataset.from_dict(eval_data)


def get_groq_api_key() -> Optional[str]:
    """Get the Groq API key from environment variable."""
    api_key = st.secrets.GROQ_API_KEY
    if not api_key:
        st.warning(
            "GROQ_API_KEY environment variable not set. RAGAS evaluation requires this to be set."
        )
    return api_key


def run_ragas_evaluation(eval_dataset: Dataset) -> Optional[Dict[str, Any]]:
    """
    Run RAGAS evaluation on the provided dataset.

    Args:
        eval_dataset: Dataset object with questions, answers, contexts, and ground_truths

    Returns:
        Dictionary with evaluation results
    """
    # Get Groq API key
    api_key = get_groq_api_key()
    if not api_key:
        return None

    evaluator_llm = LlamaIndexLLMWrapper(
        Groq(api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
    )

    # Define the metrics to evaluate - explicitly passing the LLM to all metrics
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        context_precision,
        context_recall,
    ]

    try:
        # Run the evaluation with explicit LLM parameter
        results = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=load_sentence_transformer(),  # Explicitly provide embeddings
        )
        return results.to_pandas()
    except Exception as e:
        st.error(f"Error during RAGAS evaluation: {e}")
        return None


def format_evaluation_results(results: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Format evaluation results for display.

    Args:
        results: Dictionary with evaluation results

    Returns:
        Dictionary with formatted results
    """
    if results is None:
        return {
            "error": "Evaluation failed. Please check the logs for more information."
        }

    # Extract the scores from the results
    formatted_results = {}

    # Process each metric
    for metric_name in results.keys():
        # Skip metadata columns
        if metric_name in ["answer", "question", "contexts", "ground_truths"]:
            continue

        # Format the score as a percentage
        score = results[metric_name]
        if isinstance(score, (int, float)):
            formatted_score = f"{score:.2%}"
        else:
            formatted_score = str(score)

        # Add to formatted results
        formatted_results[metric_name] = formatted_score

    return formatted_results


def evaluate_rag_system(
    chat_history: List[Dict[str, str]], chunks: List[str]
) -> Dict[str, str]:
    """
    Evaluate the RAG system using RAGAS metrics.

    Args:
        chat_history: List of chat messages with 'role' and 'content'
        chunks: List of text chunks from the document

    Returns:
        Dictionary with evaluation results
    """
    # Prepare evaluation data
    eval_dataset = prepare_evaluation_data(chat_history, chunks)

    if eval_dataset is None:
        return {
            "error": "Not enough data for evaluation. Please ask at least one question."
        }

    # Run evaluation
    results = run_ragas_evaluation(eval_dataset)

    # Format results
    formatted_results = format_evaluation_results(results)

    return results


def generate_test_dataset(
    chunks: List[str], testset_size: int = 10
) -> Optional[Dataset]:
    """
    Generate a test dataset using RAGAS's TestsetGenerator.

    Args:
        chunks: List of text chunks from the document
        testset_size: Number of test questions to generate

    Returns:
        Dataset object with generated test questions
    """
    # Get Groq API key
    api_key = get_groq_api_key()

    if not api_key:
        st.error(
            "GROQ_API_KEY environment variable not set. Test dataset generation requires this to be set."
        )
        return None

    try:
        # Initialize the Groq LLM
        groq_llm = Groq(
            api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

        # Wrap with LlamaIndexLLMWrapper for RAGAS
        generator_llm = LlamaIndexLLMWrapper(groq_llm)

        # Convert chunks to LangChain Document format
        langchain_docs = [Document(page_content=chunk) for chunk in chunks]

        # Initialize the TestsetGenerator with our embedding model
        embedding_model = load_sentence_transformer()

        try:
            # Try using the proper TestsetGenerator first
            generator = TestsetGenerator(
                llm=generator_llm,
                embedding_model=embedding_model,
            )

            # Generate test dataset
            test_dataset = generator.generate(
                langchain_docs,
                test_size=min(testset_size, len(chunks)),
            )

            return test_dataset

        except Exception as gen_error:
            st.warning(f"TestsetGenerator failed: {gen_error}. Using fallback method.")

            # Fallback: Create a simple dataset manually
            questions = []
            contexts = []

            # Use a subset of chunks to create questions
            num_chunks = min(testset_size, len(chunks))
            for i in range(num_chunks):
                # Create a more meaningful question based on the chunk content
                chunk = chunks[i]
                # Truncate chunk if it's too long
                if len(chunk) > 500:
                    chunk = chunk[:500] + "..."

                # Use the LLM to generate a better question based on the chunk
                try:
                    prompt = f"""Given the following text, generate a specific question that can be answered based on this information:
                    
                    TEXT: {chunk}
                    
                    QUESTION:"""

                    question = groq_llm.complete(prompt).text.strip()
                    # Remove any prefixes like "QUESTION: " that might be in the response
                    question = question.replace("QUESTION:", "").strip()
                    if not question:
                        # Fallback if LLM doesn't generate a proper question
                        question = (
                            f"What information can you provide about: {chunk[:50]}...?"
                        )
                except Exception:
                    # If LLM generation fails, use a simple template
                    question = (
                        f"What information can you provide about: {chunk[:50]}...?"
                    )

                # Add the question and context
                questions.append(question)
                contexts.append([chunk])

            # Create a dataset with the generated questions
            eval_data = {
                "question": questions,
                "contexts": contexts,
                "ground_truths": [""] * len(questions),  # Placeholder for ground truths
            }

            return Dataset.from_dict(eval_data)

    except Exception as e:
        st.error(f"Error during test dataset generation: {e}")
        return None


def evaluate_with_test_dataset(
    test_dataset: Optional[Dataset], rag_function: callable
) -> Dict[str, str]:
    """
    Evaluate the RAG system using a generated test dataset.

    Args:
        test_dataset: Dataset object with generated test questions
        rag_function: Function that takes a question and returns an answer and contexts

    Returns:
        Dictionary with evaluation results
    """
    if test_dataset is None:
        return {"error": "No test dataset provided."}

    if len(test_dataset) == 0:
        return {"error": "Test dataset is empty."}

    try:
        # Extract questions from the test dataset
        questions = test_dataset["question"]

        # Generate answers using the provided RAG function
        answers = []
        contexts_list = []
        references = []

        # Process each question with detailed logging
        for i, question in enumerate(questions):
            try:
                # Get answer and context from the RAG function
                result = rag_function(question)

                # Check if result is properly formatted
                if not isinstance(result, tuple) or len(result) != 2:
                    st.error(
                        f"Error: RAG function returned invalid format for question {i}: {type(result)}"
                    )
                    # Skip this question or use placeholder
                    continue

                answer, contexts = result

                # Validate answer and contexts
                if not isinstance(answer, str):
                    st.error(f"Error: Answer is not a string: {type(answer)}")
                    answer = str(answer)

                if not isinstance(contexts, list):
                    st.error(f"Error: Contexts is not a list: {type(contexts)}")
                    contexts = [str(contexts)]

                # Add to our collections
                answers.append(answer)
                contexts_list.append(contexts)
                references.append(" ".join(contexts) if contexts else "No context")

            except Exception as e:
                st.error(f"Error processing question {i}: {str(e)}")
                import traceback

                st.error(traceback.format_exc())
                # Skip this question
                continue

        # Check if we have any answers
        if len(answers) == 0:
            return {
                "error": "Failed to process any questions. Check the logs for details."
            }

        # Create evaluation dataset
        # Check if ground_truth exists in the dataset
        if "ground_truths" in test_dataset.column_names:
            ground_truths = test_dataset["ground_truths"][
                : len(answers)
            ]  # Match length
        elif "ground_truth" in test_dataset.column_names:
            ground_truths = test_dataset["ground_truth"][: len(answers)]  # Match length
        else:
            # If no ground truth is available, use answers as approximate ground truth
            ground_truths = answers.copy()

        # Ensure all lists have the same length
        min_length = min(
            len(questions),
            len(answers),
            len(contexts_list),
            len(references),
            len(ground_truths),
        )

        eval_data = {
            "question": questions[:min_length],
            "answer": answers[:min_length],
            "contexts": contexts_list[:min_length],
            "ground_truths": ground_truths[:min_length],
            "reference": references[:min_length],
        }

        # Create the dataset
        eval_dataset = Dataset.from_dict(eval_data)

        # Run evaluation with detailed error handling
        try:
            results = run_ragas_evaluation(eval_dataset)
            return results

        except Exception as eval_error:
            return {"error": f"RAGAS evaluation failed: {str(eval_error)}"}

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
