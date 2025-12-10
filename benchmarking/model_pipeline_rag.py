"""
RAG-enabled model pipeline.
"""

import sys
import os
from typing import Dict, List, Any
from copy import deepcopy

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarking.openrouter_client import OpenRouterClient
from RAG.retriever import RAGRetriever

class EvaluatedModelRAG:
    """Generate answers for an exam using a selected LLM with RAG support."""

    def __init__(self, model_name: str, retriever: RAGRetriever, top_k: int = 3):
        self._client = OpenRouterClient(model=model_name)
        self._model_name = model_name
        self._retriever = retriever
        self._top_k = top_k

    def generate_answers(self, answer_sheet: Dict) -> Dict:
        """Generate answers iteratively question by question with RAG context."""
        # Create a deep copy to fill with answers
        filled_sheet = deepcopy(answer_sheet)
        
        system_prompt = (
            "You are taking a Swiss professional exam.\n"
            "You will be presented with questions one by one.\n"
            "You will also receive 'RETRIEVED CONTEXT' from a database.\n"
            "Read each question carefully and provide the answer.\n\n"
            "INSTRUCTIONS:\n"
            "- Use the RETRIEVED CONTEXT if it helps answer the question.\n"
            "- NOTE: The retrieved context might be irrelevant. If so, ignore it and answer based on your knowledge.\n"
            "- Provide ONLY the answer text. Do not include 'Answer:', 'Here is the answer', or reasoning unless explicitly asked.\n"
            "- For multiple choice questions:\n"
            "  * Provide ONLY the answer (letter/number/option/text)\n"
            "  * If multiple answers are correct, provide all of them (e.g., 'a, c' or 'Bundesrat, Kantone')\n"
            "- For open questions: provide concise, accurate answers\n"
            "- Keep original language (do NOT translate)\n"
        )

        # Initialize chat history
        initial_messages = [{"role": "system", "content": system_prompt}]

        def process_questions(questions: List[Dict], history: List[Dict], parent_text: str = ""):
            for q in questions:
                # Create a fresh context for this question/group
                current_history = list(history)

                # Handle parent context if present (questions with subquestions)
                if "subquestions" in q and q["subquestions"]:
                    q_text = q.get('question_text', '')
                    
                    context_msg = f"QUESTION CONTEXT:\n{q_text}"
                    current_history.append({"role": "user", "content": context_msg})
                    
                    # Add a dummy assistant response to maintain conversation flow
                    current_history.append({"role": "assistant", "content": "Understood."})
                    
                    # Process subquestions, passing down the accumulated parent text
                    new_parent_text = f"{parent_text} {q_text}".strip()
                    process_questions(q["subquestions"], current_history, parent_text=new_parent_text)
                    continue

                # Handle leaf questions (answerable)
                if "answer_field" in q:
                    q_text = q.get("question_text", "")
                    
                    # Construct the full query using parent context + current question
                    query_text = f"{parent_text} {q_text}".strip()
                    
                    # Retrieve context using the combined query
                    retrieved_chunks = self._retriever.retrieve(query_text, k=self._top_k)
                    
                    # Store retrieved chunks
                    q["retrieved_chunks"] = retrieved_chunks
                    
                    # Format context string
                    context_str = ""
                    if retrieved_chunks:
                        context_str = "\n\nRETRIEVED CONTEXT:\n"
                        for i, chunk in enumerate(retrieved_chunks):
                            context_str += f"--- Chunk {i+1} (Source: {chunk['source']}) ---\n{chunk['text']}\n"
                    
                    current_history.append({"role": "user", "content": f"QUESTION:\n{q_text}{context_str}"})
                    
                    # Retry loop for empty answers
                    max_retries = 3
                    for attempt in range(max_retries):
                        response = self._client.chat(current_history)
                        cleaned_response = response.strip()
                        
                        if cleaned_response:
                            q["answer_field"] = cleaned_response
                            break
                        
                        if attempt < max_retries - 1:
                            print(f"Empty response for question (Attempt {attempt+1}/{max_retries}). Retrying...")
                        else:
                            q["answer_field"] = "" # Give up
                    
                    # No need to append to history as we branch off

        # Start processing
        if "questions" in filled_sheet:
            process_questions(filled_sheet["questions"], initial_messages)

        return filled_sheet
