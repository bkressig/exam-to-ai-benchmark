"""High level helpers to generate model answers and grade them."""

import json
from typing import Dict, List, Any
from copy import deepcopy

from openrouter_client import OpenRouterClient


class EvaluatedModel:
    """Generate answers for an exam using a selected LLM."""

    def __init__(self, model_name: str):
        self._client = OpenRouterClient(model=model_name)
        self._model_name = model_name

    def generate_answers(self, answer_sheet: Dict) -> Dict:
        """Generate answers iteratively question by question."""
        # Create a deep copy to fill with answers
        filled_sheet = deepcopy(answer_sheet)
        
        system_prompt = (
            "You are taking a Swiss professional exam.\n"
            "You will be presented with questions one by one.\n"
            "Read each question carefully and provide the answer.\n\n"
            "INSTRUCTIONS:\n"
            "- Provide ONLY the answer text. Do not include 'Answer:', 'Here is the answer', or reasoning unless explicitly asked.\n"
            "- For multiple choice questions:\n"
            "  * Provide ONLY the answer (letter/number/option/text)\n"
            "  * If multiple answers are correct, provide all of them (e.g., 'a, c' or 'Bundesrat, Kantone')\n"
            "- For open questions: provide concise, accurate answers\n"
            "- Keep original language (do NOT translate)\n"
        )

        # Initialize chat history
        initial_messages = [{"role": "system", "content": system_prompt}]

        def process_questions(questions: List[Dict], history: List[Dict]):
            for q in questions:
                # Create a fresh context for this question/group to avoid context pollution
                current_history = list(history)

                # Handle parent context if present (questions with subquestions)
                if "subquestions" in q and q["subquestions"]:
                    # Send parent context to model but don't expect an answer
                    context_msg = f"CONTEXT FOR FOLLOWING QUESTIONS:\n{q.get('question_text', '')}"
                    current_history.append({"role": "user", "content": context_msg})
                    
                    # We don't get a response for context
                    # force a dummy response to keep chat alternation valid if API requires it.
                    # OpenRouter/most APIs require alternation.
                    current_history[-1]["content"] += "\n\nPlease confirm you understood the context by replying 'Understood'."
                    
                    response = self._client.chat(current_history)
                    current_history.append({"role": "assistant", "content": response})
                    
                    # Process subquestions
                    process_questions(q["subquestions"], current_history)
                    continue

                # Handle leaf questions (answerable)
                if "answer_field" in q:
                    q_text = q.get("question_text", "")
                    current_history.append({"role": "user", "content": f"QUESTION:\n{q_text}"})
                    
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
                    
                    # No need to append to history as we branch off for each question

        # Start processing
        if "questions" in filled_sheet:
            process_questions(filled_sheet["questions"], initial_messages)

        return filled_sheet


class JudgeModel:
    """Grade a filled answer sheet using a solution sheet as reference."""

    def __init__(self, model_name: str):
        self._client = OpenRouterClient(model=model_name)
        self._model_name = model_name

    def grade(self, model_answers: Dict, solution_sheet: Dict) -> Dict:
        """Grade model answers iteratively against solution sheet."""
        # Create a deep copy to fill with grading
        graded_sheet = deepcopy(model_answers)
        
        # Build a map of solutions for easy lookup
        solution_map = {}
        def map_solutions(questions):
            for q in questions:
                if "question_id" in q:
                    solution_map[q["question_id"]] = q
                if "subquestions" in q:
                    map_solutions(q["subquestions"])
        
        if "questions" in solution_sheet:
            map_solutions(solution_sheet["questions"])

        system_prompt = (
            "You are grading a Swiss professional exam.\n"
            "You will be presented with questions, candidate answers, and official solutions one by one.\n\n"
            "INSTRUCTIONS:\n"
            "- Compare the candidate's answer with the solution.\n"
            "- Assign 'awarded_points' (0 to max points).\n"
            "- Provide brief 'feedback' (1-3 sentences).\n"
            "- For MULTIPLE CHOICE:\n"
            "  * Award either full points or 0 points (NO partial credit).\n"
            "  * ALL correct options must be selected.\n"
            "- OUTPUT FORMAT: JSON snippet ONLY\n"
            "  {\"points\": <number>, \"feedback\": \"<string>\"}\n"
            "- Keep original language (do NOT translate)\n"
        )

        # Initialize chat history
        initial_messages = [{"role": "system", "content": system_prompt}]

        def process_grading(questions: List[Dict], history: List[Dict]):
            for q in questions:
                qid = q.get("question_id")
                
                # Create a fresh context for this question/group
                current_history = list(history)

                # Handle parent context
                if "subquestions" in q and q["subquestions"]:
                    context_msg = f"CONTEXT FOR FOLLOWING QUESTIONS:\n{q.get('question_text', '')}"
                    current_history.append({"role": "user", "content": context_msg})
                    current_history[-1]["content"] += "\n\nReply 'Understood' to proceed."
                    
                    response = self._client.chat(current_history)
                    current_history.append({"role": "assistant", "content": response})
                    
                    process_grading(q["subquestions"], current_history)
                    continue

                # Handle leaf questions
                if "answer_field" in q:
                    sol_q = solution_map.get(qid, {})
                    solution_text = sol_q.get("solution_field", "N/A")
                    grading_criteria = sol_q.get("grading_criteria", "N/A")
                    max_points = sol_q.get("points", 0)
                    
                    prompt = (
                        f"QUESTION: {q.get('question_text', '')}\n\n"
                        f"CANDIDATE ANSWER: {q.get('answer_field', '')}\n\n"
                        f"OFFICIAL SOLUTION: {solution_text}\n"
                        f"GRADING CRITERIA: {grading_criteria}\n"
                        f"MAX POINTS: {max_points}\n\n"
                        "Grade this answer. Return JSON."
                    )
                    
                    # Retry loop for malformed JSON
                    max_retries = 3
                    for attempt in range(max_retries):
                        # Prepare messages for this attempt
                        messages = list(current_history)
                        messages.append({"role": "user", "content": prompt})
                        
                        response = self._client.chat(messages)
                        
                        # Parse response
                        try:
                            # Clean markdown code blocks if present
                            clean_response = response.strip()
                            if clean_response.startswith("```json"):
                                clean_response = clean_response[7:]
                            if clean_response.startswith("```"):
                                clean_response = clean_response[3:]
                            if clean_response.endswith("```"):
                                clean_response = clean_response[:-3]
                                
                            result = json.loads(clean_response.strip())
                            
                            # Handle case where points might be None or missing
                            points_val = result.get("points")
                            if points_val is None:
                                points_val = 0
                                
                            q["awarded_points"] = float(points_val)
                            q["feedback"] = result.get("feedback", "")
                            break # Success, exit retry loop
                        except (json.JSONDecodeError, ValueError) as e:
                            if attempt < max_retries - 1:
                                print(f"Error parsing grading for Q{qid} (Attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                                continue
                            else:
                                print(f"Error parsing grading for Q{qid}: {e}")
                                q["awarded_points"] = 0
                                q["feedback"] = "Error parsing judge response"

        if "questions" in graded_sheet:
            process_grading(graded_sheet["questions"], initial_messages)

        return graded_sheet

