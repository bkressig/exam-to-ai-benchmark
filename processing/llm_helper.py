"""
OpenRouter API integration for LLM-assisted processing.
Self-contained answer sheet approach.
"""

import os
from typing import Optional, Dict, List, Any
import requests
import json
from dotenv import load_dotenv

load_dotenv()


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4.5",
    ):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        # No chunking limits anymore - we process whole documents
        self.max_text_chars_per_chunk = 200000  # Just a safety cap for text block construction
    
    def call_llm_messages(self, messages: List[Dict], model: Optional[str] = None) -> str:
        """Call OpenRouter API with pre-built messages supporting text + images."""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model or self.model,
                "messages": messages,
                "plugins": [
                    {
                        "id": "file-parser",
                        "pdf": {
                            "engine": "native"
                        }
                    }
                ]
            },
            timeout=240
        )
        if not response.ok:
            print(f"Error Response: {response.text}")
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    @staticmethod
    def _clean_json_text(response: str) -> str:
        """Strip markdown fences and surrounding text."""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Convert model output into a Python dict, tolerating minor wrapping text."""
        cleaned = self._clean_json_text(response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first != -1 and last != -1 and last > first:
                snippet = cleaned[first:last + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
        print("  Could not parse JSON response. Returning empty dict.")
        return {}

    @staticmethod
    def _count_answerable_questions(questions: List[Dict]) -> int:
        """Recursively count questions that have answer_field."""
        count = 0
        for q in questions:
            if "subquestions" in q:
                count += OpenRouterClient._count_answerable_questions(q["subquestions"])
            elif "answer_field" in q:
                count += 1
        return count

    @staticmethod
    def _sum_points(questions: List[Dict]) -> float:
        """Recursively sum points from all questions."""
        total = 0.0
        for q in questions:
            if "subquestions" in q:
                total += OpenRouterClient._sum_points(q["subquestions"])
            else:
                try:
                    total += float(q.get("points", 0))
                except (TypeError, ValueError):
                    pass
        return total

    def generate_answer_sheet(self, exam_content: Dict, profession: str, year: str) -> Dict:
        """
        Generate self-contained answer_sheet.json from exam content using LLM.
        
        Extracts questions with full context verbatim from PDF.
        Sends the whole PDF directly to the LLM.
        """
        pdf_files = exam_content.get("pdf_files", [])
        if not pdf_files:
            print("  No exam PDFs supplied, returning empty answer sheet.")
            return {
                "exam_metadata": {
                    "profession": profession,
                    "year": year,
                    "total_questions": 0,
                },
                "questions": []
            }

        system_text = """You are extracting questions from an exam PDF to create a self-contained answer sheet.

YOUR TASK:
Extract ALL questions from the provided exam PDF and create a self-contained answer sheet in JSON format.
The answer sheet must be COMPLETE - no external PDFs will be provided to candidates.

=========== JSON STRUCTURE ===========

{
  "exam_metadata": {
    "profession": "INSERT_PROFESSION_HERE",
    "year": "INSERT_YEAR_HERE"
  },
  "questions": [...]
}

=========== QUESTION STRUCTURE ===========

Each question must have:
- question_id: unique identifier (e.g., "1", "2", "3a", "3b")
- question_text: VERBATIM text from PDF including:
  * ALL relevant context
  * The question itself
  * Answer format instructions
  * For MULTIPLE CHOICE: ALL available options to choose from
- answer_field: "" (empty string for all answerable questions)
- subquestions: (optional) array for grouped/nested questions

Multiple choice questions MUST include all selectable options in question_text! e.g. true/false, a/b/c/d, or full text options.

=========== HIERARCHICAL QUESTIONS ===========

For tables or multi-part questions, use hierarchy:

PARENT question (not directly answerable):
- question_id: "2"
- question_text: shared context + instructions + OPTIONS (for MC tables)
- subquestions: array of subquestions
- NO answer_field

SUBQUESTIONS (leaf nodes):
- question_id: "2a", "2b", "2c"
- question_text: specific item/row text
- answer_field: ""

=========== EXAMPLES ===========

SIMPLE OPEN QUESTION:
{
  "question_id": "1",
  "question_text": "Describe the three main differences between X and Y. Answer in 3-5 sentences.",
  "answer_field": ""
}

SIMPLE MULTIPLE CHOICE:
{
  "question_id": "2",
  "question_text": "Which statement is correct?\\n\\na) First option\\nb) Second option\\nc) Third option\\nd) Fourth option\\n\\nProvide the letter of the correct answer.",
  "answer_field": ""
}

GROUPED MC TABLE (options in parent, items in subquestions):
{
  "question_id": "3",
  "question_text": "Instructions for table-based questions. Multiple answers per row are possible.\\n\\nOptions: Option A, Option B, Option C, Option D",
  "subquestions": [
    {
      "question_id": "3a",
      "question_text": "First task or statement",
      "answer_field": ""
    },
    {
      "question_id": "3b",
      "question_text": "Second task or statement",
      "answer_field": ""
    }
  ]
}

OPEN QUESTION WITH SUBPARTS:
{
  "question_id": "4",
  "question_text": "Context describing scenario XYZ.",
  "subquestions": [
    {
      "question_id": "4a",
      "question_text": "First sub-question about scenario?",
      "answer_field": ""
    },
    {
      "question_id": "4b",
      "question_text": "Second sub-question requiring explanation.",
      "answer_field": ""
    }
  ]
}

=========== IMPORTANT RULES ===========

1. question_text must be VERBATIM from the exam (copy exactly, including all context), keep the original language
2. Multiple choice questions MUST list all available options in question_text
3. Use hierarchy (parent + subquestions) for tables and multi-part questions
4. Parent questions have NO answer_field, only subquestions do
5. All answerable questions (leaf nodes) must have answer_field: ""
6. Extract ALL questions - do not skip any
7. DO NOT include points/scores in this step.
"""

        # Prepare exam content with metadata
        instruction_text = f"""
METADATA:
Profession: {profession}
Year: {year}

Return JSON with self-contained questions:
{{
  "questions": [
    {{
      "question_id": "1",
      "question_text": "Full context and question text verbatim...",
      "answer_field": ""
    }},
    {{
      "question_id": "2",
      "question_text": "Context + instructions for table...",
      "subquestions": [
        {{"question_id": "2a", "question_text": "Item text", "answer_field": ""}},
        {{"question_id": "2b", "question_text": "Item text", "answer_field": ""}}
      ]
    }}
  ]
}}
"""

        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": instruction_text},
        ]

        for pdf_file in pdf_files:
            message_content.append({
                "type": "file",
                "file": {
                    "filename": pdf_file["filename"],
                    "file_data": pdf_file["data_url"]
                }
            })

        print(f"[AnswerSheet] Sending {len(pdf_files)} PDF file(s) (whole document)")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": message_content}
        ]

        response = self.call_llm_messages(messages)
        response_data = self._parse_json_response(response)
        collected_questions = response_data.get("questions", [])
        if not isinstance(collected_questions, list):
            collected_questions = []

        # Ensure answer_field exists for leaf questions
        for question in collected_questions:
            if "subquestions" not in question:
                question.setdefault("answer_field", "")

        total_questions = self._count_answerable_questions(collected_questions)
        
        answer_sheet = {
            "exam_metadata": {
                "profession": profession,
                "year": year,
                "total_questions": total_questions,
            },
            "questions": collected_questions
        }

        return answer_sheet
    
    def generate_solution_sheet(
        self,
        exam_content: Dict,
        solution_content: Dict,
        answer_sheet: Dict,
        profession: str,
        year: str
    ) -> Dict:
        """
        Generate solution_sheet.json based on answer_sheet structure.
        
        Uses solution PDF to fill in solution_field, grading_criteria AND points.
        Maintains EXACT same structure as answer_sheet.
        """
        pdf_files = solution_content.get("pdf_files", [])

        if not pdf_files:
            print("  No solution PDFs supplied, returning template without filled solutions.")
            return self._create_empty_solution_sheet(answer_sheet, profession, year)

        # Serialize the answer_sheet to JSON for the prompt
        answer_sheet_json = json.dumps(answer_sheet, ensure_ascii=False, indent=2)

        system_text = """You are extracting solutions from an exam solution PDF.

YOUR TASK:
Take the provided ANSWER SHEET JSON and add `solution_field`, `grading_criteria`, and `points` to each question based on the solution PDF.
Maintain the EXACT same structure as the provided answer sheet JSON.

=========== OUTPUT FORMAT ===========

Return the FULL JSON with the following fields added to each question/subquestion:
- solution_field: The correct answer (verbatim from solution PDF)
- grading_criteria: How to award points (from solution PDF or reasonably inferred)
- points: The maximum points achievable for this question (number)

Example for simple question:
{
  "question_id": "1",
  "question_text": "...",
  "answer_field": "",
  "solution_field": "The three main differences are: 1) ..., 2) ..., 3) ...",
  "grading_criteria": "1 point per correct difference (max 3 points)",
  "points": 3
}

=========== IMPORTANT RULES ===========

1. Keep EXACT same structure as the provided answer sheet (same question_ids, same hierarchy)
2. Extract solutions VERBATIM from solution PDF
3. Extract or infer reasonable grading criteria
4. Extract points for each question. For parent questions, sum the points of subquestions.
5. Keep original language (do not translate)
6. Return ONLY valid JSON
"""

        # Metadata for context
        metadata_text = f"""METADATA:
Profession: {profession}
Year: {year}

=========== INPUT DATA ===========

Here is the ANSWER SHEET JSON structure you must fill:
{answer_sheet_json}
"""

        instruction_text = """
Return the FULL JSON with solutions filled in.
"""

        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": metadata_text},
            {"type": "text", "text": instruction_text},
        ]

        for pdf_file in pdf_files:
            message_content.append({
                "type": "file",
                "file": {
                    "filename": pdf_file["filename"],
                    "file_data": pdf_file["data_url"]
                }
            })

        print(f"[SolutionSheet] Sending {len(pdf_files)} PDF file(s) (whole document)")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": message_content}
        ]

        response = self.call_llm_messages(messages)
        response_data = self._parse_json_response(response)
        
        # Validate response
        if "questions" in response_data:
            return response_data
        else:
            print("  Model response did not contain 'questions' key. Returning empty template.")
            return self._create_empty_solution_sheet(answer_sheet, profession, year)

    def _create_empty_solution_sheet(self, answer_sheet: Dict, profession: str, year: str) -> Dict:
        """Helper to create solution sheet structure with empty fields."""
        # Copy metadata from answer sheet
        base_metadata = dict(answer_sheet.get("exam_metadata") or {})
        if not base_metadata:
            base_metadata = {
                "profession": profession,
                "year": year,
            }

        # Deep copy answer sheet structure
        def copy_question_structure(questions: List[Dict]) -> List[Dict]:
            result = []
            for q in questions:
                new_q = dict(q)
                new_q.setdefault("solution_field", "")
                new_q.setdefault("grading_criteria", "")
                # Points will be filled by LLM
                
                if "subquestions" in q:
                    new_q["subquestions"] = copy_question_structure(q["subquestions"])
                
                result.append(new_q)
            return result

        return {
            "exam_metadata": base_metadata,
            "questions": copy_question_structure(answer_sheet.get("questions", []))
        }
