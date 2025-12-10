"""
Main orchestrator for benchmarking exams.
Generates model answers and graded outputs using self-contained answer sheets.
"""

import yaml
import json
import math
from pathlib import Path
from typing import Dict
from datetime import datetime
from copy import deepcopy

from exam_repository import ProcessedExamRepository
from model_pipeline import EvaluatedModel, JudgeModel
from json_utils import read_json, write_json, sanitize_name


class BenchmarkRunner:
    """Run benchmarking for processed exams."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config["raw_data_dir"]).resolve()
        self.processed_data_dir = Path(self.config["processed_data_dir"]).resolve()
        self.benchmarked_data_dir = self.processed_data_dir.parent / "benchmarked"
        self.benchmarked_data_dir.mkdir(parents=True, exist_ok=True)

        benchmarking_cfg = self.config.get("benchmarking", {})
        self.professions_filter = benchmarking_cfg.get("professions")
        self.exam_numbers_filter = benchmarking_cfg.get("exam_numbers")
        self.model_names = benchmarking_cfg.get("models", [])
        self.judge_names = benchmarking_cfg.get("judges", [])
        self.num_judge_runs = benchmarking_cfg.get("num_judge_runs", 1)

        self.repository = ProcessedExamRepository(
            self.processed_data_dir,
            self.raw_data_dir,
            self.professions_filter,
            self.exam_numbers_filter
        )

        # Instantiate models and judges
        self.evaluated_models = {name: EvaluatedModel(name) for name in self.model_names}
        self.judge_models = {name: JudgeModel(name) for name in self.judge_names}

    def run(self) -> Dict:
        processed_exams = self.repository.list_latest_exams()
        if not processed_exams:
            print("No processed exams found for benchmarking")
            return {}

        results: Dict = {}

        for exam in processed_exams:
            print(f"\n{'=' * 60}")
            print(f"Benchmarking: {exam.exam_id}")
            print(f"Processed source: {exam.processed_dir}")

            answer_path = exam.processed_dir / "answer_sheet.json"
            solution_path = exam.processed_dir / "solution_sheet.json"

            if not answer_path.exists() or not solution_path.exists():
                print("  ⚠ Missing answer_sheet.json or solution_sheet.json – skipping exam")
                continue

            answer_sheet = read_json(answer_path)
            solution_sheet = read_json(solution_path)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_results: Dict = {}

            for model_name, evaluated_model in self.evaluated_models.items():
                print(f"  → Generating answers with model {model_name}")
                model_answers = evaluated_model.generate_answers(answer_sheet)
                model_answers = self._ensure_answer_structure(answer_sheet, model_answers)
                self._enrich_metadata(model_answers, answer_sheet, model_name, exam, timestamp)

                model_dir = self._build_model_dir(exam, timestamp, model_name)
                answers_path = model_dir / "model_answers.json"
                write_json(answers_path, model_answers)

                graded_locations: Dict = {}
                for judge_name, judge_model in self.judge_models.items():
                    # Prepare the container for all runs
                    graded_answers = deepcopy(model_answers)
                    self._init_judgments(graded_answers)
                    
                    # Enrich metadata for the grading file
                    self._enrich_grading_metadata(graded_answers, model_name, judge_name, exam, timestamp)

                    for run_id in range(1, self.num_judge_runs + 1):
                        print(f"    → Grading with judge {judge_name} (Run {run_id}/{self.num_judge_runs})")
                        
                        # Perform one grading run
                        # This returns a sheet with 'awarded_points' and 'feedback' on questions
                        run_result = judge_model.grade(model_answers, solution_sheet)
                        
                        # Extract results and append to judgments list
                        self._collect_judgments(graded_answers, run_result, judge_name, run_id)

                    self._aggregate_judgments(graded_answers)
                    self._inject_max_points(graded_answers, solution_sheet)
                    self._update_grading_summary(graded_answers)

                    judge_dir = model_dir / f"judge={sanitize_name(judge_name)}"
                    graded_path = judge_dir / "graded_answers.json"
                    write_json(graded_path, graded_answers)
                    graded_locations[judge_name] = str(graded_path)

                model_results[model_name] = {
                    "model_answers": str(answers_path),
                    "graded_answers": graded_locations,
                }

            results[exam.exam_id] = model_results

        print("\n" + "="*60)
        print("Benchmarking Complete!")
        print(f"Results saved to: {self.benchmarked_data_dir}")
        print("="*60)
        return results

    def _build_model_dir(self, exam, timestamp: str, model_name: str) -> Path:
        # Structure: benchmarked/profession/number/processed_ts/model/benchmark_ts
        base = self.benchmarked_data_dir / exam.profession / exam.exam_number
        base = base / exam.timestamp
        
        model_dir_name = f"model={sanitize_name(model_name)}"
        model_dir = base / model_dir_name / timestamp
        
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @staticmethod
    def _ensure_answer_structure(reference: Dict, candidate: Dict) -> Dict:
        """Recursively ensure answer structure matches reference (hierarchical with subquestions)."""
        if not candidate:
            fallback = deepcopy(reference)
            fallback.setdefault("exam_metadata", {})
            fallback.setdefault("questions", [])
            fallback["exam_metadata"]["error"] = "Model answer generation failed"
            return fallback

        def merge_questions(ref_qs, cand_qs):
            """Recursively merge candidate questions with reference structure."""
            ref_map = {q.get("question_id"): q for q in ref_qs}
            cand_map = {q.get("question_id"): q for q in cand_qs}
            result = []

            for qid, ref_q in ref_map.items():
                merged = deepcopy(ref_q)
                if qid in cand_map:
                    cand_q = cand_map[qid]
                    # Copy answer_field if present
                    if "answer_field" in cand_q:
                        merged["answer_field"] = cand_q["answer_field"]
                    # Recursively merge subquestions if present
                    if "subquestions" in ref_q:
                        ref_subs = ref_q["subquestions"]
                        cand_subs = cand_q.get("subquestions", [])
                        merged["subquestions"] = merge_questions(ref_subs, cand_subs)
                else:
                    # Missing answer - set empty answer_field if leaf question
                    if "answer_field" in merged:
                        merged["answer_field"] = ""
                    # Recursively handle subquestions
                    if "subquestions" in merged:
                        merged["subquestions"] = merge_questions(merged["subquestions"], [])
                
                result.append(merged)
            
            return result

        candidate.setdefault("questions", [])
        candidate["questions"] = merge_questions(
            reference.get("questions", []),
            candidate.get("questions", [])
        )
        candidate.setdefault("exam_metadata", deepcopy(reference.get("exam_metadata", {})))
        return candidate

    @staticmethod
    def _init_judgments(sheet: Dict) -> None:
        """Recursively initialize 'judgments' list on all questions."""
        def recurse(questions):
            for q in questions:
                q["judgments"] = []
                # Remove legacy fields if they exist from copying
                q.pop("awarded_points", None)
                q.pop("feedback", None)
                
                if "subquestions" in q:
                    recurse(q["subquestions"])
        
        if "questions" in sheet:
            recurse(sheet["questions"])

    @staticmethod
    def _collect_judgments(accumulator: Dict, run_result: Dict, judge_name: str, run_id: int) -> None:
        """Recursively extract grades from run_result and append to accumulator."""
        # Build map of run_result questions
        run_map = {}
        def map_qs(qs):
            for q in qs:
                if "question_id" in q:
                    run_map[q["question_id"]] = q
                if "subquestions" in q:
                    map_qs(q["subquestions"])
        
        if "questions" in run_result:
            map_qs(run_result["questions"])

        # Traverse accumulator and pull data
        def recurse(questions):
            for q in questions:
                qid = q.get("question_id")
                if qid in run_map:
                    run_q = run_map[qid]
                    # Only add judgment if points were awarded (meaning it's a graded leaf)
                    if "awarded_points" in run_q:
                        judgment = {
                            "judge_name": judge_name,
                            "run_id": run_id,
                            "awarded_points": run_q["awarded_points"],
                            "feedback": run_q.get("feedback", "")
                        }
                        q["judgments"].append(judgment)
                
                if "subquestions" in q:
                    recurse(q["subquestions"])

        if "questions" in accumulator:
            recurse(accumulator["questions"])

    @staticmethod
    def _aggregate_judgments(sheet: Dict) -> None:
        """Recursively calculate average awarded_points from judgments."""
        def recurse(questions):
            for q in questions:
                if "judgments" in q and q["judgments"]:
                    points = [j["awarded_points"] for j in q["judgments"] if isinstance(j.get("awarded_points"), (int, float))]
                    if points:
                        q["awarded_points"] = sum(points) / len(points)
                    else:
                        q["awarded_points"] = 0
                
                if "subquestions" in q:
                    recurse(q["subquestions"])
        
        if "questions" in sheet:
            recurse(sheet["questions"])

    @staticmethod
    def _inject_max_points(graded_answers: Dict, solution_sheet: Dict) -> None:
        """Inject 'points' (max points) from solution sheet into graded answers."""
        points_map = {}
        
        def map_points(questions):
            for q in questions:
                if "question_id" in q and "points" in q:
                    points_map[q["question_id"]] = q["points"]
                if "subquestions" in q:
                    map_points(q["subquestions"])
        
        if "questions" in solution_sheet:
            map_points(solution_sheet["questions"])

        def inject(questions):
            for q in questions:
                if "question_id" in q and q["question_id"] in points_map:
                    q["points"] = points_map[q["question_id"]]
                if "subquestions" in q:
                    inject(q["subquestions"])
        
        if "questions" in graded_answers:
            inject(graded_answers["questions"])

    @staticmethod
    def _enrich_metadata(model_answers: Dict, answer_sheet: Dict, model_name: str, exam, timestamp: str) -> None:
        model_answers.setdefault("exam_metadata", deepcopy(answer_sheet.get("exam_metadata", {})))
        model_answers["exam_metadata"]["evaluated_model"] = model_name
        model_answers["exam_metadata"]["benchmark_timestamp"] = timestamp
        model_answers["exam_metadata"]["source_processing_run"] = exam.timestamp

    @staticmethod
    def _enrich_grading_metadata(graded_answers: Dict, model_name: str, judge_name: str, exam, timestamp: str) -> None:
        graded_answers.setdefault("grading_metadata", {})
        graded_answers["grading_metadata"]["evaluation_model"] = model_name
        graded_answers["grading_metadata"]["judge_model"] = judge_name
        graded_answers["grading_metadata"]["benchmark_timestamp"] = timestamp
        graded_answers["grading_metadata"]["source_processing_run"] = exam.timestamp

    @staticmethod
    def _update_grading_summary(graded_answers: Dict) -> None:
        """Update grading_summary with detailed statistics across judge runs."""
        total_max_points = 0.0
        run_totals: Dict[str, float] = {}  # key: "judge_name|run_id", value: total_score

        def recurse(questions):
            nonlocal total_max_points
            for q in questions:
                # If leaf question (has judgments or answer_field)
                is_leaf = "subquestions" not in q or not q["subquestions"]
                
                if is_leaf:
                    # Add to total max points
                    pts = q.get("points", 0)
                    try:
                        total_max_points += float(pts) if pts is not None else 0.0
                    except (ValueError, TypeError):
                        pass
                    
                    # Add to run totals from judgments
                    if "judgments" in q and q["judgments"]:
                        for j in q["judgments"]:
                            if isinstance(j.get("awarded_points"), (int, float)):
                                key = f"{j['judge_name']}|{j['run_id']}"
                                run_totals[key] = run_totals.get(key, 0.0) + j["awarded_points"]

                # Recurse
                if "subquestions" in q:
                    recurse(q["subquestions"])

        if "questions" in graded_answers:
            recurse(graded_answers["questions"])

        # Calculate statistics
        if not run_totals:
            summary = {
                "total_points": round(total_max_points, 2),
                "judge_runs": {},
                "aggregation": {
                    "average_points": 0.0,
                    "average_percentage": 0.0,
                    "std_dev_points": 0.0,
                    "std_dev_percentage": 0.0
                }
            }
        else:
            scores = list(run_totals.values())
            n = len(scores)
            avg_points = sum(scores) / n
            
            variance = sum((x - avg_points) ** 2 for x in scores) / n
            std_dev_points = math.sqrt(variance)
            
            avg_percentage = (avg_points / total_max_points * 100) if total_max_points > 0 else 0.0
            std_dev_percentage = (std_dev_points / total_max_points * 100) if total_max_points > 0 else 0.0
            
            judge_runs = {}
            for key, points in run_totals.items():
                pct = (points / total_max_points * 100) if total_max_points > 0 else 0.0
                judge_runs[key] = {
                    "awarded_points": round(points, 2),
                    "percentage": round(pct, 2)
                }

            summary = {
                "total_points": round(total_max_points, 2),
                "judge_runs": judge_runs,
                "aggregation": {
                    "average_points": round(avg_points, 2),
                    "average_percentage": round(avg_percentage, 2),
                    "std_dev_points": round(std_dev_points, 2),
                    "std_dev_percentage": round(std_dev_percentage, 2)
                }
            }

        graded_answers["grading_summary"] = summary


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
