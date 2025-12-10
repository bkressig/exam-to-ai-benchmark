import json
import math
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path: str, data: Dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def aggregate_judgments(graded_sheet: Dict) -> Tuple[float, Dict[str, float]]:
    """
    1. Traverse the graded sheet.
    2. Sum up 'points' (max points) found in the questions.
    3. Aggregate judgments to calculate average awarded_points per question.
    4. Return (total_max_points, run_totals_map).
    """
    total_max_points = 0.0
    run_totals: Dict[str, float] = {}  # key: "judge_name|run_id", value: total_score

    def recurse(questions):
        nonlocal total_max_points
        for q in questions:
            # If leaf question (has judgments or answer_field)
            is_leaf = "subquestions" not in q or not q["subquestions"]
            
            if is_leaf:
                # Add to total max points from the question itself
                pts = q.get("points", 0)
                try:
                    total_max_points += float(pts) if pts is not None else 0.0
                except (ValueError, TypeError):
                    pass
                
                # Process judgments
                if "judgments" in q and q["judgments"]:
                    # Calculate average for the question
                    valid_scores = [j["awarded_points"] for j in q["judgments"] if isinstance(j.get("awarded_points"), (int, float))]
                    if valid_scores:
                        q["awarded_points"] = sum(valid_scores) / len(valid_scores)
                    else:
                        q["awarded_points"] = 0.0
                    
                    # Add to run totals
                    for j in q["judgments"]:
                        if isinstance(j.get("awarded_points"), (int, float)):
                            key = f"{j['judge_name']}|{j['run_id']}"
                            run_totals[key] = run_totals.get(key, 0.0) + j["awarded_points"]
                else:
                    # No judgments, ensure awarded_points is 0
                    q["awarded_points"] = 0.0

            # Recurse
            if "subquestions" in q:
                recurse(q["subquestions"])

    if "questions" in graded_sheet:
        recurse(graded_sheet["questions"])
    
    return total_max_points, run_totals

def calculate_statistics(run_totals: Dict[str, float], total_max_points: float) -> Dict[str, Any]:
    if not run_totals:
        return {
            "average_points": 0.0,
            "average_percentage": 0.0,
            "std_dev_points": 0.0,
            "std_dev_percentage": 0.0,
            "judge_runs": {}
        }
    
    scores = list(run_totals.values())
    n = len(scores)
    avg_points = sum(scores) / n
    
    variance = sum((x - avg_points) ** 2 for x in scores) / n
    std_dev_points = math.sqrt(variance)
    
    avg_percentage = (avg_points / total_max_points * 100) if total_max_points > 0 else 0.0
    std_dev_percentage = (std_dev_points / total_max_points * 100) if total_max_points > 0 else 0.0
    
    # Format judge runs for report
    judge_runs = {}
    for key, points in run_totals.items():
        pct = (points / total_max_points * 100) if total_max_points > 0 else 0.0
        judge_runs[key] = {
            "awarded_points": round(points, 2),
            "percentage": round(pct, 2)
        }

    return {
        "average_points": avg_points,
        "average_percentage": avg_percentage,
        "std_dev_points": std_dev_points,
        "std_dev_percentage": std_dev_percentage,
        "judge_runs": judge_runs
    }

def main():
    parser = argparse.ArgumentParser(description="Recalculate aggregations for a graded answers JSON file.")
    parser.add_argument("file_path", nargs="?", help="Path to the graded_answers.json file")
    args = parser.parse_args()

    file_path = args.file_path

    path_obj = Path(file_path)
    if not path_obj.exists():
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    print(f"Loading file: {file_path}")
    graded = load_json(file_path)
    
    print("Aggregating points from graded sheet...")
    total_max_points, run_totals = aggregate_judgments(graded)
    
    print(f"Total Max Points found: {total_max_points}")
    
    print("Calculating statistics...")
    stats = calculate_statistics(run_totals, total_max_points)
    
    # Construct new summary
    summary = {
        "total_points": round(total_max_points, 2),
        "judge_runs": stats["judge_runs"],
        "aggregation": {
            "average_points": round(stats["average_points"], 2),
            "average_percentage": round(stats["average_percentage"], 2),
            "std_dev_points": round(stats["std_dev_points"], 2),
            "std_dev_percentage": round(stats["std_dev_percentage"], 2)
        }
    }
    
    graded["grading_summary"] = summary
    
    print("Saving updated file...")
    save_json(file_path, graded)
    
    print("\nUpdated Grading Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
