"""Locate processed exams eligible for benchmarking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ProcessedExam:
    """Represents a single processed exam run ready for benchmarking."""

    profession: str
    exam_number: str
    timestamp: str
    processed_dir: Path
    raw_exam_dir: Path
    year: Optional[str] = None  # Year is now metadata, not folder structure

    @property
    def exam_id(self) -> str:
        return f"{self.profession}/{self.exam_number}"


class ProcessedExamRepository:
    """Discover processed exams stored under the processed data directory."""

    def __init__(self, processed_data_dir: Path, raw_data_dir: Path,
                 professions_filter: Optional[List[str]] = None,
                 exam_numbers_filter: Optional[List[str]] = None):
        self._processed_root = processed_data_dir
        self._raw_root = raw_data_dir
        self._professions = professions_filter
        self._exam_numbers = exam_numbers_filter

    def list_latest_exams(self) -> List[ProcessedExam]:
        exams: List[ProcessedExam] = []

        for profession_dir in self._processed_root.iterdir():
            if not profession_dir.is_dir():
                continue
            profession = profession_dir.name
            if self._professions and profession not in self._professions:
                continue

            # Iterate over numbered folders (1, 2, 3...)
            for exam_num_dir in profession_dir.iterdir():
                if not exam_num_dir.is_dir():
                    continue
                
                exam_number = exam_num_dir.name
                
                # Filter by exam number if specified
                if self._exam_numbers and self._exam_numbers != "all" and exam_number not in self._exam_numbers:
                    continue
                
                # Find latest timestamp run
                latest_run = self._latest_run_with_json(exam_num_dir.iterdir())
                if not latest_run:
                    continue
                
                exams.append(
                    ProcessedExam(
                        profession=profession,
                        exam_number=exam_number,
                        timestamp=latest_run.name,
                        processed_dir=latest_run.resolve(),
                        raw_exam_dir=(self._raw_root / profession / exam_number / "exam").resolve(),
                    )
                )

        exams.sort(key=lambda item: (item.profession, item.exam_number, item.timestamp))
        return exams

    @staticmethod
    def _latest_run_with_json(directories: Iterable[Path]) -> Optional[Path]:
        candidates = [d for d in directories if d.is_dir() and (d / "answer_sheet.json").exists()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.name)
