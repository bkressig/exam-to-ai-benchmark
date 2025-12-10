"""PDF handling helpers for benchmarking."""

from dataclasses import dataclass
from io import BytesIO
import base64
from typing import List

import fitz
from PIL import Image

from exam_repository import ProcessedExam


@dataclass
class ExamPage:
    page_number: int
    source_file: str
    text: str
    image_base64: str


@dataclass
class ExamContent:
    text: str
    pages: List[ExamPage]


class ExamContentExtractor:
    """Extract text and page images from raw exam PDFs."""

    def __init__(self, dpi: int = 150):
        self._dpi = dpi

    def load(self, exam: ProcessedExam) -> ExamContent:
        if not exam.raw_exam_dir.exists():
            print(f"  ⚠ Exam directory not found: {exam.raw_exam_dir}")
            return ExamContent(text="", pages=[])

        pdf_files = list(exam.raw_exam_dir.glob("*.pdf")) + list(exam.raw_exam_dir.glob("*.PDF"))
        if not pdf_files:
            print(f"  ⚠ No exam PDFs found in {exam.raw_exam_dir}")
            return ExamContent(text="", pages=[])

        all_pages: List[ExamPage] = []
        text_parts: List[str] = []

        for pdf_path in pdf_files:
            document = fitz.open(pdf_path)
            for index in range(len(document)):
                page = document[index]
                page_text = page.get_text()
                page_image = self._page_to_base64(document, index)

                all_pages.append(
                    ExamPage(
                        page_number=index + 1,
                        source_file=pdf_path.name,
                        text=page_text,
                        image_base64=page_image,
                    )
                )
                text_parts.append(f"[Page {index + 1} from {pdf_path.name}]")
                text_parts.append(page_text)
            document.close()

        return ExamContent(text="\n".join(text_parts), pages=all_pages)

    def _page_to_base64(self, document: fitz.Document, page_index: int) -> str:
        page = document[page_index]
        matrix = fitz.Matrix(self._dpi / 72, self._dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded
