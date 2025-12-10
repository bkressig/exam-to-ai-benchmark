"""
PDF processing for exam documents.
Reads PDF files and converts them to base64 data URLs for direct LLM upload.
"""

from pathlib import Path
from typing import Dict, List
import base64


class PDFProcessor:
    """Process PDF files for exam benchmarking."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def _file_to_data_url(self, file_path: Path) -> str:
        """Read a file and return a base64 data URL."""
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f"data:application/pdf;base64,{encoded}"
    
    def extract_exam_content(self, pdf_files: List[Path], output_dir: Path) -> Dict:
        """
        Read exam PDF files as base64 data URLs.
        
        Args:
            pdf_files: List of exam PDF file paths
            output_dir: Directory for any outputs (unused now but kept for interface compatibility)
        
        Returns:
            Dict containing:
                - pdf_files: List of dicts with 'filename' and 'data_url'
        """
        pdf_data = []
        
        for pdf_file in pdf_files:
            data_url = self._file_to_data_url(pdf_file)
            pdf_data.append({
                "filename": pdf_file.name,
                "data_url": data_url
            })
        
        return {
            'pdf_files': pdf_data
        }
    
    def extract_solution_content(self, pdf_files: List[Path], output_dir: Path) -> Dict:
        """
        Read solution PDF files as base64 data URLs.
        """
        return self.extract_exam_content(pdf_files, output_dir)