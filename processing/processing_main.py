"""
Main orchestrator for processing exam data into structured JSON format.
Generates answer_sheet.json and solution_sheet.json for benchmarking.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from llm_helper import OpenRouterClient
from process_pdf import PDFProcessor


class ExamProcessor:
    """Process raw exam data into structured answer and solution sheets."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        processing_config = self.config.get('processing', {})
        model = processing_config.get('processing_model', 'anthropic/claude-haiku-4.5')

        self.llm_client = OpenRouterClient(
            model=model
        )
        self.pdf_processor = PDFProcessor(self.llm_client)
    
    def scan_raw_data(self, raw_data_dir: str, professions_filter: Optional[List[str]] = None, exam_numbers_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan raw_data directory for exam folders.
        Structure: raw/profession/number/exam and raw/profession/number/solution
        """
        raw_path = Path(raw_data_dir)
        exam_folders = []
        
        for profession_dir in raw_path.iterdir():
            if not profession_dir.is_dir():
                continue
            
            if professions_filter and profession_dir.name not in professions_filter:
                continue
            
            # Iterate over numbered folders (1, 2, 3...)
            for exam_num_dir in profession_dir.iterdir():
                if not exam_num_dir.is_dir():
                    continue
                
                # Filter by exam number if specified
                if exam_numbers_filter and exam_numbers_filter != "all" and exam_num_dir.name not in exam_numbers_filter:
                    continue
                
                
                exam_dir = exam_num_dir / "exam"
                solution_dir = exam_num_dir / "solution"
                
                if exam_dir.exists() and solution_dir.exists():
                    exam_folders.append({
                        'profession': profession_dir.name,
                        'exam_number': exam_num_dir.name,
                        'exam_dir': exam_dir,
                        'solution_dir': solution_dir
                    })
        
        return exam_folders
    
    def process_exam_folder(self, exam_info: Dict, output_base_dir: str) -> Dict:
        """
        Process a single exam folder to generate answer_sheet.json and solution_sheet.json.
        """
        profession = exam_info['profession']
        exam_number = exam_info['exam_number']
        
        # Create timestamp for this processing run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build output path: processed/<profession>/<exam_number>/<timestamp>/
        output_base = Path(output_base_dir) / profession / exam_number
        output_dir = output_base / timestamp
        
        exam_id = f"{profession}/{exam_number}"
        
        print(f"\n{'='*60}")
        print(f"Processing: {exam_id}")
        print(f"{'='*60}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process exam PDFs
        exam_pdf_files = list(set(
            list(exam_info['exam_dir'].glob("*.pdf")) + 
            list(exam_info['exam_dir'].glob("*.PDF"))
        ))
        solution_pdf_files = list(set(
            list(exam_info['solution_dir'].glob("*.pdf")) + 
            list(exam_info['solution_dir'].glob("*.PDF"))
        ))
        
        if not exam_pdf_files:
            print(f"  No exam PDFs found in {exam_info['exam_dir']}")
            return {'status': 'error', 'message': 'No exam PDFs'}
        
        if not solution_pdf_files:
            print(f"  No solution PDFs found in {exam_info['solution_dir']}")
            return {'status': 'error', 'message': 'No solution PDFs'}
        
        # More descriptive output
        exam_files_str = f"{len(exam_pdf_files)} PDF file(s)" if len(exam_pdf_files) > 1 else "1 PDF file"
        solution_files_str = f"{len(solution_pdf_files)} PDF file(s)" if len(solution_pdf_files) > 1 else "1 PDF file"
        print(f"  Exam: {exam_files_str}")
        print(f"  Solution: {solution_files_str}")
        
        # Extract exam content (text + images as base64)
        # Now returns dict with 'pdf_data_urls'
        exam_content = self.pdf_processor.extract_exam_content(exam_pdf_files, output_dir)
        
        # Extract solution content
        # Now returns dict with 'pdf_data_urls'
        solution_content = self.pdf_processor.extract_solution_content(solution_pdf_files, output_dir)
        
        # Generate answer sheet using LLM
        print("  Generating answer_sheet.json...")
        answer_sheet = self.llm_client.generate_answer_sheet(exam_content, profession, exam_number)
        
        # Generate solution sheet using LLM
        print("  Generating solution_sheet.json...")
        solution_sheet = self.llm_client.generate_solution_sheet(
            exam_content, solution_content, answer_sheet, profession, exam_number
        )
        
        # Save JSON files
        answer_sheet_path = output_dir / "answer_sheet.json"
        solution_sheet_path = output_dir / "solution_sheet.json"
        
        with open(answer_sheet_path, 'w', encoding='utf-8') as f:
            json.dump(answer_sheet, f, ensure_ascii=False, indent=2)
        print(f"  Created: answer_sheet.json")
        
        with open(solution_sheet_path, 'w', encoding='utf-8') as f:
            json.dump(solution_sheet, f, ensure_ascii=False, indent=2)
        print(f"  Created: solution_sheet.json")
        
        # Save metadata
        metadata = {
            'profession': profession,
            'exam_number': exam_number,
            'exam_id': exam_id,
            'num_questions': len(answer_sheet.get('questions', [])),
            'processing_date': datetime.now().isoformat()
        }
        
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  Created: metadata.json")
        
        return {
            'status': 'success',
            'exam_id': exam_id,
            'output_dir': str(output_dir),
            'num_questions': metadata['num_questions']
        }
    
    
    def process_all(self, raw_data_dir: Optional[str] = None, output_base_dir: Optional[str] = None) -> Dict:
        """Process all exams in the raw_data directory."""
        raw_data_dir = raw_data_dir or self.config['raw_data_dir']
        output_base_dir = output_base_dir or self.config['processed_data_dir']
        
        print(f"Raw data: {raw_data_dir}")
        print(f"Output base: {output_base_dir}")
        
        processing_config = self.config.get('processing', {})
        professions_filter = processing_config.get('professions')
        exam_numbers_filter = processing_config.get('exam_numbers')
        
        if professions_filter:
            print(f"Processing professions: {professions_filter}")
        if exam_numbers_filter and exam_numbers_filter != "all":
            print(f"Processing exam numbers: {exam_numbers_filter}")
        
        exam_folders = self.scan_raw_data(raw_data_dir, professions_filter, exam_numbers_filter)
        
        if not exam_folders:
            print("No exam folders found")
            return {}
        
        exam_count_str = f"{len(exam_folders)} exam(s)" if len(exam_folders) != 1 else "1 exam"
        print(f"Found {exam_count_str} to process\n")
        
        results = {}
        for exam_info in exam_folders:
            result = self.process_exam_folder(exam_info, output_base_dir)
            exam_id = result.get('exam_id', 'unknown')
            results[exam_id] = result
        
        print("\n" + "="*60)
        print("Processing Complete!")
        print(f"Output saved to: {output_base_dir}")
        print("="*60)
        return results


if __name__ == "__main__":
    processor = ExamProcessor()
    processor.process_all()
