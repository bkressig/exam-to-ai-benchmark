"""
Extract meaningful images from PDFs.
"""

from pathlib import Path
from typing import List, Dict
import fitz


class ImageExtractor:
    """Extract meaningful images from PDFs, filtering out logos and decorations."""
    
    def __init__(self, output_dir: str, min_width: int = 150, min_height: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_width = min_width
        self.min_height = min_height
    
    def _is_meaningful_image(self, width: int, height: int, size_bytes: int) -> bool:
        """Determine if an image is meaningful (not a logo/decoration)."""
        if width < self.min_width or height < self.min_height:
            return False
        
        if size_bytes < 1000:
            return False
        
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return False
        
        return True
    
    def extract_from_pdf(self, pdf_path: str, prefix: str = "img") -> List[Dict]:
        """Extract meaningful images from a PDF file."""
        saved_images = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            images = page.get_images(full=True)
            
            for img_index, img_info in enumerate(images):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                size_bytes = len(base_image["image"])
                
                if not self._is_meaningful_image(width, height, size_bytes):
                    continue
                
                filename = f"{prefix}_p{page_num + 1}_{img_index + 1}.{base_image['ext']}"
                filepath = self.output_dir / filename
                
                with open(filepath, "wb") as f:
                    f.write(base_image["image"])
                
                saved_images.append({
                    "filename": filename,
                    "page": page_num + 1,
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes
                })
        
        doc.close()
        return saved_images
