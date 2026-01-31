"""
Extraction modules for multi-modal document processing.

Provides utilities for:
- OCR text extraction
- Table structure extraction
- Image description generation
"""

from .image_extractor import ImageExtractor, ExtractedImage
from .table_extractor import TableExtractor
from .ocr_processor import OCRProcessor, OCRResult, OCREngine

__all__ = [
    "ImageExtractor",
    "ExtractedImage",
    "TableExtractor",
    "OCRProcessor",
    "OCRResult",
    "OCREngine",
]
