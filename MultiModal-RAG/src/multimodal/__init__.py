"""
MultiModal RAG - Multi-Modal Extensions for Enterprise RAG

This package provides multi-modal capabilities including:
- Image processing and OCR
- Table extraction and parsing
- Multi-modal embeddings (CLIP, Vision models)
- Vision-enhanced RAG chains
"""

from .image_processor import (
    ImageProcessor,
    ImageData,
    CaptioningBackend,
    OCREngine,
    extract_pdf_images,
    extract_docx_images,
    process_single_image,
)
from .table_extractor import (
    TableExtractor,
    TableData,
    TableExtractionBackend,
    DescriptionStyle,
    extract_pdf_tables,
    convert_to_dataframe as table_to_dataframe,
    describe_table,
)
from .multimodal_retriever import (
    MultiModalRetriever,
    MultiModalResult,
    ContentType,
    SearchResult,
    ImageResult,
    TableResult,
    create_retriever,
    search as multimodal_search,
)
from .vision_llm import VisionLLM, VisionProvider, VisionResponse
from .multimodal_rag import (
    MultiModalRAGChain,
    MultiModalResponse,
    Citation,
    TextCitation,
    ImageCitation,
    TableCitation,
    CitationSource,
)

__all__ = [
    "ImageProcessor",
    "ImageData",
    "CaptioningBackend",
    "OCREngine",
    "TableExtractor",
    "TableData",
    "TableExtractionBackend",
    "DescriptionStyle",
    "MultiModalRetriever",
    "MultiModalResult",
    "ContentType",
    "SearchResult",
    "ImageResult",
    "TableResult",
    "VisionLLM",
    "VisionProvider",
    "VisionResponse",
    "MultiModalRAGChain",
    "MultiModalResponse",
    "Citation",
    "TextCitation",
    "ImageCitation",
    "TableCitation",
    "CitationSource",
    "create_retriever",
    "multimodal_search",
    "extract_pdf_images",
    "extract_docx_images",
    "process_single_image",
    "extract_pdf_tables",
    "table_to_dataframe",
    "describe_table",
]

__version__ = "0.1.0"
