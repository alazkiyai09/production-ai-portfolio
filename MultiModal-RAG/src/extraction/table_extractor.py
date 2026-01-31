"""
Table Extractor (Extraction module) - Extract tables from documents.

This is a simplified version that imports from the main multimodal module.
For advanced table extraction, see src/multimodal/table_extractor.py
"""

# Re-export the main TableExtractor for convenience
from ..multimodal.table_extractor import (
    TableExtractor,
    TableData,
    TableFormat,
    extract_tables,
    tables_to_markdown,
)

__all__ = [
    "TableExtractor",
    "TableData",
    "TableFormat",
    "extract_tables",
    "tables_to_markdown",
]
