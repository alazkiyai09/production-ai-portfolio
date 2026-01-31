"""
Table Extractor - Extract tables from PDFs and convert to structured formats.

This module provides comprehensive table extraction capabilities including:
- PDF table extraction using tabula-py (primary) and Camelot (fallback)
- Conversion to pandas DataFrame
- Natural language description generation
- Embedding generation for semantic search
"""

import io
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

import numpy as np

logger = logging.getLogger(__name__)


class TableExtractionBackend(Enum):
    """Table extraction backend options."""
    TABULA = "tabula"
    CAMELOT = "camelot"
    AUTO = "auto"


class DescriptionStyle(Enum):
    """Table description style options."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    STRUCTURAL = "structural"
    NATURAL = "natural"


@dataclass
class TableData:
    """
    Container for extracted table data and metadata.

    Attributes:
        dataframe: pandas DataFrame containing the table data
        description: Natural language description of the table
        source_doc: Path to the source document
        page_number: Page number where table was found
        table_index: Index of the table on the page
        embedding: Embedding vector for semantic search
        format: Original table format (markdown, csv, json, etc.)
        row_count: Number of rows in the table
        col_count: Number of columns in the table
        metadata: Additional metadata dictionary
    """
    dataframe: Any  # pd.DataFrame
    description: Optional[str] = None
    source_doc: str = ""
    page_number: int = 0
    table_index: int = 0
    embedding: Optional[np.ndarray] = None
    format: str = "dataframe"
    row_count: int = 0
    col_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Update row and column counts from dataframe."""
        if self.dataframe is not None:
            try:
                self.row_count = len(self.dataframe)
                self.col_count = len(self.dataframe.columns) if hasattr(self.dataframe, 'columns') else 0
            except Exception:
                pass

    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        if self.dataframe is None:
            return ""

        try:
            return self.dataframe.to_markdown(index=False)
        except Exception:
            # Fallback to tab-separated format
            return self._to_tab_separated()

    def to_csv(self) -> str:
        """Convert table to CSV format."""
        if self.dataframe is None:
            return ""

        try:
            return self.dataframe.to_csv(index=False)
        except Exception as e:
            logger.warning(f"Failed to convert to CSV: {e}")
            return ""

    def to_json(self) -> str:
        """Convert table to JSON format."""
        if self.dataframe is None:
            return "[]"

        try:
            return self.dataframe.to_json(orient='records', indent=2)
        except Exception as e:
            logger.warning(f"Failed to convert to JSON: {e}")
            return "[]"

    def to_html(self) -> str:
        """Convert table to HTML format."""
        if self.dataframe is None:
            return ""

        try:
            return self.dataframe.to_html(index=False)
        except Exception as e:
            logger.warning(f"Failed to convert to HTML: {e}")
            return ""

    def _to_tab_separated(self) -> str:
        """Fallback: convert to tab-separated format."""
        try:
            output = io.StringIO()

            # Header
            if hasattr(self.dataframe, 'columns'):
                output.write("\t".join(str(c) for c in self.dataframe.columns))
                output.write("\n")

            # Data
            for _, row in self.dataframe.iterrows():
                output.write("\t".join(str(v) for v in row.values))
                output.write("\n")

            return output.getvalue()
        except Exception:
            return str(self.dataframe)

    def get_headers(self) -> List[str]:
        """Get table column headers."""
        if self.dataframe is None:
            return []

        try:
            return list(self.dataframe.columns)
        except Exception:
            return []

    def get_row(self, index: int) -> Dict[str, Any]:
        """Get a specific row as a dictionary."""
        if self.dataframe is None:
            return {}

        try:
            return self.dataframe.iloc[index].to_dict()
        except Exception:
            return {}

    def get_column(self, name: str) -> List[Any]:
        """Get a specific column as a list."""
        if self.dataframe is None:
            return []

        try:
            return self.dataframe[name].tolist()
        except Exception:
            return []

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for numeric columns."""
        if self.dataframe is None:
            return {}

        try:
            stats = {
                "row_count": self.row_count,
                "col_count": self.col_count,
                "headers": self.get_headers(),
            }

            # Add numeric stats
            numeric_cols = self.dataframe.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                stats["numeric_columns"] = numeric_cols
                stats["numeric_summary"] = self.dataframe[numeric_cols].describe().to_dict()

            return stats
        except Exception as e:
            logger.warning(f"Failed to generate summary stats: {e}")
            return {}

    def __repr__(self) -> str:
        return (
            f"TableData(source={self.source_doc}, page={self.page_number}, "
            f"index={self.table_index}, shape=({self.row_count}x{self.col_count}), "
            f"has_description={self.description is not None})"
        )


class TableExtractor:
    """
    Advanced table extractor for multi-modal RAG systems.

    Features:
    - Extract tables from PDF using tabula-py or Camelot
    - Convert to pandas DataFrame
    - Generate natural language descriptions
    - Create embeddings for semantic search
    - Multiple output formats (Markdown, CSV, JSON, HTML)
    """

    def __init__(
        self,
        backend: TableExtractionBackend = TableExtractionBackend.AUTO,
        description_style: DescriptionStyle = DescriptionStyle.NATURAL,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        extract_descriptions: bool = True,
        extract_embeddings: bool = True,
    ):
        """
        Initialize the TableExtractor.

        Args:
            backend: Preferred extraction backend
            description_style: Style for table descriptions
            embedding_model: Sentence transformer model for embeddings
            device: Device to use for models ('cuda', 'cpu', or None for auto)
            extract_descriptions: Whether to generate descriptions
            extract_embeddings: Whether to generate embeddings
        """
        self.backend = backend
        self.description_style = description_style
        self.embedding_model_name = embedding_model
        self.extract_descriptions = extract_descriptions
        self.extract_embeddings = extract_embeddings

        # Determine device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Initialize components
        self.embedding_model = None
        self.tabula_available = False
        self.camelot_available = False

        self._init_backends()
        self._init_embedding_model()

        logger.info(
            f"TableExtractor initialized: backend={backend.value}, "
            f"device={self.device}, "
            f"tabula={self.tabula_available}, "
            f"camelot={self.camelot_available}"
        )

    def _init_backends(self):
        """Initialize table extraction backends."""
        # Check tabula-py
        try:
            import tabula
            self.tabula_available = True
            logger.debug("tabula-py is available")
        except ImportError:
            logger.debug("tabula-py not available. Install with: pip install tabula-py")

        # Check Camelot
        try:
            import camelot
            self.camelot_available = True
            logger.debug("Camelot is available")
        except ImportError:
            logger.debug("Camelot not available. Install with: pip install camelot-py[cv]")

    def _init_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        if not self.extract_embeddings:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_model.to(self.device)
            logger.info(f"Embedding model loaded: {self.embedding_model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")

    def extract_tables_from_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[Union[str, List[int]]] = None,
        extract_embeddings: Optional[bool] = None,
        extract_descriptions: Optional[bool] = None,
        **kwargs
    ) -> List[TableData]:
        """
        Extract tables from a PDF document.

        Args:
            pdf_path: Path to the PDF file
            pages: Pages to extract ('all' or list of page numbers)
            extract_embeddings: Whether to generate embeddings
            extract_descriptions: Whether to generate descriptions
            **kwargs: Additional parameters for extraction backend

        Returns:
            List of TableData objects

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If PDF processing fails
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting tables from PDF: {pdf_path}")

        # Use instance defaults if not specified
        if extract_embeddings is None:
            extract_embeddings = self.extract_embeddings
        if extract_descriptions is None:
            extract_descriptions = self.extract_descriptions

        # Try different backends
        extracted_tables = []

        if self.backend == TableExtractionBackend.AUTO:
            # Try tabula first, then camelot
            if self.tabula_available:
                extracted_tables = self._extract_with_tabula(
                    pdf_path, pages, **kwargs
                )

            if not extracted_tables and self.camelot_available:
                logger.info("Tabula found no tables, trying Camelot")
                extracted_tables = self._extract_with_camelot(
                    pdf_path, pages, **kwargs
                )

        elif self.backend == TableExtractionBackend.TABULA:
            if not self.tabula_available:
                raise ImportError("tabula-py is required but not available")

            extracted_tables = self._extract_with_tabula(pdf_path, pages, **kwargs)

        elif self.backend == TableExtractionBackend.CAMELOT:
            if not self.camelot_available:
                raise ImportError("Camelot is required but not available")

            extracted_tables = self._extract_with_camelot(pdf_path, pages, **kwargs)

        # Process each table
        results = []

        for table_idx, table_data in enumerate(extracted_tables):
            try:
                df, page_num = table_data

                # Skip empty dataframes
                if df is None or df.empty:
                    continue

                # Create TableData
                table = TableData(
                    dataframe=df,
                    source_doc=str(pdf_path),
                    page_number=page_num,
                    table_index=table_idx,
                )

                # Generate description
                if extract_descriptions:
                    table.description = self.generate_table_description(df)

                # Generate embedding
                if extract_embeddings and table.description:
                    table.embedding = self.get_table_embedding(table.description)

                results.append(table)
                logger.debug(
                    f"Extracted table {table_idx + 1} from page {page_num}: "
                    f"{len(df)} rows x {len(df.columns)} cols"
                )

            except Exception as e:
                logger.warning(f"Failed to process table {table_idx}: {e}")
                continue

        logger.info(f"Extracted {len(results)} tables from PDF")
        return results

    def _extract_with_tabula(
        self,
        pdf_path: Path,
        pages: Optional[Union[str, List[int]]] = None,
        **kwargs
    ) -> List[Tuple[Any, int]]:
        """Extract tables using tabula-py."""
        try:
            import tabula

            # Set default options
            options = {
                'pages': pages or 'all',
                'multiple_tables': True,
                'pandas_options': {'header': 0},
            }
            options.update(kwargs)

            logger.debug(f"Extracting with tabula options: {options}")

            # Extract tables
            tables = tabula.read_pdf(str(pdf_path), **options)

            # Determine page numbers (tabula doesn't provide page info easily)
            # We'll approximate by spreading tables across pages
            result = []
            for i, df in enumerate(tables):
                # Estimate page number (this is approximate)
                page_num = 1  # tabula doesn't provide page info
                result.append((df, page_num))

            return result

        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")
            return []

    def _extract_with_camelot(
        self,
        pdf_path: Path,
        pages: Optional[Union[str, List[int]]] = None,
        flavor: str = "lattice",
        **kwargs
    ) -> List[Tuple[Any, int]]:
        """Extract tables using Camelot."""
        try:
            import camelot

            # Convert pages to Camelot format
            if pages is None or pages == "all":
                pages_str = "all"
            elif isinstance(pages, list):
                pages_str = ",".join(str(p) for p in pages)
            else:
                pages_str = str(pages)

            # Set default options
            options = {
                'pages': pages_str,
                'flavor': flavor,
                'strip_text': True,
            }
            options.update(kwargs)

            logger.debug(f"Extracting with Camelot options: {options}")

            # Extract tables
            c_tables = camelot.read_pdf(str(pdf_path), **options)

            result = []
            for i, c_table in enumerate(c_tables):
                df = c_table.df
                page_num = c_table.page

                # Clean dataframe
                df = self._clean_dataframe(df)

                result.append((df, page_num))

            return result

        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
            return []

    def _clean_dataframe(self, df: Any) -> Any:
        """Clean extracted dataframe."""
        try:
            import pandas as pd

            # Remove empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')

            # Reset index
            df = df.reset_index(drop=True)

            return df
        except Exception as e:
            logger.warning(f"Failed to clean dataframe: {e}")
            return df

    def convert_to_dataframe(self, table: Any) -> Any:
        """
        Convert table to pandas DataFrame.

        Args:
            table: Table data (can be various formats)

        Returns:
            pandas DataFrame

        Raises:
            ValueError: If table format is not supported
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion")

        # Already a DataFrame
        if isinstance(table, pd.DataFrame):
            return table

        # List of lists
        if isinstance(table, list):
            if all(isinstance(row, list) for row in table):
                return pd.DataFrame(table[1:], columns=table[0] if table else None)
            if all(isinstance(row, dict) for row in table):
                return pd.DataFrame(table)

        # Dictionary
        if isinstance(table, dict):
            return pd.DataFrame(table)

        # String (CSV, Markdown, etc.)
        if isinstance(table, str):
            # Try CSV
            try:
                import io
                return pd.read_csv(io.StringIO(table))
            except Exception:
                pass

            # Try Markdown-like format
            try:
                lines = table.strip().split('\n')
                if '|' in lines[0]:
                    # Markdown table
                    rows = [line.split('|')[1:-1] for line in lines if line.strip() and not line.strip().startswith('|-')]
                    if rows:
                        return pd.DataFrame(rows[1:], columns=rows[0])
            except Exception:
                pass

        raise ValueError(f"Unsupported table format: {type(table)}")

    def generate_table_description(
        self,
        df: Any,
        style: Optional[DescriptionStyle] = None,
    ) -> str:
        """
        Generate natural language description of a table.

        Args:
            df: pandas DataFrame
            style: Description style (None for instance default)

        Returns:
            Natural language description
        """
        if df is None or df.empty:
            return "Empty table"

        style = style or self.description_style

        try:
            headers = list(df.columns)
            row_count = len(df)
            col_count = len(headers)

            if style == DescriptionStyle.SUMMARY:
                return self._summary_description(df, headers, row_count, col_count)

            elif style == DescriptionStyle.DETAILED:
                return self._detailed_description(df, headers, row_count, col_count)

            elif style == DescriptionStyle.STRUCTURAL:
                return self._structural_description(df, headers, row_count, col_count)

            else:  # NATURAL
                return self._natural_description(df, headers, row_count, col_count)

        except Exception as e:
            logger.warning(f"Failed to generate description: {e}")
            return f"Table with {len(df.columns)} columns and {len(df)} rows"

    def _summary_description(
        self,
        df: Any,
        headers: List[str],
        row_count: int,
        col_count: int
    ) -> str:
        """Generate a brief summary description."""
        return f"A table with {col_count} columns ({', '.join(headers[:5])}{'...' if col_count > 5 else ''}) and {row_count} rows"

    def _detailed_description(
        self,
        df: Any,
        headers: List[str],
        row_count: int,
        col_count: int
    ) -> str:
        """Generate a detailed description with sample data."""
        description_parts = [
            f"A {row_count} by {col_count} table containing the following columns: {', '.join(headers)}."
        ]

        # Add data types
        try:
            dtypes = df.dtypes.astype(str).to_dict()
            description_parts.append("\nColumn data types:")
            for col, dtype in dtypes.items():
                description_parts.append(f"  - {col}: {dtype}")
        except Exception:
            pass

        # Add sample rows
        try:
            sample_rows = df.head(3)
            description_parts.append("\nSample data:")
            for idx, row in sample_rows.iterrows():
                row_str = " | ".join(str(v)[:20] for v in row.values)
                description_parts.append(f"  Row {idx}: {row_str}")
        except Exception:
            pass

        return "\n".join(description_parts)

    def _structural_description(
        self,
        df: Any,
        headers: List[str],
        row_count: int,
        col_count: int
    ) -> str:
        """Generate a structural description of the table."""
        parts = []

        # Basic structure
        parts.append(f"Table structure: {row_count} rows × {col_count} columns")

        # Column names
        parts.append(f"Columns: {', '.join(headers)}")

        # Identify numeric columns
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
        except Exception:
            pass

        # Identify text columns
        try:
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                parts.append(f"Text columns: {', '.join(text_cols)}")
        except Exception:
            pass

        return ". ".join(parts)

    def _natural_description(
        self,
        df: Any,
        headers: List[str],
        row_count: int,
        col_count: int
    ) -> str:
        """Generate a natural language description."""
        parts = []

        # Opening
        if row_count <= 5:
            parts.append(f"This is a small table with {col_count} columns and {row_count} rows.")
        elif row_count <= 20:
            parts.append(f"This table contains {col_count} columns and {row_count} rows of data.")
        else:
            parts.append(f"This is a large table with {col_count} columns and {row_count} rows.")

        # Column description
        parts.append(f"The columns are {', '.join(headers[:5])}")
        if col_count > 5:
            parts.append(f"and {col_count - 5} others")

        # Data type info
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                if len(numeric_cols) == col_count:
                    parts.append("All columns contain numeric data.")
                else:
                    parts.append(f"The numeric columns are: {', '.join(numeric_cols[:3])}")
        except Exception:
            pass

        return " ".join(parts) + "."

    def get_table_embedding(
        self,
        description: str,
    ) -> Optional[np.ndarray]:
        """
        Generate embedding for table description.

        Args:
            description: Table description text

        Returns:
            Embedding vector as numpy array or None
        """
        if self.embedding_model is None:
            logger.warning("Embedding model not initialized")
            return None

        if not description or not description.strip():
            return None

        try:
            embedding = self.embedding_model.encode(
                description,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Normalize
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def save_tables(
        self,
        tables: List[TableData],
        output_dir: Union[str, Path],
        format: str = "csv",
        naming_pattern: str = "doc_{doc}_page_{page}_tbl_{idx}.{ext}",
    ) -> List[Path]:
        """
        Save extracted tables to files.

        Args:
            tables: List of TableData objects
            output_dir: Output directory path
            format: Output format (csv, markdown, json, html)
            naming_pattern: Naming pattern with placeholders

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for table in tables:
            try:
                # Generate filename
                doc_name = Path(table.source_doc).stem if table.source_doc else "unknown"

                filename = naming_pattern.format(
                    doc=doc_name,
                    page=table.page_number,
                    idx=table.table_index,
                    ext=format,
                )

                output_path = output_dir / filename

                # Save based on format
                if format == "csv":
                    content = table.to_csv()
                elif format == "markdown" or format == "md":
                    content = table.to_markdown()
                elif format == "json":
                    content = table.to_json()
                elif format == "html":
                    content = table.to_html()
                else:
                    content = table.to_csv()

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                saved_paths.append(output_path)
                logger.debug(f"Saved table to {output_path}")

            except Exception as e:
                logger.warning(f"Failed to save table: {e}")
                continue

        return saved_paths


# Convenience functions
def extract_pdf_tables(
    pdf_path: Union[str, Path],
    **kwargs
) -> List[TableData]:
    """
    Convenience function to extract tables from PDF.

    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional arguments for TableExtractor

    Returns:
        List of TableData objects
    """
    extractor = TableExtractor(**kwargs)
    return extractor.extract_tables_from_pdf(pdf_path)


def convert_to_dataframe(table: Any) -> Any:
    """
    Convenience function to convert table to DataFrame.

    Args:
        table: Table data in various formats

    Returns:
        pandas DataFrame
    """
    extractor = TableExtractor()
    return extractor.convert_to_dataframe(table)


def describe_table(df: Any, style: str = "natural") -> str:
    """
    Convenience function to generate table description.

    Args:
        df: pandas DataFrame
        style: Description style (summary, detailed, structural, natural)

    Returns:
        Table description
    """
    extractor = TableExtractor()
    style_enum = DescriptionStyle(style.lower())
    return extractor.generate_table_description(df, style_enum)
