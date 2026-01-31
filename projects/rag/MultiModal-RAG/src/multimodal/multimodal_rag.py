"""
Multi-Modal RAG Chain - RAG with support for images and tables.

This module provides comprehensive multi-modal RAG capabilities including:
- Query processing with text, image, and table context
- Visual context integration with vision LLMs
- Multi-modal citation generation
- Streaming response support
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np

logger = logging.getLogger(__name__)


class CitationSource(Enum):
    """Source types for citations."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class Citation:
    """
    Base citation for referenced content.

    Attributes:
        content: The cited content
        source: Source document path
        page_number: Page number
        score: Relevance score
        metadata: Additional metadata
    """
    content: str
    source: str
    page_number: int = 0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "page_number": self.page_number,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class TextCitation(Citation):
    """Citation for text content."""
    chunk: str = ""
    start_index: int = 0
    end_index: int = 0


@dataclass
class ImageCitation(Citation):
    """Citation for image content."""
    image_data: Optional[bytes] = None
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    image_format: str = "png"
    base64_data: Optional[str] = None

    def get_base64(self) -> Optional[str]:
        """Get base64 encoded image."""
        if self.base64_data:
            return self.base64_data
        if self.image_data:
            return base64.b64encode(self.image_data).decode("utf-8")
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "source": self.source,
            "page_number": self.page_number,
            "score": self.score,
            "base64": self.get_base64(),
        }


@dataclass
class TableCitation(Citation):
    """Citation for table content."""
    dataframe: Any = None  # pd.DataFrame
    description: Optional[str] = None
    row_count: int = 0
    col_count: int = 0

    def get_markdown(self) -> str:
        """Get table as Markdown."""
        if self.dataframe is not None:
            try:
                return self.dataframe.to_markdown(index=False)
            except Exception:
                return self.description or ""
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "markdown": self.get_markdown(),
            "row_count": self.row_count,
            "col_count": self.col_count,
            "source": self.source,
            "page_number": self.page_number,
            "score": self.score,
        }


@dataclass
class MultiModalResponse:
    """
    Complete response from multi-modal RAG.

    Attributes:
        answer: Generated answer
        text_citations: Citations from text documents
        image_citations: Citations from images
        table_citations: Citations from tables
        query: Original query
        sources_used: List of source types used
        metadata: Additional metadata
    """
    answer: str
    text_citations: List[TextCitation] = field(default_factory=list)
    image_citations: List[ImageCitation] = field(default_factory=list)
    table_citations: List[TableCitation] = field(default_factory=list)
    query: str = ""
    sources_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_citations(self) -> List[Citation]:
        """Get all citations regardless of type."""
        all_citations = []
        all_citations.extend(self.text_citations)
        all_citations.extend(self.image_citations)
        all_citations.extend(self.table_citations)
        # Sort by score
        all_citations.sort(key=lambda c: c.score, reverse=True)
        return all_citations

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "answer": self.answer,
            "query": self.query,
            "sources_used": self.sources_used,
            "text_citations": [c.to_dict() for c in self.text_citations],
            "image_citations": [c.to_dict() for c in self.image_citations],
            "table_citations": [c.to_dict() for c in self.table_citations],
            "metadata": self.metadata,
        }

    def format_for_display(self) -> str:
        """Format response for display."""
        parts = [f"## Answer\n\n{self.answer}\n"]

        if self.text_citations:
            parts.append("\n## Text Sources\n")
            for i, citation in enumerate(self.text_citations, 1):
                parts.append(
                    f"{i}. **{Path(citation.source).name}** "
                    f"(page {citation.page_number}, score: {citation.score:.2f})\n"
                    f"   {citation.content[:200]}...\n"
                )

        if self.image_citations:
            parts.append("\n## Image Sources\n")
            for i, citation in enumerate(self.image_citations, 1):
                caption = citation.caption or "No caption"
                parts.append(
                    f"{i}. **{Path(citation.source).name}** "
                    f"(page {citation.page_number}, score: {citation.score:.2f})\n"
                    f"   {caption}\n"
                )

        if self.table_citations:
            parts.append("\n## Table Sources\n")
            for i, citation in enumerate(self.table_citations, 1):
                desc = citation.description or "Table data"
                parts.append(
                    f"{i}. **{Path(citation.source).name}** "
                    f"(page {citation.page_number}, score: {citation.score:.2f})\n"
                    f"   {desc}\n"
                )

        return "\n".join(parts)


class MultiModalRAGChain:
    """
    Multi-modal RAG chain for handling queries with text, images, and tables.

    Features:
    - Unified query processing across all modalities
    - Visual context integration with vision LLMs
    - Multi-modal citation generation
    - Streaming response support
    - Configurable context building
    """

    def __init__(
        self,
        retriever,
        vision_llm=None,
        text_llm=None,
        max_context_items: int = 10,
        max_images_in_prompt: int = 3,
        max_tables_in_prompt: int = 3,
        include_image_captions: bool = True,
        include_ocr_text: bool = True,
        include_table_descriptions: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the multi-modal RAG chain.

        Args:
            retriever: MultiModalRetriever instance
            vision_llm: VisionLLM instance for image understanding
            text_llm: Text LLM for generation (optional, uses vision_llm if None)
            max_context_items: Maximum items to include in context
            max_images_in_prompt: Maximum images to include in prompt
            max_tables_in_prompt: Maximum tables to include in prompt
            include_image_captions: Whether to include image captions
            include_ocr_text: Whether to include OCR text
            include_table_descriptions: Whether to include table descriptions
            system_prompt: Custom system prompt
        """
        self.retriever = retriever
        self.vision_llm = vision_llm
        self.text_llm = text_llm or vision_llm
        self.max_context_items = max_context_items
        self.max_images_in_prompt = max_images_in_prompt
        self.max_tables_in_prompt = max_tables_in_prompt
        self.include_image_captions = include_image_captions
        self.include_ocr_text = include_ocr_text
        self.include_table_descriptions = include_table_descriptions

        # Default system prompt
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant that answers questions based on "
            "provided context from documents, including text, images, and tables. "
            "When answering, cite the specific sources you used. "
            "If the provided context doesn't contain enough information to answer "
            "the question, say so clearly."
        )

        if self.text_llm is None:
            logger.warning("No LLM provided. Query processing will be limited.")

    def query(
        self,
        question: str,
        include_images: bool = True,
        include_tables: bool = True,
        content_types: Union[str, List[str]] = "all",
        top_k: int = 10,
        min_score: float = 0.0,
        stream: bool = False,
    ) -> MultiModalResponse:
        """
        Process a query with multi-modal context.

        Args:
            question: User question
            include_images: Whether to include images in context
            include_tables: Whether to include tables in context
            content_types: Content types to search
            top_k: Number of results to retrieve
            min_score: Minimum similarity score
            stream: Whether to stream the response

        Returns:
            MultiModalResponse with answer and citations
        """
        logger.info(f"Processing query: {question[:100]}...")

        # Retrieve relevant content
        results = self.retriever.retrieve(
            query=question,
            content_types=content_types,
            top_k=top_k,
            min_score=min_score,
        )

        # Filter results based on preferences
        text_results = [r for r in results if r.content_type == "text"]
        image_results = [r for r in results if r.content_type == "image"] if include_images else []
        table_results = [r for r in results if r.content_type == "table"] if include_tables else []

        # Build context
        context_text = self._build_multimodal_context(
            text_results=text_results,
            image_results=image_results,
            table_results=table_results,
        )

        # Prepare images for vision LLM
        images_for_llm = []
        if include_images and image_results:
            images_for_llm = [
                r.content for r in image_results[:self.max_images_in_prompt]
            ]

        # Generate response
        if stream:
            answer = self._generate_stream(
                question=question,
                context=context_text,
                images=images_for_llm,
            )
        else:
            answer = self._generate_with_vision(
                question=question,
                context=context_text,
                images=images_for_llm,
            )

        # Create citations
        text_citations = self._create_text_citations(text_results)
        image_citations = self._create_image_citations(image_results)
        table_citations = self._create_table_citations(table_results)

        # Track sources used
        sources_used = []
        if text_citations:
            sources_used.append("text")
        if image_citations:
            sources_used.append("images")
        if table_citations:
            sources_used.append("tables")

        response = MultiModalResponse(
            answer=answer,
            text_citations=text_citations,
            image_citations=image_citations,
            table_citations=table_citations,
            query=question,
            sources_used=sources_used,
            metadata={
                "total_results": len(results),
                "text_results": len(text_results),
                "image_results": len(image_results),
                "table_results": len(table_results),
            },
        )

        logger.info(f"Generated response with {len(sources_used)} source types")
        return response

    def _build_multimodal_context(
        self,
        text_results: List = [],
        image_results: List = [],
        table_results: List = [],
    ) -> str:
        """
        Build context string from multi-modal results.

        Args:
            text_results: Text search results
            image_results: Image search results
            table_results: Table search results

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add text context
        if text_results:
            context_parts.append("## Text Context\n")
            for i, result in enumerate(text_results[:self.max_context_items], 1):
                source_name = Path(result.source).name
                context_parts.append(
                    f"[{i}] From {source_name} (page {result.page_number}, "
                    f"relevance: {result.score:.2f}):\n{result.content}\n"
                )

        # Add image context
        if image_results and self.include_image_captions:
            context_parts.append("\n## Image Context\n")
            for i, result in enumerate(image_results[:self.max_images_in_prompt], 1):
                source_name = Path(result.source).name
                context_parts.append(f"\n[{i}] From {source_name} (page {result.page_number}):")

                if self.include_image_captions and result.metadata.get("caption"):
                    context_parts.append(f"  Caption: {result.metadata['caption']}")

                if self.include_ocr_text and result.metadata.get("ocr_text"):
                    ocr_preview = result.metadata["ocr_text"][:200]
                    context_parts.append(f"  Text in image: {ocr_preview}...")

                context_parts.append("")

        # Add table context
        if table_results and self.include_table_descriptions:
            context_parts.append("\n## Table Context\n")
            for i, result in enumerate(table_results[:self.max_tables_in_prompt], 1):
                source_name = Path(result.source).name
                context_parts.append(f"\n[{i}] From {source_name} (page {result.page_number}):")

                if result.metadata.get("description"):
                    context_parts.append(f"  Description: {result.metadata['description']}")

                # Add table preview
                if result.content is not None:
                    try:
                        preview_df = result.content.head(3)
                        markdown = preview_df.to_markdown(index=False)
                        context_parts.append(f"  Preview:\n{markdown}")
                    except Exception:
                        pass

                context_parts.append("")

        return "\n".join(context_parts)

    def _generate_with_vision(
        self,
        question: str,
        context: str,
        images: List[bytes] = [],
    ) -> str:
        """
        Generate response using vision LLM with images.

        Args:
            question: User question
            context: Text context
            images: List of image bytes

        Returns:
            Generated answer
        """
        if self.vision_llm is None:
            # Fallback to text-only generation
            return self._generate_text_only(question, context)

        try:
            # Build prompt
            prompt = f"""{self.system_prompt}

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the provided context. Include specific citations to the sources you use."""

            # If images available, use vision LLM
            if images and self.vision_llm:
                response = self.vision_llm.generate(
                    prompt=prompt,
                    images=images,
                )
                return response.content

            # Text-only fallback
            else:
                return self._generate_text_only(question, context)

        except Exception as e:
            logger.error(f"Vision LLM generation failed: {e}")
            return self._generate_text_only(question, context)

    def _generate_text_only(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Generate response using text-only LLM.

        Args:
            question: User question
            context: Text context

        Returns:
            Generated answer
        """
        if self.text_llm is None:
            # Return context as fallback
            return f"Context:\n{context}\n\nNote: No LLM available for generation."

        try:
            prompt = f"""{self.system_prompt}

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the provided context."""

            # Use text LLM (might be vision LLM without images)
            if hasattr(self.text_llm, 'generate') and not hasattr(self.text_llm, 'vision_client'):
                # Standard text LLM
                response = self.text_llm.generate(
                    prompt=prompt,
                )
                return response.content if hasattr(response, 'content') else str(response)

            elif hasattr(self.text_llm, 'chat'):
                # Chat-style LLM
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
                response = self.text_llm.chat(messages=messages)
                return response.content if hasattr(response, 'content') else str(response)

            else:
                # Fallback
                return f"Context retrieved:\n{context}"

        except Exception as e:
            logger.error(f"Text LLM generation failed: {e}")
            return f"Error generating response: {e}"

    def _generate_stream(
        self,
        question: str,
        context: str,
        images: List[bytes] = [],
    ):
        """
        Generate streaming response.

        Args:
            question: User question
            context: Text context
            images: List of image bytes

        Yields:
            Response chunks
        """
        if self.vision_llm is None or not hasattr(self.vision_llm, 'stream_generate'):
            # Non-streaming fallback
            yield self._generate_with_vision(question, context, images)
            return

        try:
            prompt = f"""{self.system_prompt}

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the provided context."""

            for chunk in self.vision_llm.stream_generate(
                prompt=prompt,
                images=images,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {e}"

    def _create_text_citations(self, results: List) -> List[TextCitation]:
        """Create text citations from results."""
        citations = []

        for result in results:
            citation = TextCitation(
                content=result.content,
                source=result.source,
                page_number=result.page_number,
                score=result.score,
                metadata=result.metadata.copy(),
            )
            citations.append(citation)

        return citations

    def _create_image_citations(self, results: List) -> List[ImageCitation]:
        """Create image citations from results."""
        citations = []

        for result in results[:self.max_images_in_prompt]:
            citation = ImageCitation(
                content=result.preview,
                source=result.source,
                page_number=result.page_number,
                score=result.score,
                image_data=result.content if isinstance(result.content, bytes) else None,
                caption=result.metadata.get("caption"),
                ocr_text=result.metadata.get("ocr_text"),
                metadata=result.metadata.copy(),
            )
            citations.append(citation)

        return citations

    def _create_table_citations(self, results: List) -> List[TableCitation]:
        """Create table citations from results."""
        citations = []

        for result in results[:self.max_tables_in_prompt]:
            df = result.content
            row_count = len(df) if df is not None else 0
            col_count = len(df.columns) if df is not None and hasattr(df, 'columns') else 0

            citation = TableCitation(
                content=result.preview,
                source=result.source,
                page_number=result.page_number,
                score=result.score,
                dataframe=df,
                description=result.metadata.get("description"),
                row_count=row_count,
                col_count=col_count,
                metadata=result.metadata.copy(),
            )
            citations.append(citation)

        return citations

    def query_with_image(
        self,
        question: str,
        query_image: Union[bytes, str, Path],
        include_text: bool = True,
        include_tables: bool = True,
        top_k: int = 10,
    ) -> MultiModalResponse:
        """
        Query with an image (cross-modal search).

        Args:
            question: User question
            query_image: Image to use as query
            include_text: Whether to include text results
            include_tables: Whether to include table results
            top_k: Number of results

        Returns:
            MultiModalResponse
        """
        logger.info(f"Processing image query: {question[:100]}...")

        # Retrieve using image
        results = self.retriever.retrieve_by_image(
            image=query_image,
            content_types="all",
            top_k=top_k,
        )

        # Filter results
        text_results = [r for r in results if r.content_type == "text"] if include_text else []
        image_results = [r for r in results if r.content_type == "image"]
        table_results = [r for r in results if r.content_type == "table"] if include_tables else []

        # Build context
        context_text = self._build_multimodal_context(
            text_results=text_results,
            image_results=image_results,
            table_results=table_results,
        )

        # Add query image to images list
        images_for_llm = [query_image]
        images_for_llm.extend([
            r.content for r in image_results[:self.max_images_in_prompt - 1]
        ])

        # Generate response
        answer = self._generate_with_vision(
            question=question,
            context=context_text,
            images=images_for_llm,
        )

        # Create citations
        text_citations = self._create_text_citations(text_results)
        image_citations = self._create_image_citations(image_results)
        table_citations = self._create_table_citations(table_results)

        response = MultiModalResponse(
            answer=answer,
            text_citations=text_citations,
            image_citations=image_citations,
            table_citations=table_citations,
            query=question,
            sources_used=["image_query"] + (
                ["text"] if text_citations else []
            ) + (
                ["images"] if image_citations else []
            ) + (
                ["tables"] if table_citations else []
            ),
        )

        return response

    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[MultiModalResponse]:
        """
        Process multiple queries in batch.

        Args:
            questions: List of questions
            **kwargs: Additional arguments for query()

        Returns:
            List of MultiModalResponse objects
        """
        responses = []

        for question in questions:
            try:
                response = self.query(question, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to process question: {question[:50]}... Error: {e}")
                responses.append(
                    MultiModalResponse(
                        answer=f"Error processing question: {e}",
                        query=question,
                    )
                )

        return responses
