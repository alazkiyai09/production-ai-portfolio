"""
Multi-Modal Retriever - Unified search across text, images, and tables.

This module provides comprehensive multi-modal retrieval capabilities including:
- Unified search across text, images, and tables
- Content-type filtering
- Combined scoring with configurable weights
- Rich results with previews and metadata
"""

import io
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import base64

import numpy as np

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type identifiers."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    ALL = "all"


@dataclass
class MultiModalResult:
    """
    Unified result from multi-modal retrieval.

    Attributes:
        content_type: Type of content (text, image, table)
        content: The actual content (text, image bytes, or DataFrame)
        preview: Text preview/summary of the content
        metadata: Additional metadata about the result
        score: Combined relevance score
        text_score: Score from text similarity (if applicable)
        image_score: Score from image similarity (if applicable)
        table_score: Score from table similarity (if applicable)
        source: Source document or file path
        page_number: Page number (for document sources)
    """
    content_type: Literal["text", "image", "table"]
    content: Union[str, bytes, Any]  # Any for pd.DataFrame
    preview: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    text_score: Optional[float] = None
    image_score: Optional[float] = None
    table_score: Optional[float] = None
    source: str = ""
    page_number: int = 0

    def get_preview(self, max_length: int = 200) -> str:
        """Get a text preview of the content."""
        if self.content_type == "text":
            text = self.content if isinstance(self.content, str) else str(self.content)
            if len(text) > max_length:
                return text[:max_length] + "..."
            return text

        elif self.content_type == "image":
            return f"[Image] {self.metadata.get('caption', 'No caption')}"

        elif self.content_type == "table":
            if self.content is not None:
                try:
                    # Get first few rows as preview
                    return self.content.head(3).to_markdown(index=False)
                except Exception:
                    return f"[Table] {self.metadata.get('description', 'No description')}"
            return "[Table] Empty"

        return str(self.content)[:max_length]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "content_type": self.content_type,
            "preview": self.preview,
            "score": self.score,
            "text_score": self.text_score,
            "image_score": self.image_score,
            "table_score": self.table_score,
            "source": self.source,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"MultiModalResult(type={self.content_type}, "
            f"score={self.score:.3f}, "
            f"source={self.source}, "
            f"preview={self.preview[:50]}...)"
        )


@dataclass
class SearchResult:
    """Intermediate text search result."""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResult:
    """Intermediate image search result."""
    image_data: bytes
    score: float
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableResult:
    """Intermediate table search result."""
    dataframe: Any  # pd.DataFrame
    score: float
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiModalRetriever:
    """
    Unified multi-modal retriever for text, images, and tables.

    Features:
    - Search across text, images, and tables simultaneously
    - Content-type filtering
    - Configurable scoring weights for each modality
    - CLIP-based cross-modal search
    - Rich result previews
    - Batch indexing support
    """

    def __init__(
        self,
        text_weight: float = 0.5,
        image_weight: float = 0.3,
        table_weight: float = 0.2,
        text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model: str = "ViT-B/32",
        device: Optional[str] = None,
        enable_clip: bool = True,
        enable_reranking: bool = True,
        rerank_top_k: int = 20,
    ):
        """
        Initialize the multi-modal retriever.

        Args:
            text_weight: Weight for text similarity in combined score
            image_weight: Weight for image similarity in combined score
            table_weight: Weight for table similarity in combined score
            text_embedding_model: Sentence transformer model for text
            clip_model: CLIP model for image embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
            enable_clip: Whether to enable CLIP for cross-modal search
            enable_reranking: Whether to apply cross-encoder reranking
            rerank_top_k: Number of results to rerank
        """
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.table_weight = table_weight
        self.text_embedding_model_name = text_embedding_model
        self.clip_model_name = clip_model
        self.enable_clip = enable_clip
        self.enable_reranking = enable_reranking
        self.rerank_top_k = rerank_top_k

        # Normalize weights
        total_weight = text_weight + image_weight + table_weight
        if total_weight > 0:
            self.text_weight = text_weight / total_weight
            self.image_weight = image_weight / total_weight
            self.table_weight = table_weight / total_weight

        # Determine device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Initialize models
        self.text_encoder = None
        self.clip_model = None
        self.clip_preprocess = None
        self.reranker = None

        # Storage
        self.text_nodes: List[Dict[str, Any]] = []
        self.image_nodes: List[Dict[str, Any]] = []
        self.table_nodes: List[Dict[str, Any]] = []
        self.index_built = False

        self._init_models()
        logger.info(
            f"MultiModalRetriever initialized: device={self.device}, "
            f"weights=(text={self.text_weight:.2f}, "
            f"image={self.image_weight:.2f}, "
            f"table={self.table_weight:.2f})"
        )

    def _init_models(self):
        """Initialize embedding models."""
        # Initialize text encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(self.text_embedding_model_name)
            self.text_encoder.to(self.device)
            logger.debug(f"Text encoder loaded: {self.text_embedding_model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available")
        except Exception as e:
            logger.warning(f"Failed to load text encoder: {e}")

        # Initialize CLIP
        if self.enable_clip:
            try:
                import clip
                import torch
                self.clip_model, self.clip_preprocess = clip.load(
                    self.clip_model_name,
                    device=self.device,
                    download_root=str(Path.home() / ".cache" / "clip")
                )
                self.clip_model.eval()
                logger.debug(f"CLIP model loaded: {self.clip_model_name}")
            except ImportError:
                logger.warning("CLIP not available")
            except Exception as e:
                logger.warning(f"Failed to load CLIP: {e}")

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding."""
        if self.text_encoder is None:
            return None

        try:
            embedding = self.text_encoder.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.warning(f"Text encoding failed: {e}")
            return None

    def encode_text_clip(self, text: str) -> Optional[np.ndarray]:
        """Encode text using CLIP (for cross-modal search)."""
        if self.clip_model is None:
            return None

        try:
            import clip
            import torch

            text_input = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)

            embedding = text_features.cpu().numpy()[0]
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.warning(f"CLIP text encoding failed: {e}")
            return None

    def add_text_node(
        self,
        node_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a text document to the index.

        Args:
            node_id: Unique identifier
            content: Text content
            metadata: Additional metadata

        Returns:
            Node ID
        """
        embedding = self.encode_text(content)

        node = {
            "id": node_id,
            "content": content,
            "content_type": ContentType.TEXT,
            "embedding": embedding,
            "metadata": metadata or {},
        }

        self.text_nodes.append(node)
        self.index_built = False

        logger.debug(f"Added text node: {node_id}")
        return node_id

    def add_image_node(
        self,
        node_id: str,
        image: Union[bytes, str, Path],
        content: str,
        caption: Optional[str] = None,
        ocr_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an image to the index.

        Args:
            node_id: Unique identifier
            image: Image bytes or file path
            content: Image description
            caption: Generated caption
            ocr_text: Extracted OCR text
            metadata: Additional metadata

        Returns:
            Node ID
        """
        # Encode image if CLIP available
        image_embedding = None
        if self.enable_clip and self.clip_model is not None:
            image_embedding = self._encode_image(image)

        # Combine text and image embeddings
        text_embedding = self.encode_text(content)
        if image_embedding is not None and text_embedding is not None:
            combined = (text_embedding + image_embedding) / 2
        elif text_embedding is not None:
            combined = text_embedding
        else:
            combined = image_embedding

        # Read image bytes
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = image

        node = {
            "id": node_id,
            "content": content,
            "caption": caption,
            "ocr_text": ocr_text,
            "image_bytes": image_bytes,
            "content_type": ContentType.IMAGE,
            "embedding": combined,
            "metadata": metadata or {},
        }

        self.image_nodes.append(node)
        self.index_built = False

        logger.debug(f"Added image node: {node_id}")
        return node_id

    def _encode_image(self, image: Union[bytes, str, Path]) -> Optional[np.ndarray]:
        """Encode image to CLIP embedding."""
        if self.clip_model is None:
            return None

        try:
            from PIL import Image
            import torch

            # Load image
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            else:
                pil_image = Image.open(io.BytesIO(image)).convert("RGB")

            # Preprocess and encode
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)

            embedding = image_features.cpu().numpy()[0]
            return embedding / np.linalg.norm(embedding)

        except Exception as e:
            logger.warning(f"Image encoding failed: {e}")
            return None

    def add_table_node(
        self,
        node_id: str,
        dataframe: Any,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a table to the index.

        Args:
            node_id: Unique identifier
            dataframe: pandas DataFrame
            description: Table description
            metadata: Additional metadata

        Returns:
            Node ID
        """
        # Encode description
        embedding = self.encode_text(description)

        node = {
            "id": node_id,
            "dataframe": dataframe,
            "description": description,
            "content_type": ContentType.TABLE,
            "embedding": embedding,
            "metadata": metadata or {},
        }

        self.table_nodes.append(node)
        self.index_built = False

        logger.debug(f"Added table node: {node_id}")
        return node_id

    def retrieve(
        self,
        query: str,
        content_types: Union[str, List[str]] = "all",
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[MultiModalResult]:
        """
        Retrieve relevant content across all modalities.

        Args:
            query: Search query
            content_types: Content types to search ('all', 'text', 'image', 'table', or list)
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of MultiModalResult objects
        """
        # Normalize content types
        if isinstance(content_types, str):
            if content_types.lower() == "all":
                types_to_search = [ContentType.TEXT, ContentType.IMAGE, ContentType.TABLE]
            else:
                types_to_search = [ContentType(content_types.lower())]
        else:
            types_to_search = [ContentType(ct.lower()) for ct in content_types]

        logger.info(f"Searching for '{query}' across {[t.value for t in types_to_search]}")

        # Encode query
        text_query_emb = self.encode_text(query)
        clip_query_emb = self.encode_text_clip(query) if self.enable_clip else None

        # Search each modality
        all_results = []

        if ContentType.TEXT in types_to_search:
            text_results = self._search_text(query, text_query_emb, top_k * 2)
            all_results.extend([(r, "text") for r in text_results])

        if ContentType.IMAGE in types_to_search:
            image_results = self._search_images(query, clip_query_emb, top_k * 2)
            all_results.extend([(r, "image") for r in image_results])

        if ContentType.TABLE in types_to_search:
            table_results = self._search_tables(query, text_query_emb, top_k * 2)
            all_results.extend([(r, "table") for r in table_results])

        # Merge and rank
        merged = self._merge_results(all_results, top_k)

        # Filter by minimum score
        filtered = [r for r in merged if r.score >= min_score]

        logger.info(f"Retrieved {len(filtered)} results")
        return filtered[:top_k]

    def _search_text(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
    ) -> List[SearchResult]:
        """Search text documents."""
        if not self.text_nodes:
            return []

        results = []

        for node in self.text_nodes:
            if node["embedding"] is None or query_embedding is None:
                continue

            # Cosine similarity
            similarity = float(np.dot(query_embedding, node["embedding"]))

            # Apply text weight
            weighted_score = similarity * self.text_weight

            results.append(
                SearchResult(
                    content=node["content"],
                    score=weighted_score,
                    metadata={
                        "node_id": node["id"],
                        "source": node["metadata"].get("source", ""),
                        "page_number": node["metadata"].get("page_number", 0),
                        **node["metadata"],
                    },
                )
            )

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _search_images(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
    ) -> List[ImageResult]:
        """Search images."""
        if not self.image_nodes:
            return []

        results = []

        for node in self.image_nodes:
            node_score = 0.0

            # CLIP similarity
            if node["embedding"] is not None and query_embedding is not None:
                clip_sim = float(np.dot(query_embedding, node["embedding"]))
                node_score += clip_sim * 0.7

            # Text similarity with caption/OCR
            if self.text_encoder is not None:
                # Search in caption
                if node.get("caption"):
                    caption_emb = self.encode_text(node["caption"])
                    if caption_emb is not None and query_embedding is not None:
                        text_sim = float(np.dot(query_embedding, caption_emb))
                        node_score += text_sim * 0.2

                # Search in OCR text
                if node.get("ocr_text"):
                    ocr_emb = self.encode_text(node["ocr_text"])
                    if ocr_emb is not None and query_embedding is not None:
                        text_sim = float(np.dot(query_embedding, ocr_emb))
                        node_score += text_sim * 0.1

            # Apply image weight
            weighted_score = node_score * self.image_weight

            results.append(
                ImageResult(
                    image_data=node["image_bytes"],
                    score=weighted_score,
                    caption=node.get("caption"),
                    ocr_text=node.get("ocr_text"),
                    metadata={
                        "node_id": node["id"],
                        "source": node["metadata"].get("source", ""),
                        "page_number": node["metadata"].get("page_number", 0),
                        **node["metadata"],
                    },
                )
            )

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _search_tables(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
    ) -> List[TableResult]:
        """Search tables."""
        if not self.table_nodes:
            return []

        results = []

        for node in self.table_nodes:
            if node["embedding"] is None or query_embedding is None:
                continue

            # Cosine similarity
            similarity = float(np.dot(query_embedding, node["embedding"]))

            # Apply table weight
            weighted_score = similarity * self.table_weight

            results.append(
                TableResult(
                    dataframe=node["dataframe"],
                    score=weighted_score,
                    description=node.get("description"),
                    metadata={
                        "node_id": node["id"],
                        "source": node["metadata"].get("source", ""),
                        "page_number": node["metadata"].get("page_number", 0),
                        **node["metadata"],
                    },
                )
            )

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _merge_results(
        self,
        results: List[Tuple[Any, str]],
        top_k: int,
    ) -> List[MultiModalResult]:
        """
        Merge results from different modalities.

        Args:
            results: List of (result, content_type) tuples
            top_k: Number of results to return

        Returns:
            List of MultiModalResult objects
        """
        # Convert to MultiModalResult
        multimodal_results = []

        for result, content_type in results:
            if content_type == "text" and isinstance(result, SearchResult):
                multimodal_results.append(
                    MultiModalResult(
                        content_type="text",
                        content=result.content,
                        preview=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        score=result.score,
                        text_score=result.score,
                        metadata=result.metadata,
                        source=result.metadata.get("source", ""),
                        page_number=result.metadata.get("page_number", 0),
                    )
                )

            elif content_type == "image" and isinstance(result, ImageResult):
                # Generate preview
                preview_parts = []
                if result.caption:
                    preview_parts.append(f"Caption: {result.caption}")
                if result.ocr_text:
                    ocr_preview = result.ocr_text[:100]
                    preview_parts.append(f"OCR: {ocr_preview}...")
                preview = " | ".join(preview_parts) if preview_parts else "[Image]"

                multimodal_results.append(
                    MultiModalResult(
                        content_type="image",
                        content=result.image_data,
                        preview=preview,
                        score=result.score,
                        image_score=result.score,
                        metadata={
                            "caption": result.caption,
                            "ocr_text": result.ocr_text,
                            **result.metadata,
                        },
                        source=result.metadata.get("source", ""),
                        page_number=result.metadata.get("page_number", 0),
                    )
                )

            elif content_type == "table" and isinstance(result, TableResult):
                # Generate preview
                if result.dataframe is not None:
                    try:
                        preview = result.dataframe.head(3).to_markdown(index=False)
                    except Exception:
                        preview = result.description or "[Table]"
                else:
                    preview = result.description or "[Table]"

                multimodal_results.append(
                    MultiModalResult(
                        content_type="table",
                        content=result.dataframe,
                        preview=preview,
                        score=result.score,
                        table_score=result.score,
                        metadata={
                            "description": result.description,
                            **result.metadata,
                        },
                        source=result.metadata.get("source", ""),
                        page_number=result.metadata.get("page_number", 0),
                    )
                )

        # Sort by combined score
        multimodal_results.sort(key=lambda r: r.score, reverse=True)

        return multimodal_results[:top_k]

    def retrieve_by_image(
        self,
        image: Union[bytes, str, Path],
        content_types: Union[str, List[str]] = "all",
        top_k: int = 10,
    ) -> List[MultiModalResult]:
        """
        Retrieve using an image query (cross-modal search).

        Args:
            image: Query image
            content_types: Content types to search
            top_k: Number of results

        Returns:
            List of MultiModalResult objects
        """
        if not self.enable_clip or self.clip_model is None:
            logger.warning("CLIP not available for image query")
            return []

        # Encode query image
        query_embedding = self._encode_image(image)
        if query_embedding is None:
            return []

        # Normalize content types
        if isinstance(content_types, str):
            if content_types.lower() == "all":
                types_to_search = [ContentType.TEXT, ContentType.IMAGE, ContentType.TABLE]
            else:
                types_to_search = [ContentType(content_types.lower())]
        else:
            types_to_search = [ContentType(ct.lower()) for ct in content_types]

        all_results = []

        # Search images (higher weight for image-to-image)
        if ContentType.IMAGE in types_to_search:
            for node in self.image_nodes:
                if node["embedding"] is None:
                    continue

                similarity = float(np.dot(query_embedding, node["embedding"]))
                # Boost image-to-image similarity
                weighted_score = similarity * self.image_weight * 1.5

                all_results.append(
                    (
                        ImageResult(
                            image_data=node["image_bytes"],
                            score=weighted_score,
                            caption=node.get("caption"),
                            ocr_text=node.get("ocr_text"),
                            metadata={
                                "node_id": node["id"],
                                "source": node["metadata"].get("source", ""),
                                **node["metadata"],
                            },
                        ),
                        "image",
                    )
                )

        # Search text (CLIP-based)
        if ContentType.TEXT in types_to_search:
            for node in self.text_nodes:
                if node["embedding"] is None:
                    continue

                # Cross-modal similarity
                similarity = float(np.dot(query_embedding, node["embedding"]))
                weighted_score = similarity * self.text_weight * 0.5

                all_results.append(
                    (
                        SearchResult(
                            content=node["content"],
                            score=weighted_score,
                            metadata={
                                "node_id": node["id"],
                                "source": node["metadata"].get("source", ""),
                                **node["metadata"],
                            },
                        ),
                        "text",
                    )
                )

        # Search tables
        if ContentType.TABLE in types_to_search:
            for node in self.table_nodes:
                if node["embedding"] is None:
                    continue

                similarity = float(np.dot(query_embedding, node["embedding"]))
                weighted_score = similarity * self.table_weight * 0.5

                all_results.append(
                    (
                        TableResult(
                            dataframe=node["dataframe"],
                            score=weighted_score,
                            description=node.get("description"),
                            metadata={
                                "node_id": node["id"],
                                "source": node["metadata"].get("source", ""),
                                **node["metadata"],
                            },
                        ),
                        "table",
                    )
                )

        # Merge results
        return self._merge_results(all_results, top_k)

    def save_index(self, path: Union[str, Path]):
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        # Convert images to base64 for serialization
        serializable_nodes = []
        for node in self.image_nodes:
            node_copy = node.copy()
            node_copy["image_bytes"] = base64.b64encode(node["image_bytes"]).decode("utf-8")
            serializable_nodes.append(node_copy)

        data = {
            "text_nodes": self.text_nodes,
            "image_nodes": serializable_nodes,
            "table_nodes": self.table_nodes,
            "config": {
                "text_weight": self.text_weight,
                "image_weight": self.image_weight,
                "table_weight": self.table_weight,
                "text_embedding_model": self.text_embedding_model_name,
                "clip_model": self.clip_model_name,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        self.index_built = True
        logger.info(f"Index saved to {path}")

    def load_index(self, path: Union[str, Path]):
        """Load index from disk."""
        path = Path(path)

        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.text_nodes = data["text_nodes"]
        self.table_nodes = data["table_nodes"]

        # Convert images back from base64
        self.image_nodes = []
        for node in data["image_nodes"]:
            node_copy = node.copy()
            node_copy["image_bytes"] = base64.b64decode(node["image_bytes"])
            self.image_nodes.append(node_copy)

        self.index_built = True
        logger.info(f"Index loaded from {path}")

    def clear(self):
        """Clear all nodes from the index."""
        self.text_nodes = []
        self.image_nodes = []
        self.table_nodes = []
        self.index_built = False
        logger.info("Index cleared")

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_nodes": len(self.text_nodes) + len(self.image_nodes) + len(self.table_nodes),
            "text_nodes": len(self.text_nodes),
            "image_nodes": len(self.image_nodes),
            "table_nodes": len(self.table_nodes),
            "index_built": self.index_built,
            "has_text_encoder": self.text_encoder is not None,
            "has_clip": self.clip_model is not None,
            "device": self.device,
            "weights": {
                "text": self.text_weight,
                "image": self.image_weight,
                "table": self.table_weight,
            },
        }


# Convenience functions
def create_retriever(**kwargs) -> MultiModalRetriever:
    """Create a MultiModalRetriever with default or custom settings."""
    return MultiModalRetriever(**kwargs)


def search(
    query: str,
    retriever: MultiModalRetriever,
    content_types: Union[str, List[str]] = "all",
    top_k: int = 10,
) -> List[MultiModalResult]:
    """Convenience function for searching."""
    return retriever.retrieve(query, content_types, top_k)
