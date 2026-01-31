"""
Image Processor - Multi-modal image processing for RAG systems.

This module provides comprehensive image processing capabilities including:
- Image extraction from PDF and DOCX documents
- Caption generation using GPT-4V (cloud) or LLaVA (local)
- OCR text extraction from images
- CLIP embeddings for semantic search
"""

import io
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np

logger = logging.getLogger(__name__)


class CaptioningBackend(Enum):
    """Caption generation backend options."""
    GPT4V = "gpt4v"
    LLAVA = "llava"
    BLIP = "blip"
    NONE = "none"


class OCREngine(Enum):
    """OCR engine options."""
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"
    AUTO = "auto"


@dataclass
class ImageData:
    """
    Container for processed image data and metadata.

    Attributes:
        image_bytes: Raw image data as bytes
        caption: Generated caption describing the image
        ocr_text: Text extracted from the image via OCR
        source_doc: Path to the source document
        page_number: Page number where image was found
        image_index: Index of the image on the page
        embedding: CLIP embedding vector for semantic search
        format: Image format (png, jpeg, etc.)
        size: Image dimensions (width, height)
        metadata: Additional metadata dictionary
    """
    image_bytes: bytes
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    source_doc: str = ""
    page_number: int = 0
    image_index: int = 0
    embedding: Optional[np.ndarray] = None
    format: str = "png"
    size: Tuple[int, int] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_pil(self):
        """Convert bytes to PIL Image."""
        try:
            from PIL import Image
            return Image.open(io.BytesIO(self.image_bytes))
        except ImportError:
            raise ImportError("PIL is required to convert image bytes to PIL Image")

    def save(self, path: Union[str, Path]):
        """Save image to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(self.image_bytes)

    def to_base64(self) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(self.image_bytes).decode("utf-8")

    def get_mime_type(self) -> str:
        """Get MIME type based on format."""
        mime_types = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }
        return mime_types.get(self.format.lower(), "image/png")

    def __repr__(self) -> str:
        return (
            f"ImageData(source={self.source_doc}, page={self.page_number}, "
            f"index={self.image_index}, format={self.format}, "
            f"has_caption={self.caption is not None}, "
            f"has_ocr={self.ocr_text is not None})"
        )


class ImageProcessor:
    """
    Advanced image processor for multi-modal RAG systems.

    Features:
    - Extract images from PDF and DOCX documents
    - Generate captions using GPT-4V or local models
    - Extract text from images using OCR
    - Generate CLIP embeddings for semantic search
    - Batch processing support
    """

    def __init__(
        self,
        captioning_backend: CaptioningBackend = CaptioningBackend.GPT4V,
        ocr_engine: OCREngine = OCREngine.TESSERACT,
        clip_model: str = "ViT-B/32",
        device: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        min_image_size: Tuple[int, int] = (100, 100),
        extract_metadata: bool = True,
    ):
        """
        Initialize the ImageProcessor.

        Args:
            captioning_backend: Backend for caption generation
            ocr_engine: OCR engine to use for text extraction
            clip_model: CLIP model name for embeddings
            device: Device to use for models ('cuda', 'cpu', or None for auto)
            openai_api_key: OpenAI API key for GPT-4V
            anthropic_api_key: Anthropic API key for Claude Vision
            min_image_size: Minimum image size (width, height) to include
            extract_metadata: Whether to extract image metadata
        """
        self.captioning_backend = captioning_backend
        self.ocr_engine = ocr_engine
        self.clip_model_name = clip_model
        self.min_image_size = min_image_size
        self.extract_metadata = extract_metadata

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
        self.clip_model = None
        self.clip_preprocess = None
        self.ocr_processor = None
        self.vision_client = None

        self._init_clip()
        self._init_ocr()
        self._init_vision_llm(openai_api_key, anthropic_api_key)

        logger.info(f"ImageProcessor initialized with device={self.device}")

    def _init_clip(self):
        """Initialize CLIP model for embeddings."""
        try:
            import clip
            import torch

            self.clip_model, self.clip_preprocess = clip.load(
                self.clip_model_name,
                device=self.device,
                download_root=str(Path.home() / ".cache" / "clip")
            )
            self.clip_model.eval()
            logger.info(f"CLIP model {self.clip_model_name} loaded successfully")
        except ImportError:
            logger.warning("CLIP not available. Install with: pip install openai-clip")
        except Exception as e:
            logger.warning(f"Failed to initialize CLIP: {e}")

    def _init_ocr(self):
        """Initialize OCR processor."""
        try:
            from .extraction.ocr_processor import OCRProcessor

            self.ocr_processor = OCRProcessor(
                engine=self.ocr_engine,
                confidence_threshold=0.5,
            )
            logger.info(f"OCR processor initialized with {self.ocr_engine.value}")
        except ImportError:
            logger.warning("OCR processor not available")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR: {e}")

    def _init_vision_llm(
        self,
        openai_api_key: Optional[str],
        anthropic_api_key: Optional[str],
    ):
        """Initialize vision LLM client for captioning."""
        if self.captioning_backend == CaptioningBackend.NONE:
            return

        try:
            from .vision_llm import VisionLLM, VisionProvider

            if self.captioning_backend == CaptioningBackend.GPT4V:
                provider = VisionProvider.OPENAI
                api_key = openai_api_key
            elif self.captioning_backend == CaptioningBackend.LLAVA:
                provider = VisionProvider.LLAVA
                api_key = None
            else:
                provider = VisionProvider.OPENAI
                api_key = openai_api_key

            self.vision_client = VisionLLM(
                provider=provider,
                api_key=api_key,
                temperature=0.7,
            )
            logger.info(f"Vision LLM client initialized: {self.captioning_backend.value}")
        except ImportError:
            logger.warning("Vision LLM not available")
        except Exception as e:
            logger.warning(f"Failed to initialize vision LLM: {e}")

    def extract_images_from_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        extract_embeddings: bool = True,
        generate_captions: bool = True,
        extract_ocr: bool = True,
    ) -> List[ImageData]:
        """
        Extract images from a PDF document.

        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract (None for all pages)
            extract_embeddings: Whether to generate CLIP embeddings
            generate_captions: Whether to generate image captions
            extract_ocr: Whether to extract text via OCR

        Returns:
            List of ImageData objects

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If PDF processing fails
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting images from PDF: {pdf_path}")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF image extraction. "
                "Install with: pip install PyMuPDF"
            )

        extracted_images = []

        try:
            doc = fitz.open(str(pdf_path))

            for page_num in range(len(doc)):
                if pages is not None and (page_num + 1) not in pages:
                    continue

                page = doc[page_num]
                image_list = page.get_images(full=True)

                logger.debug(f"Found {len(image_list)} images on page {page_num + 1}")

                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]

                        # Extract image bytes
                        base_image = doc.extract_image(xref)

                        if not base_image:
                            continue

                        image_bytes = base_image["image"]
                        image_format = base_image.get("ext", "png")

                        # Check size
                        try:
                            from PIL import Image
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size

                            if width < self.min_image_size[0] or height < self.min_image_size[1]:
                                logger.debug(
                                    f"Skipping small image {width}x{height} on page {page_num + 1}"
                                )
                                continue
                        except Exception as e:
                            logger.warning(f"Could not check image size: {e}")
                            width, height = 0, 0

                        # Create ImageData
                        image_data = ImageData(
                            image_bytes=image_bytes,
                            source_doc=str(pdf_path),
                            page_number=page_num + 1,
                            image_index=img_index,
                            format=image_format,
                            size=(width, height),
                        )

                        # Generate embedding
                        if extract_embeddings:
                            image_data.embedding = self.get_image_embedding(image_bytes)

                        # Generate caption
                        if generate_captions:
                            image_data.caption = self.generate_caption(image_bytes)

                        # Extract OCR text
                        if extract_ocr:
                            image_data.ocr_text = self.extract_text_ocr(image_bytes)

                        extracted_images.append(image_data)
                        logger.debug(
                            f"Extracted image {img_index + 1} from page {page_num + 1}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num + 1}: {e}"
                        )
                        continue

            doc.close()
            logger.info(f"Extracted {len(extracted_images)} images from PDF")

        except Exception as e:
            raise RuntimeError(f"Failed to process PDF: {e}")

        return extracted_images

    def extract_images_from_docx(
        self,
        docx_path: Union[str, Path],
        extract_embeddings: bool = True,
        generate_captions: bool = True,
        extract_ocr: bool = True,
    ) -> List[ImageData]:
        """
        Extract images from a DOCX document.

        Args:
            docx_path: Path to the DOCX file
            extract_embeddings: Whether to generate CLIP embeddings
            generate_captions: Whether to generate image captions
            extract_ocr: Whether to extract text via OCR

        Returns:
            List of ImageData objects

        Raises:
            FileNotFoundError: If DOCX file doesn't exist
            RuntimeError: If DOCX processing fails
        """
        docx_path = Path(docx_path)

        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        logger.info(f"Extracting images from DOCX: {docx_path}")

        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX image extraction. "
                "Install with: pip install python-docx"
            )

        extracted_images = []

        try:
            doc = Document(docx_path)

            # Extract images from document relationships
            image_index = 0

            for rel in doc.part.rels.values():
                if "image" not in rel.target_ref:
                    continue

                try:
                    # Get image bytes
                    image_part = rel.target_part
                    image_bytes = image_part.blob

                    # Check size
                    try:
                        from PIL import Image
                        img = Image.open(io.BytesIO(image_bytes))
                        width, height = img.size

                        if width < self.min_image_size[0] or height < self.min_image_size[1]:
                            logger.debug(f"Skipping small image {width}x{height}")
                            continue
                    except Exception:
                        width, height = 0, 0

                    # Get image format
                    image_format = img.format or "png" if 'img' in locals() else "png"

                    # Create ImageData
                    image_data = ImageData(
                        image_bytes=image_bytes,
                        source_doc=str(docx_path),
                        page_number=1,  # DOCX doesn't have pages
                        image_index=image_index,
                        format=image_format.lower(),
                        size=(width, height),
                    )

                    # Generate embedding
                    if extract_embeddings:
                        image_data.embedding = self.get_image_embedding(image_bytes)

                    # Generate caption
                    if generate_captions:
                        image_data.caption = self.generate_caption(image_bytes)

                    # Extract OCR text
                    if extract_ocr:
                        image_data.ocr_text = self.extract_text_ocr(image_bytes)

                    extracted_images.append(image_data)
                    image_index += 1
                    logger.debug(f"Extracted image {image_index} from DOCX")

                except Exception as e:
                    logger.warning(f"Failed to extract image from DOCX: {e}")
                    continue

            logger.info(f"Extracted {len(extracted_images)} images from DOCX")

        except Exception as e:
            raise RuntimeError(f"Failed to process DOCX: {e}")

        return extracted_images

    def generate_caption(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str, Path],
        prompt: str = "Describe this image in detail. Focus on the main subject, key elements, and any text present.",
    ) -> Optional[str]:
        """
        Generate a caption for an image using vision LLM.

        Args:
            image: Image as bytes, PIL Image, numpy array, or file path
            prompt: Caption generation prompt

        Returns:
            Generated caption or None if generation fails
        """
        if self.vision_client is None:
            logger.warning("Vision LLM client not initialized")
            return None

        try:
            # Convert to PIL Image if needed
            if isinstance(image, bytes):
                from PIL import Image
                pil_image = Image.open(io.BytesIO(image))
            elif isinstance(image, (str, Path)):
                pil_image = image
            elif isinstance(image, np.ndarray):
                from PIL import Image
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Generate caption
            response = self.vision_client.analyze_image(pil_image, prompt)
            caption = response.content.strip()

            logger.debug(f"Generated caption: {caption[:100]}...")
            return caption

        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return None

    def extract_text_ocr(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str, Path],
    ) -> Optional[str]:
        """
        Extract text from an image using OCR.

        Args:
            image: Image as bytes, PIL Image, numpy array, or file path

        Returns:
            Extracted text or None if OCR fails
        """
        if self.ocr_processor is None:
            logger.warning("OCR processor not initialized")
            return None

        try:
            # Convert to appropriate format
            if isinstance(image, bytes):
                from PIL import Image
                img_array = np.array(Image.open(io.BytesIO(image)))
            elif isinstance(image, (str, Path)):
                img_array = str(image)
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                # PIL Image
                img_array = np.array(image)

            # Run OCR
            result = self.ocr_processor.extract_text(img_array)

            if result and result.text:
                logger.debug(f"Extracted {len(result.text)} characters via OCR")
                return result.text

            return None

        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None

    def get_image_embedding(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str, Path],
    ) -> Optional[np.ndarray]:
        """
        Generate CLIP embedding for an image.

        Args:
            image: Image as bytes, PIL Image, numpy array, or file path

        Returns:
            Embedding vector as numpy array or None if generation fails
        """
        if self.clip_model is None:
            logger.warning("CLIP model not initialized")
            return None

        try:
            import torch
            from PIL import Image

            # Convert to PIL Image
            if isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image))
            elif isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Preprocess and encode
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)

            embedding = image_features.cpu().numpy()[0]
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)

            logger.debug(f"Generated embedding with shape {embedding.shape}")
            return embedding

        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def process_image(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str, Path],
        source_doc: str = "",
        page_number: int = 0,
        image_index: int = 0,
        generate_caption: bool = True,
        extract_ocr: bool = True,
        generate_embedding: bool = True,
    ) -> ImageData:
        """
        Process a single image with all capabilities.

        Args:
            image: Image to process
            source_doc: Source document path
            page_number: Page number
            image_index: Image index
            generate_caption: Whether to generate caption
            extract_ocr: Whether to extract OCR text
            generate_embedding: Whether to generate embedding

        Returns:
            ImageData with all extracted information
        """
        # Convert to bytes
        if isinstance(image, bytes):
            image_bytes = image
        elif isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, np.ndarray):
            from PIL import Image
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        else:
            # PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        # Get image info
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            size = img.size
            format = img.format or "png"
        except Exception:
            size = (0, 0)
            format = "png"

        # Create ImageData
        image_data = ImageData(
            image_bytes=image_bytes,
            source_doc=source_doc,
            page_number=page_number,
            image_index=image_index,
            format=format.lower(),
            size=size,
        )

        # Generate embedding
        if generate_embedding:
            image_data.embedding = self.get_image_embedding(image_bytes)

        # Generate caption
        if generate_caption:
            image_data.caption = self.generate_caption(image_bytes)

        # Extract OCR
        if extract_ocr:
            image_data.ocr_text = self.extract_text_ocr(image_bytes)

        return image_data

    def batch_process(
        self,
        images: List[Union[bytes, Image.Image, np.ndarray, str, Path]],
        **kwargs
    ) -> List[ImageData]:
        """
        Process multiple images in batch.

        Args:
            images: List of images
            **kwargs: Additional arguments for process_image

        Returns:
            List of ImageData objects
        """
        results = []

        for i, image in enumerate(images):
            try:
                result = self.process_image(image, image_index=i, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process image {i}: {e}")
                continue

        return results

    def save_images(
        self,
        images: List[ImageData],
        output_dir: Union[str, Path],
        naming_pattern: str = "doc_{doc}_page_{page}_img_{idx}.{ext}",
    ) -> List[Path]:
        """
        Save processed images to directory.

        Args:
            images: List of ImageData objects
            output_dir: Output directory path
            naming_pattern: Naming pattern with placeholders:
                {doc} = source document name
                {page} = page number
                {idx} = image index
                {ext} = image format

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for img_data in images:
            try:
                # Generate filename
                doc_name = Path(img_data.source_doc).stem if img_data.source_doc else "unknown"

                filename = naming_pattern.format(
                    doc=doc_name,
                    page=img_data.page_number,
                    idx=img_data.image_index,
                    ext=img_data.format,
                )

                output_path = output_dir / filename
                img_data.save(output_path)
                saved_paths.append(output_path)

                logger.debug(f"Saved image to {output_path}")

            except Exception as e:
                logger.warning(f"Failed to save image: {e}")
                continue

        return saved_paths


# Convenience functions
def extract_pdf_images(
    pdf_path: Union[str, Path],
    **kwargs
) -> List[ImageData]:
    """
    Convenience function to extract images from PDF.

    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional arguments for ImageProcessor

    Returns:
        List of ImageData objects
    """
    processor = ImageProcessor(**kwargs)
    return processor.extract_images_from_pdf(pdf_path)


def extract_docx_images(
    docx_path: Union[str, Path],
    **kwargs
) -> List[ImageData]:
    """
    Convenience function to extract images from DOCX.

    Args:
        docx_path: Path to DOCX file
        **kwargs: Additional arguments for ImageProcessor

    Returns:
        List of ImageData objects
    """
    processor = ImageProcessor(**kwargs)
    return processor.extract_images_from_docx(docx_path)


def process_single_image(
    image: Union[bytes, Image.Image, np.ndarray, str, Path],
    **kwargs
) -> ImageData:
    """
    Convenience function to process a single image.

    Args:
        image: Image to process
        **kwargs: Additional arguments for ImageProcessor

    Returns:
        ImageData object
    """
    processor = ImageProcessor(**kwargs)
    return processor.process_image(image)
