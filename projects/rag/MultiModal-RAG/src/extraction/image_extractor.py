"""
Image Extractor - Extract images from documents (PDFs, DOCX, etc.).

Supports:
- PDF image extraction
- DOCX image extraction
- Image preprocessing and optimization
- Batch extraction
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import io

# Optional dependencies
try:
    from PIL import Image
    import fitz  # PyMuPDF
    PIL_AVAILABLE = True
    PYMUPDF_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class ImageFormat(Enum):
    """Output image formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


@dataclass
class ExtractedImage:
    """
    An extracted image from a document.

    Attributes:
        image: PIL Image object
        page: Page number where image was found
        index: Image index on the page
        format: Original image format
        size: Image size (width, height)
        metadata: Additional metadata
    """
    image: Optional[Image.Image]
    page: int
    index: int
    format: str
    size: Tuple[int, int]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def save(self, path: Union[str, Path], format: ImageFormat = ImageFormat.PNG):
        """
        Save the image to a file.

        Args:
            path: Output path
            format: Image format
        """
        if self.image is None:
            raise ValueError("No image data to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.image.save(path, format=format.value.upper())

    def to_bytes(self, format: ImageFormat = ImageFormat.PNG) -> bytes:
        """
        Convert image to bytes.

        Args:
            format: Image format

        Returns:
            Image as bytes
        """
        if self.image is None:
            raise ValueError("No image data to convert")

        buffer = io.BytesIO()
        self.image.save(buffer, format=format.value.upper())
        return buffer.getvalue()

    def __repr__(self) -> str:
        return f"ExtractedImage(page={self.page}, index={self.index}, size={self.size}, format={self.format})"


class ImageExtractor:
    """
    Extract images from documents.

    Supports:
    - PDF documents (PyMuPDF)
    - DOCX documents (python-docx)
    - Image preprocessing and optimization
    """

    def __init__(
        self,
        min_size: Tuple[int, int] = (100, 100),
        max_size: Optional[Tuple[int, int]] = None,
        resize_mode: str = "fit",
    ):
        """
        Initialize image extractor.

        Args:
            min_size: Minimum image size (width, height)
            max_size: Maximum image size (width, height) - None for no limit
            resize_mode: Resize mode ('fit', 'crop', 'stretch')
        """
        self.min_size = min_size
        self.max_size = max_size
        self.resize_mode = resize_mode

    def extract_from_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from a PDF file.

        Args:
            pdf_path: Path to PDF file
            pages: List of pages to extract (None for all)

        Returns:
            List of ExtractedImage objects
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for PDF image extraction")

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        extracted = []

        try:
            doc = fitz.open(str(pdf_path))

            # Iterate through pages
            for page_num in doc:
                if pages is not None and (page_num + 1) not in pages:
                    continue

                page = doc[page_num]
                image_list = page.get_images(full=True)

                # Extract each image
                for img_index, img in enumerate(image_list):
                    xref = img[0]

                    # Extract image
                    base_image = doc.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_format = base_image.get("ext", "png")

                        # Load with PIL
                        img_data = io.BytesIO(image_bytes)
                        pil_image = Image.open(img_data)

                        # Apply size filtering
                        if self._should_include(pil_image.size):
                            # Resize if needed
                            if self.max_size:
                                pil_image = self._resize_image(pil_image)

                            extracted_image = ExtractedImage(
                                image=pil_image,
                                page=page_num + 1,
                                index=img_index,
                                format=image_format,
                                size=pil_image.size,
                                metadata={
                                    "xref": xref,
                                    "source": str(pdf_path),
                                }
                            )

                            extracted.append(extracted_image)

            doc.close()

        except Exception as e:
            raise RuntimeError(f"Failed to extract images from PDF: {e}")

        return extracted

    def extract_from_docx(
        self,
        docx_path: Union[str, Path],
    ) -> List[ExtractedImage]:
        """
        Extract images from a DOCX file.

        Args:
            docx_path: Path to DOCX file

        Returns:
            List of ExtractedImage objects
        """
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX image extraction")

        docx_path = Path(docx_path)

        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX not found: {docx_path}")

        extracted = []

        try:
            doc = Document(docx_path)

            # Extract images from relationships
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob

                        # Load with PIL
                        img_data_io = io.BytesIO(image_data)
                        pil_image = Image.open(img_data_io)

                        if self._should_include(pil_image.size):
                            if self.max_size:
                                pil_image = self._resize_image(pil_image)

                            extracted_image = ExtractedImage(
                                image=pil_image,
                                page=1,  # DOCX doesn't have pages
                                index=len(extracted),
                                format=pil_image.format or "unknown",
                                size=pil_image.size,
                                metadata={
                                    "source": str(docx_path),
                                }
                            )

                            extracted.append(extracted_image)

                    except Exception as e:
                        print(f"Warning: Failed to extract image: {e}")
                        continue

        except Exception as e:
            raise RuntimeError(f"Failed to extract images from DOCX: {e}")

        return extracted

    def extract_from_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from a directory of image files.

        Args:
            directory: Directory path
            extensions: List of file extensions to include

        Returns:
            List of ExtractedImage objects
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]

        directory = Path(directory)
        extracted = []

        for image_path in directory.rglob("*"):
            if image_path.suffix.lower() not in extensions:
                continue

            try:
                pil_image = Image.open(image_path)

                if self._should_include(pil_image.size):
                    if self.max_size:
                        pil_image = self._resize_image(pil_image)

                    extracted_image = ExtractedImage(
                        image=pil_image,
                        page=0,
                        index=len(extracted),
                        format=pil_image.format or "unknown",
                        size=pil_image.size,
                        metadata={
                            "source": str(image_path),
                        }
                    )

                    extracted.append(extracted_image)

            except Exception as e:
                print(f"Warning: Failed to load {image_path}: {e}")
                continue

        return extracted

    def _should_include(self, size: Tuple[int, int]) -> bool:
        """Check if image meets minimum size requirements."""
        width, height = size
        min_width, min_height = self.min_size
        return width >= min_width and height >= min_height

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image according to settings."""
        if not self.max_size:
            return image

        max_width, max_height = self.max_size
        width, height = image.size

        # Check if resize needed
        if width <= max_width and height <= max_height:
            return image

        if self.resize_mode == "fit":
            # Fit within bounds (maintain aspect ratio)
            ratio = min(max_width / width, max_height / height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.LANCZOS)

        elif self.resize_mode == "crop":
            # Crop to fit (maintain aspect ratio)
            ratio = max(max_width / width, max_height / height)
            new_size = (int(width * ratio), int(height * ratio))

            resized = image.resize(new_size, Image.LANCZOS)

            # Center crop
            left = (new_size[0] - max_width) // 2
            top = (new_size[1] - max_height) // 2
            right = left + max_width
            bottom = top + max_height

            return resized.crop((left, top, right, bottom))

        elif self.resize_mode == "stretch":
            # Stretch to fit (no aspect ratio)
            return image.resize(self.max_size, Image.LANCZOS)

        return image

    def save_images(
        self,
        images: List[ExtractedImage],
        output_dir: Union[str, Path],
        format: ImageFormat = ImageFormat.PNG,
        naming: str = "page_index",
    ) -> List[Path]:
        """
        Save extracted images to directory.

        Args:
            images: List of ExtractedImage objects
            output_dir: Output directory
            format: Image format
            naming: Naming pattern ('page_index', 'sequential', 'original')

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        for i, img in enumerate(images):
            if naming == "page_index":
                filename = f"page_{img.page}_img_{img.index}.{format.value}"
            elif naming == "sequential":
                filename = f"image_{i:04d}.{format.value}"
            else:
                filename = f"image_{i}.{format.value}"

            output_path = output_dir / filename
            img.save(output_path, format)
            saved.append(output_path)

        return saved
