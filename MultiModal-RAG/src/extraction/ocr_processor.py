"""
OCR Processor - Text extraction from images using multiple OCR engines.

Supports:
- Tesseract OCR
- PaddleOCR
- EasyOCR
- Combined OCR results with confidence scoring
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Optional dependencies
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCREngine(Enum):
    """Supported OCR engines."""
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"
    AUTO = "auto"


@dataclass
class OCRResult:
    """
    Result from OCR processing.

    Attributes:
        text: Extracted text
        confidence: Average confidence score
        engine: OCR engine used
        regions: List of detected text regions
        language: Detected language(s)
    """
    text: str
    confidence: float
    engine: OCREngine
    regions: List[Dict[str, Any]] = None
    language: str = "en"

    def __post_init__(self):
        if self.regions is None:
            self.regions = []


class OCRProcessor:
    """
    Multi-engine OCR processor.

    Features:
    - Multiple OCR backends (Tesseract, PaddleOCR, EasyOCR)
    - Automatic engine selection
    - Result confidence scoring
    - Multi-language support
    - Region extraction with bounding boxes
    """

    def __init__(
        self,
        engine: OCREngine = OCREngine.AUTO,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        combine_results: bool = False,
    ):
        """
        Initialize OCR processor.

        Args:
            engine: OCR engine to use
            languages: List of language codes (e.g., ['en', 'zh'])
            confidence_threshold: Minimum confidence for text extraction
            combine_results: Whether to combine results from multiple engines
        """
        self.engine = engine
        self.languages = languages or ['en']
        self.confidence_threshold = confidence_threshold
        self.combine_results = combine_results

        # Initialize engines
        self.tesseract_config = None
        self.paddleocr_engine = None
        self.easyocr_reader = None

        self._init_engines()

    def _init_engines(self):
        """Initialize OCR engines."""
        # Tesseract
        if TESSERACT_AVAILABLE:
            lang_str = "+".join(self.languages)
            self.tesseract_config = f'--oem 3 --psm 6 -l {lang_str}'

        # PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddleocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en' if 'en' in self.languages else 'ch',
                    show_log=False
                )
            except Exception as e:
                print(f"Warning: Failed to initialize PaddleOCR: {e}")

        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(
                    self.languages,
                    gpu=False,
                    verbose=False
                )
            except Exception as e:
                print(f"Warning: Failed to initialize EasyOCR: {e}")

    def extract_text(
        self,
        image: Union[str, Path, np.ndarray],
        engine: Optional[OCREngine] = None,
    ) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: Image path or numpy array
            engine: OCR engine to use (None for default)

        Returns:
            OCRResult object
        """
        engine = engine or self.engine

        if engine == OCREngine.AUTO:
            # Try each engine until one succeeds
            for eng in [OCREngine.TESSERACT, OCREngine.PADDLEOCR, OCREngine.EASYOCR]:
                try:
                    result = self._extract_with_engine(image, eng)
                    if result.text.strip():
                        return result
                except Exception:
                    continue

            # If all fail, return empty result
            return OCRResult(
                text="",
                confidence=0.0,
                engine=OCREngine.AUTO
            )

        elif self.combine_results:
            # Combine results from all available engines
            return self._combine_engines(image)

        else:
            return self._extract_with_engine(image, engine)

    def _extract_with_engine(
        self,
        image: Union[str, Path, np.ndarray],
        engine: OCREngine,
    ) -> OCRResult:
        """Extract text with a specific engine."""
        if engine == OCREngine.TESSERACT:
            return self._extract_tesseract(image)
        elif engine == OCREngine.PADDLEOCR:
            return self._extract_paddleocr(image)
        elif engine == OCREngine.EASYOCR:
            return self._extract_easyocr(image)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def _extract_tesseract(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> OCRResult:
        """Extract text using Tesseract."""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not available")

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            img_array = np.array(img)
        else:
            img_array = image

        # Get detailed data
        data = pytesseract.image_to_data(
            img_array,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )

        # Extract text with confidence
        text_parts = []
        confidences = []
        regions = []

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i]
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

            if conf > self.confidence_threshold * 100 and text.strip():
                text_parts.append(text)
                confidences.append(conf / 100.0)

                regions.append({
                    'text': text,
                    'confidence': conf / 100.0,
                    'bbox': (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i],
                    )
                })

        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            engine=OCREngine.TESSERACT,
            regions=regions,
        )

    def _extract_paddleocr(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> OCRResult:
        """Extract text using PaddleOCR."""
        if not PADDLEOCR_AVAILABLE or not self.paddleocr_engine:
            raise RuntimeError("PaddleOCR not available")

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = np.array(Image.open(image))
        else:
            img_array = image

        # Run OCR
        result = self.paddleocr_engine.ocr(img_array, cls=True)

        if not result or not result[0]:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=OCREngine.PADDLEOCR,
            )

        # Extract text and confidence
        text_parts = []
        confidences = []
        regions = []

        for line in result[0]:
            bbox = line[0]
            text_info = line[1]
            text = text_info[0]
            conf = text_info[1]

            text_parts.append(text)
            confidences.append(conf)

            regions.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox
            })

        full_text = '\n'.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            engine=OCREngine.PADDLEOCR,
            regions=regions,
        )

    def _extract_easyocr(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> OCRResult:
        """Extract text using EasyOCR."""
        if not EASYOCR_AVAILABLE or not self.easyocr_reader:
            raise RuntimeError("EasyOCR not available")

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = np.array(Image.open(image))
        else:
            img_array = image

        # Run OCR
        results = self.easyocr_reader.readtext(img_array)

        if not results:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=OCREngine.EASYOCR,
            )

        # Extract text and confidence
        text_parts = []
        confidences = []
        regions = []

        for (bbox, text, conf) in results:
            if conf > self.confidence_threshold:
                text_parts.append(text)
                confidences.append(conf)

                regions.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox
                })

        full_text = '\n'.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            engine=OCREngine.EASYOCR,
            regions=regions,
        )

    def _combine_engines(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> OCRResult:
        """Combine results from multiple engines."""
        results = []

        # Try each engine
        for engine in [OCREngine.TESSERACT, OCREngine.PADDLEOCR, OCREngine.EASYOCR]:
            try:
                result = self._extract_with_engine(image, engine)
                if result.text.strip():
                    results.append(result)
            except Exception:
                pass

        if not results:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=OCREngine.AUTO,
            )

        # Combine by weighted average (confidence as weight)
        total_weight = sum(r.confidence for r in results)

        if total_weight == 0:
            # Equal weights if no confidence
            combined_text = '\n\n'.join(r.text for r in results)
            avg_confidence = np.mean([r.confidence for r in results])
        else:
            # Weighted combination
            texts = []
            for r in results:
                weight = r.confidence / total_weight
                texts.append(f"{r.text} (confidence: {r.confidence:.2f})")

            combined_text = '\n\n'.join(texts)
            avg_confidence = total_weight / len(results)

        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            engine=OCREngine.AUTO,
        )

    def batch_extract(
        self,
        images: List[Union[str, Path, np.ndarray]],
        **kwargs
    ) -> List[OCRResult]:
        """
        Extract text from multiple images.

        Args:
            images: List of images
            **kwargs: Additional arguments

        Returns:
            List of OCRResult objects
        """
        results = []

        for image in images:
            try:
                result = self.extract_text(image, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Failed to process {image}: {e}")
                results.append(
                    OCRResult(
                        text="",
                        confidence=0.0,
                        engine=self.engine,
                    )
                )

        return results
