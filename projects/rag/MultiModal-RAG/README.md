# MultiModal-RAG: Multi-Modal RAG System

Advanced RAG system with support for text, images, and tables.

## Features

- **Multi-Modal Document Processing**
  - Text extraction from PDFs, DOCX, HTML, Markdown
  - OCR with multiple engines (Tesseract, PaddleOCR, EasyOCR)
  - Image captioning with BLIP
  - Table extraction and parsing (Camelot, Tabula)

- **Multi-Modal Embeddings**
  - CLIP embeddings for image-text similarity
  - Text embeddings (sentence-transformers)
  - Cross-modal retrieval

- **Vision-Enhanced RAG**
  - GPT-4V integration
  - Claude Vision support
  - Image + text combined queries

- **Production-Ready**
  - FastAPI REST API
  - Streamlit web interface
  - Docker deployment
  - Comprehensive testing

## Installation

### Prerequisites

```bash
# System dependencies for OCR
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0  # For OpenCV
```

### Python Setup

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: CUDA Support

For GPU acceleration (CLIP, BLIP, embeddings):

```bash
# PyTorch with CUDA (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Configuration

Create `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_key

# Vector Store
VECTOR_DB_TYPE=chroma  # or qdrant
CHROMA_PERSIST_DIR=./data/vectordb

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vision
ENABLE_VISION=true
CLIP_MODEL=ViT-B/32
CAPTIONING_MODEL=Salesforce/blip-image-captioning-base
```

### 2. Process Documents

```python
from src.multimodal import ImageProcessor, TableExtractor
from src.ingestion.document_processor import DocumentProcessor

# Process images
processor = ImageProcessor()
analysis = processor.analyze("data/sample_images/chart.png")
print(f"Caption: {analysis.caption}")
print(f"OCR Text: {analysis.ocr_text}")

# Extract tables
extractor = TableExtractor()
tables = extractor.extract_from_pdf("data/documents/report.pdf")
for table in tables:
    print(table.to_markdown())
```

### 3. Build Multi-Modal Index

```python
from src.multimodal import MultiModalRetriever

retriever = MultiModalRetriever()

# Add text nodes
retriever.add_text_node("doc1", "This is a sample document.")

# Add image nodes
retriever.add_image_node("img1", "path/to/image.jpg", "A chart showing trends")

# Add table nodes
retriever.add_table_node("tbl1", "Sales data table", table_json)

# Search
results = retriever.retrieve("sales trends", top_k=5)
for result in results:
    print(f"{result.score:.2f}: {result.node.content[:100]}")
```

### 4. Vision-Enhanced RAG

```python
from src.multimodal import VisionLLM

llm = VisionLLM(provider=VisionProvider.OPENAI)

# Analyze an image
response = llm.analyze_image("chart.png", "Describe the trends in this chart.")
print(response.content)

# Answer with image context
response = llm.answer_with_context(
    question="What does the chart show?",
    context="The chart displays quarterly revenue.",
    images=["chart.png"]
)
```

### 5. Run API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run Web UI

```bash
streamlit run src/ui/app.py
```

## Project Structure

```
MultiModal-RAG/
├── src/
│   ├── multimodal/           # Multi-modal processing
│   │   ├── image_processor.py    # Image OCR, captioning, CLIP
│   │   ├── table_extractor.py    # Table extraction
│   │   ├── multimodal_retriever.py  # Multi-modal search
│   │   └── vision_llm.py          # Vision LLM integration
│   ├── extraction/           # Extraction utilities
│   │   ├── ocr_processor.py
│   │   ├── image_extractor.py
│   │   └── table_extractor.py
│   ├── ingestion/           # Document processing
│   ├── retrieval/           # Embeddings & vector store
│   ├── generation/          # RAG chain
│   ├── api/                # FastAPI endpoints
│   ├── ui/                 # Streamlit interface
│   └── config.py           # Configuration
├── tests/
├── data/
│   ├── documents/
│   ├── sample_images/
│   └── sample_tables/
└── requirements.txt
```

## API Endpoints

### Multi-Modal Endpoints

```bash
# Ingest document with images
POST /api/v1/documents/ingest
Content-Type: multipart/form-data

# Query with image
POST /api/v1/query/multimodal
{
  "query": "What does this chart show?",
  "image": "base64_encoded_image"
}

# Analyze image
POST /api/v1/vision/analyze
{
  "image": "base64_encoded_image",
  "prompt": "Describe this image"
}

# Extract tables
POST /api/v1/tables/extract
{
  "pdf": "file"
}
```

## Docker Deployment

```bash
# Build image
docker build -t multimodal-rag .

# Run with GPU support
docker-compose -f docker-compose.yml up -d
```

## Testing

```bash
# Run all tests
pytest

# Run multi-modal tests
pytest tests/unit/multimodal/

# Run with coverage
pytest --cov=src/multimodal
```

## License

MIT
