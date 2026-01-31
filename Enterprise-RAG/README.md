# Enterprise-RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade Retrieval-Augmented Generation (RAG) system for enterprise document intelligence. Features hybrid retrieval (dense + sparse), cross-encoder reranking, multi-format document ingestion, and comprehensive evaluation.

## Overview

Enterprise-RAG is a complete, production-ready RAG system designed for building intelligent document Q&A systems. It combines state-of-the-art retrieval techniques with LLM generation to provide accurate, citation-backed answers from your enterprise documents.

### Key Features

- **Hybrid Retrieval**: Combines dense vector search (semantic) with sparse BM25 (keyword) using Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: MS-MARCO based reranking for improved result relevance
- **Multi-Format Ingestion**: Support for PDF, DOCX, Markdown, TXT, and HTML documents
- **Intelligent Chunking**: Context-aware document splitting with overlap and metadata preservation
- **LLM Agnostic**: Support for OpenAI, Anthropic, Ollama, and GLM providers
- **RAGAS Evaluation**: Built-in evaluation metrics (faithfulness, answer relevancy, context precision/recall)
- **Production API**: FastAPI REST API with async support and comprehensive error handling
- **Web Interface**: Professional Streamlit UI for interactive Q&A
- **Docker Deployment**: Multi-stage Docker builds for dev and production environments

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Enterprise-RAG                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   FastAPI    │      │  Streamlit   │      │   Docker     │  │
│  │     API      │      │      UI      │      │  Deployment  │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                      │          │
│         └─────────────────────┴──────────────────────┘          │
│                           │                                      │
│         ┌─────────────────▼──────────────────┐                  │
│         │         RAG Chain Orchestrator     │                  │
│         │    - Query routing & management    │                  │
│         │    - Citation extraction          │                  │
│         │    - LLM generation               │                  │
│         └─────────────────┬──────────────────┘                  │
│                           │                                      │
│         ┌─────────────────▼──────────────────┐                  │
│         │      Cross-Encoder Reranker        │                  │
│         │    - MS-MARCO model               │                  │
│         │    - Sigmoid normalization         │                  │
│         └─────────────────┬──────────────────┘                  │
│                           │                                      │
│         ┌─────────────────▼──────────────────┐                  │
│         │        Hybrid Retriever            │                  │
│         │    ┌─────────────┬────────────┐   │                  │
│         │    │   Dense     │   Sparse   │   │                  │
│         │    │  (Vectors)  │   (BM25)   │   │                  │
│         │    └──────┬──────┴─────┬──────┘   │                  │
│         │           │            │          │                  │
│         │    ┌──────▼──────┐ ┌───▼──────┐   │                  │
│         │    │ ChromaDB /  │ │ BM25    │   │                  │
│         │    │  Qdrant     │ │ Index   │   │                  │
│         │    └─────────────┘ └─────────┘   │                  │
│         └─────────────────┬──────────────────┘                  │
│                           │                                      │
│         ┌─────────────────▼──────────────────┐                  │
│         │      Document Processor            │                  │
│         │    - Multi-format parsing          │                  │
│         │    - Intelligent chunking          │                  │
│         │    - Metadata extraction           │                  │
│         └─────────────────┬──────────────────┘                  │
│                           │                                      │
│         ┌─────────────────▼──────────────────┐                  │
│         │      Embedding Service             │                  │
│         │    - Sentence Transformers          │                  │
│         │    - Batch processing               │                  │
│         │    - LRU caching                    │                  │
│         └─────────────────────────────────────┘                  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              RAGAS Evaluation Module                   │   │
│  │    - Faithfulness, Answer Relevancy                    │   │
│  │    - Context Precision, Context Recall                 │   │
│  │    - Automated report generation                       │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional)
- OpenAI API key or other LLM provider credentials

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Enterprise-RAG.git
cd Enterprise-RAG
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Download embedding and reranker models**:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

### Running the Application

**Option 1: Using Docker (Recommended)**

Development mode:
```bash
docker-compose -f docker-compose.dev.yml up --build
```

Production mode:
```bash
docker-compose up -d
```

**Option 2: Local Python**

Start the API server:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Start the Streamlit UI:
```bash
streamlit run src/ui/app.py --server.port 8501
```

### First Query

Once the services are running:

**Via API**:
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "top_k": 5,
    "use_reranking": true
  }'
```

**Via Web UI**:
Open your browser to `http://localhost:8501` and start asking questions!

## API Documentation

### Endpoints

#### Query Endpoints

**POST /api/v1/query**
Query the RAG system with a question.

```json
{
  "question": "What is the refund policy?",
  "top_k": 5,
  "use_reranking": true,
  "filters": {"file_type": "pdf"}
}
```

Response:
```json
{
  "answer": "According to the policy, refunds are processed within 5-7 business days...",
  "citations": [
    {
      "source": "policy.pdf",
      "chunk_id": "doc_1_chunk_0",
      "content_preview": "Refunds are processed within 5-7 business days...",
      "relevance_score": 0.92
    }
  ],
  "model_used": "gpt-4o-mini",
  "provider_used": "openai",
  "processing_time": 2.5
}
```

**POST /api/v1/query/stream**
Server-sent events streaming for real-time responses.

#### Document Management

**POST /api/v1/documents/ingest**
Upload and process a document.

```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest" \
  -F "file=@document.pdf"
```

**GET /api/v1/documents**
List all ingested documents with statistics.

**DELETE /api/v1/documents/{doc_id}**
Delete a document and all its chunks.

#### Evaluation

**POST /api/v1/evaluation/run**
Run RAGAS evaluation on the system.

```json
{
  "num_samples": 10
}
```

**GET /api/v1/evaluation/metrics**
Get available evaluation metrics and their status.

#### Conversation Management

**POST /api/v1/conversation/clear**
Clear conversation history.

**GET /api/v1/conversation**
Get current conversation history.

#### System Endpoints

**GET /**
API information and available endpoints.

**GET /health**
Health check status.

**GET /stats**
System statistics and metrics.

For full API documentation, visit `http://localhost:8000/docs` when the server is running.

## Evaluation Results

The system uses RAGAS framework for comprehensive evaluation:

### Metrics

- **Faithfulness**: Measures factual consistency of the answer with retrieved context
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Context Precision**: Measures signal-to-noise ratio in retrieved context
- **Context Recall**: Measures ability to retrieve all relevant information

### Running Evaluation

```bash
# Via API
curl -X POST "http://localhost:8000/api/v1/evaluation/run" \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 10}'

# Via Python
from src.evaluation.rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator(
    rag_chain=rag_chain,
    vector_store=vector_store
)

results = evaluator.evaluate_samples(num_samples=10)
print(results)
```

### Sample Evaluation Report

```
# RAG System Evaluation Report

## Overall Score: 0.76 / 1.00

### Metric Breakdown
- Faithfulness: 0.85
- Answer Relevancy: 0.78
- Context Precision: 0.72
- Context Recall: 0.69

### Top Performing Areas
- High faithfulness scores indicate answers are well-grounded in retrieved context
- Strong answer relevancy shows responses directly address questions

### Areas for Improvement
- Context recall could benefit from expanding document corpus
- Consider adjusting hybrid retrieval weights for better precision
```

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
GLM_API_KEY=...

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0.1

# Retrieval Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3

# Vector Store
VECTOR_STORE_TYPE=chroma  # chroma or qdrant
CHROMA_PERSIST_DIR=./data/chroma
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=enterprise_rag

# Evaluation
ENABLE_EVALUATION=true
RAGAS_EVALUATOR_OPENAI_API_KEY=sk-...

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:8501"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # json or text
```

### Configuration File

You can also use `config.yaml` for advanced configuration:

```yaml
retrieval:
  dense_weight: 0.7
  sparse_weight: 0.3
  rrf_k: 60
  top_k_retrieve: 10
  top_k_rerank: 5

generation:
  temperature: 0.1
  max_tokens: 1000
  system_prompt: |
    You are a helpful assistant. Answer questions based on the provided context.

processing:
  chunk_size: 512
  chunk_overlap: 50
  min_chunk_size: 100
  preserve_headers: true
```

## Development

### Project Structure

```
Enterprise-RAG/
├── src/
│   ├── api/                 # FastAPI application and routes
│   │   ├── main.py
│   │   ├── routes/
│   │   └── middleware.py
│   ├── core/                # Core business logic
│   ├── evaluation/          # RAGAS evaluation
│   ├── generation/          # RAG chain and LLM integration
│   ├── ingestion/           # Document processing
│   ├── models/              # Pydantic models
│   ├── retrieval/           # Retrieval components
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   ├── sparse_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── ui/                  # Streamlit interface
│   ├── config.py            # Configuration management
│   ├── exceptions.py        # Custom exceptions
│   └── logging_config.py    # Logging setup
├── tests/                   # Comprehensive test suite
│   ├── conftest.py
│   ├── test_document_processor.py
│   ├── test_retrieval.py
│   ├── test_rag_chain.py
│   └── test_api.py
├── data/                    # Persistent data storage
├── scripts/                 # Utility scripts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Run all quality checks
black src/ tests/ && ruff check src/ tests/ && mypy src/
```

### Adding New Features

1. **New Retrieval Method**: Extend `VectorStoreBase` in `src/retrieval/vector_store.py`

2. **New LLM Provider**: Add provider to `LLMProvider` enum in `src/generation/rag_chain.py`

3. **New Document Format**: Add extraction method to `DocumentProcessor` in `src/ingestion/document_processor.py`

4. **New Evaluation Metric**: Extend `RAGEvaluator` in `src/evaluation/rag_evaluator.py`

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Format code (`black src/ tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Performance Tips

1. **Enable GPU**: Set `DEVICE=cuda` for embedding and reranking models
2. **Use Production Vector Store**: Switch from ChromaDB to Qdrant for large-scale deployments
3. **Tune Weights**: Adjust `DENSE_WEIGHT` and `SPARSE_WEIGHT` based on your data
4. **Batch Processing**: Process multiple documents in parallel using the batch API
5. **Cache Embeddings**: Enable embedding cache for faster repeated queries

## Troubleshooting

### Common Issues

**Issue**: Low retrieval quality
- **Solution**: Tune hybrid retrieval weights or increase `TOP_K_RETRIEVAL`

**Issue**: Slow query response
- **Solution**: Disable reranking (`use_reranking=false`) or use a smaller model

**Issue**: Out of memory errors
- **Solution**: Reduce `CHUNK_SIZE`, `BATCH_SIZE`, or use CPU inference

**Issue**: Poor answer quality
- **Solution**: Improve document quality, adjust `TEMPERATURE`, or use better LLM

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [LlamaIndex](https://www.llamaindex.ai/) for RAG framework
- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the UI framework

## Contact

For questions, issues, or contributions, please visit:
- GitHub: https://github.com/yourusername/Enterprise-RAG
- Issues: https://github.com/yourusername/Enterprise-RAG/issues

---

Built with ❤️ for the AI Engineer community
