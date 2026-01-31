# FraudDocs-RAG

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-orange.svg)](https://www.llamaindex.ai/)

**Production-Grade RAG System for Financial Fraud Detection Documents**

A complete Retrieval-Augmented Generation system for querying financial regulations, AML/KYC procedures, and fraud detection documentation.

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [API](#api-endpoints) • [Deployment](#deployment)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

FraudDocs-RAG is a production-grade RAG (Retrieval-Augmented Generation) system specifically designed for financial fraud detection and compliance documents. It enables users to query complex regulatory documents using natural language and receive accurate, sourced answers.

**Perfect for:**
- Financial institutions needing quick access to compliance information
- Fraud detection teams researching AML/KYC procedures
- Regulatory compliance departments
- AI Engineer portfolio projects

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **🧠 Semantic Search** | Context-aware document retrieval using vector embeddings |
| **📚 Multi-Format Support** | PDF, DOCX, TXT, HTML document processing |
| **🎯 Smart Classification** | Auto-categorization into AML, KYC, Fraud, Regulation |
| **🔄 Cross-Encoder Reranking** | Improved retrieval accuracy with neural reranking |
| **🔗 Source Citations** | All answers include document source references |
| **🤖 Multi-LLM Support** | Ollama (dev), GLM-4 (demo), OpenAI (prod) |
| **⚡ Fast API** | Built with FastAPI for high performance |
| **🐳 Docker Ready** | Complete containerization with Docker Compose |
| **🎨 React Frontend** | Modern web interface with Tailwind CSS |
| **📊 Document Deduplication** | Content hashing to avoid duplicate processing |

### Document Categories

Automatically classifies documents into:
- 🛡️ **AML** - Anti-Money Laundering policies and SAR requirements
- 👤 **KYC** - Know Your Customer and CDD procedures
- 🔍 **Fraud** - Fraud detection and investigation protocols
- 📋 **Regulation** - Regulatory compliance requirements
- 📄 **General** - Other financial documents

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ─────────────────────────────────────────────────────────────  │
│  React + Tailwind CSS (Port 3000)                               │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                               │
│  ─────────────────────────────────────────────────────────────  │
│  FastAPI Application (Port 8000)                                │
│  • POST /query      • POST /ingest                              │
│  • GET  /health     • GET  /collections                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                               │
│  ─────────────────────────────────────────────────────────────  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Retrieval   │  │ Generation  │  │ Ingestion   │             │
│  │             │  │             │  │             │             │
│  │ • Vector    │  │ • LLM       │  │ • Loader    │             │
│  │   Search    │  │   Handler   │  │ • Chunker   │             │
│  │ • Reranker  │  │ • Prompting │  │ • Classifier│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Storage                               │
│  ─────────────────────────────────────────────────────────────  │
│  • ChromaDB (Vector Store)                                      │
│  • Document Files                                               │
│  • Model Cache                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Framework
- **RAG Framework**: [LlamaIndex](https://www.llamaindex.ai/) v0.10+
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) v0.5+
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) v0.115+
- **Frontend**: React + Vite + Tailwind CSS

### NLP & ML
- **Embeddings**: HuggingFace `BAAI/bge-small-en-v1.5`
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Providers**: Ollama, GLM-4, OpenAI

### Python
- **Version**: 3.11+
- **Package Manager**: pip / poetry
- **Testing**: pytest, pytest-asyncio
- **Code Quality**: black, ruff, mypy

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or poetry
- 4GB RAM minimum (8GB recommended)
- 10GB free disk space

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-docs-rag.git
cd fraud-docs-rag

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit .env with your settings
nano .env
```

### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-docs-rag.git
cd fraud-docs-rag

# Copy environment configuration
cp .env.example .env

# Start with Docker Compose
docker-compose up -d
```

---

## Quick Start

### 1. Set Up Environment

Edit `.env` file:

```bash
# Environment
ENVIRONMENT=development

# For development (Ollama - free, local)
# No API key needed

# For demo (GLM-4)
ZHIPUAI_API_KEY=your_glm_api_key_here

# For production (OpenAI)
# OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Ingest Documents

```bash
# Using CLI
python -m fraud_docs_rag.main ingest ./data/documents/

# Or using Python
from fraud_docs_rag.ingestion.document_processor import DocumentProcessor

processor = DocumentProcessor()
nodes = processor.process_directory("./data/documents/")
```

### 3. Start the API Server

```bash
# Using CLI
python -m fraud_docs_rag.main serve --reload

# Or using uvicorn
uvicorn fraud_docs_rag.api.main:app --reload
```

### 4. Query the System

```bash
# Using CLI
python -m fraud_docs_rag.main query "What are SAR filing requirements?" --filter aml

# Or using curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are SAR requirements?", "doc_type_filter": "aml"}'
```

### 5. Access the Web UI

```bash
# Frontend (in separate terminal)
cd frontend
npm install
npm run dev

# Access at http://localhost:3000
```

---

## Usage

### Command-Line Interface

```bash
# Query the knowledge base
frauddocs query "What are KYC requirements?" --filter kyc

# Ingest documents
frauddocs ingest ./documents/ --recursive

# Start the API server
frauddocs serve --host 0.0.0.0 --port 8000

# Check system health
frauddocs health

# Interactive mode
frauddocs interactive

# Display statistics
frauddocs stats

# Delete a collection
frauddocs delete-collection financial_documents --confirm
```

### Python API

```python
from fraud_docs_rag.generation.rag_chain import RAGChain
from fraud_docs_rag.retrieval.hybrid_retriever import HybridRetriever

# Initialize retriever
retriever = HybridRetriever()
retriever.load_index()

# Initialize RAG chain
rag_chain = RAGChain(retriever, environment="demo")

# Query
answer, sources = rag_chain.query(
    "What are the suspicious transaction reporting requirements?",
    doc_type_filter="aml"
)

print(answer)
print(f"\nSources: {len(sources)}")
```

### Web Interface

Navigate to `http://localhost:3000` and:
1. Enter your question in the chat input
2. Optionally filter by document type
3. View answers with source citations
4. Click on sources to expand previews

---

## API Endpoints

### Query Endpoint

```http
POST /query
Content-Type: application/json

{
  "question": "What are SAR filing deadlines?",
  "doc_type_filter": "aml",
  "use_rerank": true,
  "top_k": 10
}
```

**Response:**
```json
{
  "answer": "Based on the retrieved documents, SAR must be filed within 30 days...",
  "sources": [
    {
      "index": 1,
      "source": "aml_sar_requirements.pdf",
      "doc_type": "aml",
      "score": 0.924,
      "preview": "Suspicious Activity Reports (SAR) must be filed...",
      "title": "SAR Filing Requirements"
    }
  ],
  "query": "What are SAR filing deadlines?",
  "processing_time": 2.34,
  "environment": "demo"
}
```

### Ingest Endpoint

```http
POST /ingest
Content-Type: multipart/form-data

file: document.pdf
doc_type: aml
```

**Response:**
```json
{
  "status": "success",
  "documents_processed": 1,
  "chunks_created": 12,
  "categories": {"aml": 12},
  "processing_time": 8.5,
  "errors": []
}
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "demo",
  "retriever_status": "loaded",
  "chain_status": "ready",
  "uptime": 3600.5,
  "collection_stats": {
    "total_docs": 150
  }
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Environment type (development/demo/production) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `financial_documents` | Collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `TOP_K_RETRIEVAL` | `10` | Documents to retrieve |
| `RERANK_TOP_N` | `5` | Results after reranking |
| `ZHIPUAI_API_KEY` | - | GLM-4 API key (demo) |
| `OPENAI_API_KEY` | - | OpenAI API key (production) |

### LLM Provider Selection

```python
# Development - Ollama (free, local)
ENVIRONMENT=development

# Demo - GLM-4 (ZhipuAI)
ENVIRONMENT=demo
ZHIPUAI_API_KEY=your_key

# Production - OpenAI
ENVIRONMENT=production
OPENAI_API_KEY=your_key
```

---

## Development

### Project Structure

```
fraud-docs-rag/
├── src/fraud_docs_rag/
│   ├── api/                 # FastAPI application
│   │   └── main.py          # API endpoints
│   ├── generation/          # LLM generation
│   │   └── rag_chain.py     # RAG chain
│   ├── ingestion/           # Document processing
│   │   └── document_processor.py
│   ├── retrieval/           # Vector search
│   │   └── hybrid_retriever.py
│   ├── utils/               # Utilities
│   ├── config.py            # Configuration
│   └── main.py              # CLI
├── tests/                   # Test suite
├── frontend/                # React UI
├── data/                    # Data storage
├── Dockerfile               # Docker image
├── docker-compose.yml       # Full stack
└── requirements.txt         # Dependencies
```

### Running Tests

```bash
# Run all tests
pytest

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest --cov=src/fraud_docs_rag --cov-report=html

# Run specific test
pytest tests/integration/test_integration.py::test_e2e_query -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

---

## Deployment

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

### Production Deployment

1. **Set environment variables:**
```bash
export ENVIRONMENT=production
export OPENAI_API_KEY=your_key
export ZHIPUAI_API_KEY=your_key
```

2. **Build production image:**
```bash
docker build --target runtime -t frauddocs-rag:prod .
```

3. **Deploy with Docker Compose:**
```bash
docker-compose --profile production up -d
```

### Cloud Deployment

#### AWS EC2
```bash
# Launch EC2 instance (Ubuntu 22.04)
# Install Docker
# Clone repository
# Run docker-compose up -d
```

#### Google Cloud Run
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT-ID/frauddocs-rag

# Deploy
gcloud run deploy frauddocs-rag --image gcr.io/PROJECT-ID/frauddocs-rag
```

#### Azure Container Instances
```bash
# Create resource group
az group create --name frauddocs-rg --location eastus

# Deploy container
az container create \
  --resource-group frauddocs-rg \
  --name frauddocs-rag \
  --image frauddocs-rag:latest \
  --ports 8000
```

---

## Testing

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows
- **API Tests**: Endpoint testing
- **Performance Tests**: Latency and load testing

### Run Tests

```bash
# All tests
pytest -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest --cov=src/fraud_docs_rag --cov-report=html

# Marked tests (requires LLM)
pytest -m llm -v
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Run tests before submitting

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LlamaIndex** - Excellent RAG framework
- **ChromaDB** - Vector database
- **FastAPI** - Modern Python web framework
- **Sentence Transformers** - Embedding models

---

## Support

For issues, questions, or contributions:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/fraud-docs-rag/issues)
- 📖 Docs: [Full Documentation](https://frauddocs-rag.readthedocs.io/)

---

## Roadmap

- [ ] Hybrid search (vector + keyword)
- [ ] Query expansion and refinement
- [ ] Multi-language support
- [ ] Document versioning
- [ ] User feedback integration
- [ ] Advanced analytics dashboard
- [ ] Batch document processing API

---

<div align="center">

**Built with ❤️ for the AI Engineer community**

[⬆ Back to Top](#frauddocs-rag)

</div>
