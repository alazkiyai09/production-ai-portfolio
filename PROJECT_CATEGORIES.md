# AI Engineer Portfolio Projects

Organized portfolio of AI Engineer projects demonstrating RAG systems, AI agents, evaluation frameworks, and infrastructure.

## 📁 Project Organization

### 🔄 projects/rag/
Retrieval-Augmented Generation systems with various architectures and use cases.

| Project | Description | Key Features |
|---------|-------------|--------------|
| **Enterprise-RAG** | Production-grade hybrid RAG system | Dense + sparse retrieval, cross-encoder reranking, RAGAS evaluation, multi-format ingestion |
| **MultiModal-RAG** | Multi-modal RAG for images and text | Vision + text embeddings, image retrieval |
| **DataChat-RAG** | Conversational data analysis | SQL generation, data visualization |
| **fraud-docs-rag** | Fraud detection document RAG | Specialized for financial fraud analysis |

### 🤖 projects/agents/
LangGraph-based AI agents with memory, tools, and real-time capabilities.

| Project | Description | Key Features |
|---------|-------------|--------------|
| **CustomerSupport-Agent** | Customer service chatbot | LangGraph, long-term memory, ChromaDB FAQ, WebSocket API, sentiment analysis |
| **FraudTriage-Agent** | Fraud triage and analysis | Document processing, risk scoring, escalation logic |
| **AdInsights-Agent** | Advertising analytics agent | Campaign analysis, insights generation |

### 📊 projects/evaluation/
LLM evaluation and testing frameworks for model comparison and optimization.

| Project | Description | Key Features |
|---------|-------------|--------------|
| **LLMOps-Eval** | Comprehensive LLM evaluation | 9 evaluation metrics, multi-model comparison, prompt optimization, cost tracking |

### 🏗️ projects/infrastructure/
Supporting infrastructure and utilities.

| Project | Description | Key Features |
|---------|-------------|--------------|
| **StreamProcess-Pipeline** | Real-time streaming pipeline | Async processing, stream analytics |
| **aiguard** | AI safety and monitoring | Content moderation, safety checks |

## 🛠️ Shared Utilities

### shared/
Common utilities used across all projects:
- **security.py** - API key redaction, log sanitization, sensitive data filtering
- **rate_limit.py** - Rate limiting with slowapi, client identification strategies

### reviews/
Code review outputs and documentation:
- Enterprise-RAG-review.md
- LLMOps-Eval-review.md
- CustomerSupport-Agent-review.md
- CODE_REVIEW_SUMMARY.md
- CRITICAL_ISSUES_FIX_STATUS.md

## 📋 Root Documentation

| File | Description |
|------|-------------|
| **README.md** | Main project overview |
| **PROJECT_STRUCTURE.md** | Detailed architecture documentation |
| **TECHNICAL.md** | Technical specifications |
| **CODE_REVIEW_SUMMARY.md** | Portfolio-wide code review results |
| **CRITICAL_ISSUES_FIX_STATUS.md** | Security and reliability fixes status |

## 🔧 Quick Start

### Running a RAG Project
```bash
cd projects/rag/Enterprise-RAG
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### Running an Agent Project
```bash
cd projects/agents/CustomerSupport-Agent
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### Running Evaluation Framework
```bash
cd projects/evaluation/LLMOps-Eval
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## 🎯 Portfolio Highlights

### Best Production-Ready: Enterprise-RAG
- Hybrid retrieval (dense + sparse)
- Cross-encoder reranking
- Comprehensive testing
- Security hardening

### Best Agent Architecture: CustomerSupport-Agent
- LangGraph state machine
- Multi-level memory
- WebSocket real-time chat
- Sentiment-driven escalation

### Best Evaluation Framework: LLMOps-Eval
- 9 evaluation metrics
- Multi-provider support
- Cost optimization
- Prompt A/B testing

## 🔐 Security Features Implemented

All projects now include:
- ✅ API key redaction in logs
- ✅ Rate limiting infrastructure
- ✅ Input validation
- ✅ Error handling best practices

## 📦 Dependencies by Category

### RAG Projects
- llama-index, chromadb, qdrant-client
- sentence-transformers, rank-bm25
- ragas, langchain

### Agent Projects
- langgraph, langchain
- chromadb, textblob
- websockets, sqlalchemy

### Evaluation
- openai, anthropic, cohere
- prometheus-client
- pandas, plotly

### Infrastructure
- fastapi, uvicorn
- slowapi
- python-dotenv

## 🚀 Deployment

All projects use:
- FastAPI for REST APIs
- Streamlit for UIs
- Docker for containerization
- pytest for testing

## 📊 Project Metrics

| Category | Projects | Total LOC | Test Coverage |
|----------|----------|-----------|---------------|
| RAG | 4 | ~15,000 | 75% |
| Agents | 3 | ~8,000 | 68% |
| Evaluation | 1 | ~6,500 | 72% |
| Infrastructure | 2 | ~4,000 | 60% |

## 🔄 CI/CD Integration

All projects support:
- GitHub Actions workflows
- Docker multi-stage builds
- Automated testing
- Code quality checks (black, ruff, mypy)

## 📝 License

Each project may have its own license. See individual project directories for details.

---

**Last Updated:** 2026-01-31
**Projects:** 10 total across 4 categories
**Total Lines of Code:** ~33,500
