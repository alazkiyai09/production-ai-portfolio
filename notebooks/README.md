# AI Engineer Portfolio - Jupyter Notebooks

Interactive Jupyter notebooks demonstrating all AI Engineer portfolio projects.

## 📁 Notebook Organization

```
notebooks/
├── README.md (this file)
├── rag/                          # RAG System Demos
│   ├── Enterprise-RAG-Demo.ipynb         # Complete hybrid RAG walkthrough
│   └── Other-RAG-Projects-Demo.ipynb     # MultiModal, DataChat, fraud-docs
├── agents/                       # AI Agent Demos
│   ├── CustomerSupport-Agent-Demo.ipynb  # LangGraph support chatbot
│   └── Other-Agents-Demo.ipynb           # FraudTriage, AdInsights
├── evaluation/                   # Evaluation Framework Demo
│   └── LLMOps-Eval-Demo.ipynb             # LLM evaluation & comparison
└── infrastructure/               # Infrastructure Demos
    └── Infrastructure-Demo.ipynb          # Pipeline & monitoring
```

## 🚀 Quick Start

### Option 1: Run Individual Notebooks

```bash
# Navigate to notebook directory
cd notebooks/rag

# Start Jupyter
jupyter notebook

# Open Enterprise-RAG-Demo.ipynb
```

### Option 2: Run with Jupyter Lab

```bash
# From root directory
jupyter lab notebooks/

# Access at http://localhost:8888
```

### Option 3: Run with Google Colab

1. Upload individual notebook to Google Drive
2. Open in Google Colab
3. Run cells sequentially

## 📚 Available Notebooks

### 🔄 RAG Systems (4 notebooks)

#### 1. Enterprise-RAG-Demo.ipynb
**Production-Grade Hybrid RAG System**

**Demonstrates:**
- Hybrid retrieval (dense vector + sparse BM25)
- Cross-encoder reranking
- Multi-format document ingestion
- RAGAS evaluation metrics
- Performance comparison
- Interactive chat

**Runtime:** ~15 minutes

**Requirements:**
- Python 3.11+
- llama-index, chromadb, sentence-transformers
- OpenAI API key (or local model)

---

#### 2. Other-RAG-Projects-Demo.ipynb
**MultiModal, DataChat, and Fraud Detection RAG**

**Demonstrates:**
- MultiModal-RAG: Image + text retrieval
- DataChat-RAG: Natural language SQL
- fraud-docs-rag: Financial fraud analysis

**Runtime:** ~5 minutes (overview only)

---

### 🤖 AI Agents (2 notebooks)

#### 3. CustomerSupport-Agent-Demo.ipynb
**LangGraph-Based Customer Service Chatbot**

**Demonstrates:**
- LangGraph conversation flow
- FAQ knowledge base search
- Sentiment analysis
- Frustration detection
- Automatic escalation
- Ticket creation
- Memory management

**Runtime:** ~15 minutes

**Requirements:**
- langgraph, langchain, chromadb
- OpenAI API key

---

#### 4. Other-Agents-Demo.ipynb
**FraudTriage and AdInsights Agents**

**Demonstrates:**
- FraudTriage-Agent: Risk scoring and fraud detection
- AdInsights-Agent: Marketing analytics and insights

**Runtime:** ~5 minutes (overview only)

---

### 📊 Evaluation (1 notebook)

#### 5. LLMOps-Eval-Demo.ipynb
**Comprehensive LLM Evaluation Framework**

**Demonstrates:**
- Multi-model comparison (OpenAI, Anthropic, Cohere)
- 9 evaluation metrics
- Performance tracking
- Results visualization
- Prompt A/B testing
- Cost analysis

**Runtime:** ~20 minutes

**Requirements:**
- openai, anthropic, cohere-ai
- pandas, plotly
- API keys for providers

---

### 🏗️ Infrastructure (1 notebook)

#### 6. Infrastructure-Demo.ipynb
**High-Throughput Pipeline & AI Safety**

**Demonstrates:**
- StreamProcess-Pipeline: 10K+ events/sec processing
- aiguard: Content moderation and safety
- Performance simulation
- Architecture overview

**Runtime:** ~10 minutes

---

## 📋 Notebook Features

### ✅ Common Features Across All Notebooks:

1. **Step-by-step tutorials** - Follow along with clear explanations
2. **Code examples** - Ready-to-run code snippets
3. **Visualizations** - Plotly charts and graphs
4. **Performance metrics** - Timing and accuracy measurements
5. **Comparisons** - Side-by-side feature comparisons
6. **Best practices** - Production-ready patterns

### 🎯 Learning Path:

**Beginner → Advanced:**
1. Start with **Other-RAG-Projects-Demo.ipynb** (overview)
2. Then **Enterprise-RAG-Demo.ipynb** (complete walkthrough)
3. Next **CustomerSupport-Agent-Demo.ipynb** (agents)
4. Then **LLMOps-Eval-Demo.ipynb** (evaluation)
5. Finally **Infrastructure-Demo.ipynb** (deployment)

## 🔧 Setup Instructions

### 1. Install Jupyter

```bash
pip install jupyter jupyterlab
```

### 2. Install Project Dependencies

Each notebook will install its own dependencies, but you can pre-install:

```bash
# Core dependencies
pip install llama-index langchain langgraph openai anthropic

# Vector databases
pip install chromadb qdrant-client

# ML/AI
pip install sentence-transformers transformers torch

# Utilities
pip install pandas plotly fastapi celery redis
```

### 3. Set API Keys

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export COHERE_API_KEY="your-key-here"
```

## 💡 Usage Tips

### Running Cells:
- Run cells in order (Shift+Enter)
- Wait for each cell to complete before proceeding
- Some cells require API keys - you'll be prompted

### Saving Work:
- Notebooks are in read-only mode in repo
- File → Save As to save your own copy
- Download as Python if needed

### Performance:
- Some notebooks make API calls (may take time)
- Reduce dataset sizes for faster experimentation
- Use local models when available to save API costs

## 📊 What You'll Learn

### By Category:

**🔄 RAG Systems:**
- Hybrid retrieval architectures
- Vector embeddings and sparse search
- Cross-encoder reranking
- Evaluation metrics (RAGAS)

**🤖 AI Agents:**
- LangGraph state machines
- Tool calling and function routing
- Memory management
- Sentiment analysis

**📊 Evaluation:**
- Multi-model comparison
- Prompt engineering
- Cost optimization
- Performance benchmarking

**🏗️ Infrastructure:**
- High-throughput pipelines
- Distributed processing
- Monitoring and observability
- Kubernetes deployment

## 🐛 Troubleshooting

### Import Errors:
```python
# If you get import errors, add project to path:
import sys
sys.path.insert(0, '../../projects/rag/Enterprise-RAG')
```

### API Key Errors:
```python
# Set keys in notebook:
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'
```

### Memory Issues:
- Restart kernel: Kernel → Restart
- Clear outputs: Cell → All Output → Clear
- Use smaller datasets

## 📚 Additional Resources

### Documentation:
- [Project Categories Guide](../PROJECT_CATEGORIES.md)
- [Code Review Summary](../CODE_REVIEW_SUMMARY.md)
- [Critical Fixes Status](../CRITICAL_ISSUES_FIX_STATUS.md)

### Project READMEs:
- [Enterprise-RAG](../projects/rag/Enterprise-RAG/)
- [CustomerSupport-Agent](../projects/agents/CustomerSupport-Agent/)
- [LLMOps-Eval](../projects/evaluation/LLMOps-Eval/)

## 🤝 Contributing

Want to add more notebooks?

1. Create notebook in appropriate category folder
2. Follow existing naming convention
3. Include requirements cell
4. Add clear explanations
5. Test thoroughly

## 📝 License

Same as parent repository.

---

**Last Updated:** 2026-01-31
**Total Notebooks:** 6
**Categories:** 4 (RAG, Agents, Evaluation, Infrastructure)
