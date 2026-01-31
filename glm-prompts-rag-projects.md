# GLM-4.7 Implementation Guide: RAG Projects
## 2 Production-Ready RAG Systems

---

# PROJECT 1A: Enterprise-RAG
## Production-Grade RAG with Hybrid Retrieval & Evaluation

### Project Overview

**What You'll Build**: A complete RAG system with:
- Hybrid retrieval (dense + sparse)
- Cross-encoder reranking
- Multi-format document ingestion
- RAGAS evaluation integration
- Production API + Demo UI

**Why This Matters for Jobs**:
- EY: "Implement vector database solutions, integrate LangChain"
- Harnham: "Develop RAG applications that utilize enterprise data"
- Turing: "Build RAG pipelines grounded in data"

**Time Estimate**: 10-14 days

---

## SESSION SETUP PROMPT

Copy and paste this to start your GLM-4.7 session:

```
You are an expert Python developer helping me build a production-grade RAG system.

PROJECT: Enterprise-RAG
PURPOSE: A complete RAG implementation for my AI Engineer portfolio demonstrating:
- Hybrid retrieval (dense vectors + sparse BM25)
- Cross-encoder reranking for accuracy improvement
- Multi-document ingestion (PDF, DOCX, MD, TXT)
- RAGAS evaluation framework integration
- Production-ready FastAPI backend
- Streamlit demo interface

TECH STACK:
- LlamaIndex 0.10+ for RAG orchestration
- ChromaDB for development, Qdrant for production
- sentence-transformers for embeddings (all-MiniLM-L6-v2)
- cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking
- FastAPI + uvicorn for API
- Streamlit for demo UI
- RAGAS for evaluation metrics
- Docker for deployment
- Python 3.11+

QUALITY REQUIREMENTS:
- Type hints on all functions
- Docstrings with examples
- Error handling with custom exceptions
- Logging throughout
- Unit tests for each module
- Production-ready code (not prototypes)

USER CONTEXT:
- I'm transitioning from fraud detection to AI Engineering
- Targeting remote AI Engineer roles (Turing, Toptal, startups)
- This portfolio must demonstrate production-readiness

RULES:
1. Generate complete, runnable code (no placeholders or "...")
2. Include all imports at the top of each file
3. Add comments explaining key decisions
4. Follow Python best practices (PEP 8, etc.)

Please confirm you understand these requirements, then we'll build this system file by file.
```

---

## PROMPT 1.1: Project Structure & Configuration

```
Create the complete project structure for Enterprise-RAG.

Generate these files:

1. Directory structure (show as tree)
2. requirements.txt with pinned versions
3. pyproject.toml with project metadata
4. .env.example with all environment variables
5. src/__init__.py
6. src/config.py with Pydantic settings

For requirements.txt, include:
- llama-index>=0.10.0
- llama-index-vector-stores-chroma>=0.1.0
- llama-index-embeddings-huggingface>=0.1.0
- chromadb>=0.4.0
- sentence-transformers>=2.2.0
- fastapi>=0.109.0
- uvicorn[standard]>=0.27.0
- streamlit>=1.31.0
- ragas>=0.1.0
- langchain>=0.1.0 (for RAGAS compatibility)
- pypdf>=4.0.0
- python-docx>=1.1.0
- python-multipart>=0.0.6
- python-dotenv>=1.0.0
- pydantic>=2.5.0
- pydantic-settings>=2.1.0
- httpx>=0.26.0
- pytest>=7.4.0
- pytest-asyncio>=0.23.0
- rank-bm25>=0.2.2

For config.py, use Pydantic BaseSettings with:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY (optional)
- EMBEDDING_MODEL
- RERANKER_MODEL
- CHUNK_SIZE
- CHUNK_OVERLAP
- TOP_K_RETRIEVAL
- TOP_K_RERANK
- CHROMA_PATH
- LOG_LEVEL

Output all files completely with no placeholders.
```

---

## PROMPT 1.2: Custom Exceptions & Logging

```
Create the error handling and logging setup for Enterprise-RAG.

File 1: src/exceptions.py

Define custom exceptions:
- RAGException (base)
- DocumentProcessingError
- RetrievalError
- GenerationError
- EvaluationError
- ConfigurationError

Each should include:
- Custom message
- Optional original exception
- Error code for API responses

File 2: src/logging_config.py

Set up structured logging with:
- Console handler with colors
- File handler (logs/app.log)
- JSON format for production
- Different levels per module
- Request ID tracking support

Include helper function:
- get_logger(name: str) -> logging.Logger

Output both files completely.
```

---

## PROMPT 1.3: Document Processing Pipeline

```
Create the document ingestion and processing pipeline.

File: src/ingestion/document_processor.py

Requirements:
1. Support file types: PDF, DOCX, MD, TXT, HTML
2. Intelligent chunking with configurable size and overlap
3. Metadata extraction (title, date, source, file type, page numbers)
4. Text cleaning and normalization
5. Chunk ID generation for tracking

Classes and functions:

@dataclass
class Document:
    content: str
    metadata: dict
    doc_id: str
    chunk_id: str
    
@dataclass  
class ProcessingResult:
    documents: List[Document]
    total_chunks: int
    processing_time: float
    errors: List[str]

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        ...
    
    def process_file(self, file_path: Path) -> ProcessingResult:
        """Process a single file into chunks."""
        ...
    
    def process_directory(self, dir_path: Path, recursive: bool = True) -> ProcessingResult:
        """Process all supported files in a directory."""
        ...
    
    def process_bytes(self, content: bytes, filename: str) -> ProcessingResult:
        """Process file from bytes (for API uploads)."""
        ...
    
    def _detect_file_type(self, file_path: Path) -> str:
        ...
    
    def _extract_text_pdf(self, file_path: Path) -> Tuple[str, dict]:
        ...
    
    def _extract_text_docx(self, file_path: Path) -> Tuple[str, dict]:
        ...
    
    def _extract_text_markdown(self, file_path: Path) -> Tuple[str, dict]:
        ...
    
    def _clean_text(self, text: str) -> str:
        """Remove extra whitespace, fix encoding issues."""
        ...
    
    def _chunk_text(self, text: str, metadata: dict) -> List[Document]:
        """Split text into overlapping chunks preserving paragraph boundaries."""
        ...
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        ...

Implementation notes:
- Use pypdf for PDF extraction
- Use python-docx for Word documents
- Preserve paragraph boundaries in chunking when possible
- Add "header" context to each chunk (first 100 chars of document)
- Handle encoding errors gracefully

Output the complete file with all methods implemented.
```

---

## PROMPT 1.4: Embedding Service

```
Create the embedding service for Enterprise-RAG.

File: src/retrieval/embedding_service.py

Requirements:
1. Lazy loading of embedding model
2. Batch processing for efficiency
3. Caching of recent embeddings
4. GPU support with CPU fallback
5. Multiple model support

Implementation:

from functools import lru_cache
from typing import List, Optional
import numpy as np

class EmbeddingService:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        ...
    
    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings."""
        ...
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, handling batching automatically."""
        ...
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        ...
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (may use different prompt template)."""
        ...
    
    @lru_cache(maxsize=1000)
    def _cached_embed(self, text: str) -> tuple:
        """Cache embeddings for repeated texts."""
        ...
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        ...
    
    def get_model_info(self) -> dict:
        """Return model information for debugging."""
        ...

Include benchmark method that reports:
- Embeddings per second
- Memory usage
- Device being used

Output the complete file.
```

---

## PROMPT 1.5: Vector Store Abstraction

```
Create the vector store abstraction layer for Enterprise-RAG.

File: src/retrieval/vector_store.py

Requirements:
1. Abstract base class for vector stores
2. ChromaDB implementation for development
3. Easy to add Qdrant/Pinecone implementations
4. Metadata filtering support
5. Batch operations

Implementation:

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    content: str
    metadata: dict
    score: float
    
class VectorStoreBase(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> int:
        """Add documents with their embeddings. Returns count added."""
        ...
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        ...
    
    @abstractmethod
    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        ...
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Return store statistics."""
        ...

class ChromaVectorStore(VectorStoreBase):
    def __init__(
        self,
        collection_name: str = "enterprise_rag",
        persist_directory: str = "./data/chroma",
        embedding_function: Optional[Any] = None
    ):
        ...
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> int:
        ...
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        ...
    
    def delete(self, doc_ids: List[str]) -> int:
        ...
    
    def get_stats(self) -> dict:
        ...
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        ...

# Factory function
def create_vector_store(
    store_type: str = "chroma",
    **kwargs
) -> VectorStoreBase:
    """Create a vector store instance."""
    ...

Output the complete file with ChromaDB fully implemented.
```

---

## PROMPT 1.6: BM25 Sparse Retriever

```
Create the BM25 sparse retriever for hybrid search.

File: src/retrieval/sparse_retriever.py

Requirements:
1. BM25 index building and persistence
2. Tokenization with basic preprocessing
3. Search with score normalization
4. Index updates (add/remove documents)

Implementation:

from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

@dataclass
class SparseSearchResult:
    doc_id: str
    chunk_id: str
    content: str
    score: float
    
class BM25Retriever:
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self.bm25 = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
    
    def build_index(self, documents: List[Document]) -> None:
        """Build BM25 index from documents."""
        ...
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing index (rebuilds)."""
        ...
    
    def search(self, query: str, top_k: int = 10) -> List[SparseSearchResult]:
        """Search using BM25."""
        ...
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple word tokenization with lowercasing
        # Remove punctuation and stopwords
        ...
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores to 0-1 range."""
        ...
    
    def save_index(self, path: str) -> None:
        """Persist index to disk."""
        ...
    
    def load_index(self, path: str) -> None:
        """Load index from disk."""
        ...
    
    def get_stats(self) -> dict:
        """Return index statistics."""
        ...

Output the complete file.
```

---

## PROMPT 1.7: Hybrid Retriever with Fusion

```
Create the hybrid retriever that combines dense and sparse search.

File: src/retrieval/hybrid_retriever.py

Requirements:
1. Combine dense (vector) and sparse (BM25) results
2. Reciprocal Rank Fusion for score combination
3. Configurable weights
4. Deduplication of results

Implementation:

@dataclass
class HybridSearchResult:
    doc_id: str
    chunk_id: str
    content: str
    metadata: dict
    dense_score: float
    sparse_score: float
    fused_score: float
    
class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_service: EmbeddingService,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        filters: Optional[dict] = None
    ) -> List[HybridSearchResult]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: If False, only use dense retrieval
            filters: Metadata filters for dense search
        
        Returns:
            List of search results with fused scores
        """
        ...
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict]
    ) -> List[Tuple[str, float]]:
        """Perform dense vector search."""
        ...
    
    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Perform sparse BM25 search."""
        ...
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank_i)) for each ranking
        """
        ...
    
    def _merge_results(
        self,
        fused_rankings: List[Tuple[str, float]],
        dense_results: dict,
        sparse_results: dict,
        top_k: int
    ) -> List[HybridSearchResult]:
        """Merge rankings with original result data."""
        ...
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to both dense and sparse indexes."""
        embeddings = self.embedding_service.embed_texts(
            [doc.content for doc in documents]
        )
        self.vector_store.add_documents(documents, embeddings)
        self.bm25_retriever.add_documents(documents)
        return len(documents)

Output the complete file with RRF algorithm fully implemented.
```

---

## PROMPT 1.8: Cross-Encoder Reranker

```
Create the cross-encoder reranker for improved accuracy.

File: src/retrieval/reranker.py

Requirements:
1. Cross-encoder model for pairwise scoring
2. Batch processing for efficiency
3. Score normalization
4. Model caching

Implementation:

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 16
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self._model = None
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model
    
    def rerank(
        self,
        query: str,
        results: List[HybridSearchResult],
        top_k: int = 5
    ) -> List[HybridSearchResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original search query
            results: Results from hybrid retrieval
            top_k: Number of top results to return
        
        Returns:
            Reranked results with updated scores
        """
        ...
    
    def _batch_score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Score query-document pairs in batches."""
        ...
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using sigmoid."""
        ...
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' to actual device."""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def benchmark(self, query: str, documents: List[str]) -> dict:
        """Benchmark reranking performance."""
        ...

Output the complete file.
```

---

## PROMPT 1.9: RAG Chain - Response Generation

```
Create the main RAG chain that orchestrates retrieval and generation.

File: src/generation/rag_chain.py

Requirements:
1. Orchestrate: retrieve → rerank → generate
2. Support multiple LLM providers
3. Citation extraction
4. Streaming support
5. Conversation history (optional)

Implementation:

from typing import AsyncGenerator, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GLM = "glm"

@dataclass
class Citation:
    source: str
    chunk_id: str
    content_preview: str
    relevance_score: float

@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    retrieval_results: List[HybridSearchResult]
    model_used: str
    processing_time: float
    token_usage: Optional[dict] = None

class RAGChain:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm_provider: LLMProvider = LLMProvider.OPENAI,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self._llm_client = None
    
    @property
    def llm_client(self):
        """Lazy load LLM client based on provider."""
        ...
    
    def query(
        self,
        question: str,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5,
        use_reranking: bool = True,
        filters: Optional[dict] = None
    ) -> RAGResponse:
        """
        Execute RAG query: retrieve, rerank, generate.
        
        Args:
            question: User's question
            top_k_retrieve: Number of documents to retrieve
            top_k_rerank: Number of documents after reranking
            use_reranking: Whether to apply cross-encoder reranking
            filters: Metadata filters for retrieval
        
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve
        ...
        
        # Step 2: Rerank (optional)
        ...
        
        # Step 3: Build context
        context = self._build_context(results)
        
        # Step 4: Generate response
        answer, token_usage = self._generate(question, context)
        
        # Step 5: Extract citations
        citations = self._extract_citations(answer, results)
        
        processing_time = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieval_results=results,
            model_used=self.model_name,
            processing_time=processing_time,
            token_usage=token_usage
        )
    
    async def query_stream(
        self,
        question: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream the response for real-time display."""
        ...
    
    def _build_context(self, results: List[HybridSearchResult]) -> str:
        """Build context string from retrieval results."""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] Source: {result.metadata.get('source', 'Unknown')}\n"
                f"{result.content}\n"
            )
        return "\n---\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Return the system prompt for RAG."""
        return """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only answer based on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Cite your sources using [1], [2], etc. matching the context numbers
4. Be concise but thorough
5. If you're uncertain, express that uncertainty"""
    
    def _generate(self, question: str, context: str) -> Tuple[str, dict]:
        """Generate response using LLM."""
        ...
    
    def _extract_citations(
        self,
        answer: str,
        results: List[HybridSearchResult]
    ) -> List[Citation]:
        """Extract citation references from the answer."""
        import re
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set(int(n) for n in re.findall(citation_pattern, answer))
        
        citations = []
        for num in sorted(cited_numbers):
            if 1 <= num <= len(results):
                result = results[num - 1]
                citations.append(Citation(
                    source=result.metadata.get('source', 'Unknown'),
                    chunk_id=result.chunk_id,
                    content_preview=result.content[:200] + "...",
                    relevance_score=result.fused_score
                ))
        
        return citations

Include implementations for all LLM providers (OpenAI, Anthropic, Ollama, GLM).
Output the complete file.
```

---

## PROMPT 1.10: RAGAS Evaluation Module

```
Create the evaluation module using RAGAS metrics.

File: src/evaluation/rag_evaluator.py

Requirements:
1. Integrate RAGAS evaluation metrics
2. Create and manage test datasets
3. Run batch evaluations
4. Generate evaluation reports

Implementation:

from dataclasses import dataclass
from typing import List, Optional
import json
from pathlib import Path

@dataclass
class EvaluationSample:
    question: str
    ground_truth: str
    contexts: Optional[List[str]] = None
    answer: Optional[str] = None

@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    individual_results: List[dict]
    evaluation_time: float

class RAGEvaluator:
    def __init__(self, rag_chain: RAGChain):
        self.rag_chain = rag_chain
        self._ragas_metrics = None
    
    @property
    def ragas_metrics(self):
        """Lazy load RAGAS metrics."""
        if self._ragas_metrics is None:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            self._ragas_metrics = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall
            }
        return self._ragas_metrics
    
    def create_test_dataset(
        self,
        samples: List[EvaluationSample]
    ) -> 'Dataset':
        """Create a RAGAS-compatible dataset."""
        from datasets import Dataset
        
        data = {
            'question': [],
            'ground_truth': [],
            'contexts': [],
            'answer': []
        }
        
        for sample in samples:
            # Generate answer if not provided
            if sample.answer is None:
                response = self.rag_chain.query(sample.question)
                sample.answer = response.answer
                sample.contexts = [r.content for r in response.retrieval_results]
            
            data['question'].append(sample.question)
            data['ground_truth'].append(sample.ground_truth)
            data['contexts'].append(sample.contexts or [])
            data['answer'].append(sample.answer)
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        test_dataset: 'Dataset',
        metrics: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation on a dataset.
        
        Args:
            test_dataset: Dataset with questions, answers, contexts, ground_truth
            metrics: List of metric names to compute (default: all)
        
        Returns:
            EvaluationResult with all metric scores
        """
        from ragas import evaluate
        import time
        
        start_time = time.time()
        
        metrics_to_use = [
            self.ragas_metrics[m] 
            for m in (metrics or self.ragas_metrics.keys())
        ]
        
        results = evaluate(test_dataset, metrics=metrics_to_use)
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            faithfulness=results.get('faithfulness', 0.0),
            answer_relevancy=results.get('answer_relevancy', 0.0),
            context_precision=results.get('context_precision', 0.0),
            context_recall=results.get('context_recall', 0.0),
            overall_score=sum(results.values()) / len(results),
            individual_results=results.to_pandas().to_dict('records'),
            evaluation_time=evaluation_time
        )
    
    def evaluate_single(
        self,
        question: str,
        ground_truth: str
    ) -> dict:
        """Evaluate a single question-answer pair."""
        sample = EvaluationSample(question=question, ground_truth=ground_truth)
        dataset = self.create_test_dataset([sample])
        result = self.evaluate(dataset)
        return result.individual_results[0]
    
    def generate_report(self, result: EvaluationResult) -> str:
        """Generate a markdown evaluation report."""
        report = f"""# RAG Evaluation Report

## Overall Scores

| Metric | Score |
|--------|-------|
| Faithfulness | {result.faithfulness:.3f} |
| Answer Relevancy | {result.answer_relevancy:.3f} |
| Context Precision | {result.context_precision:.3f} |
| Context Recall | {result.context_recall:.3f} |
| **Overall** | **{result.overall_score:.3f}** |

## Evaluation Details

- Samples evaluated: {len(result.individual_results)}
- Evaluation time: {result.evaluation_time:.2f}s

## Interpretation

- **Faithfulness**: How factually consistent is the answer with the context?
- **Answer Relevancy**: How relevant is the answer to the question?
- **Context Precision**: How relevant are the retrieved contexts?
- **Context Recall**: How much of the ground truth is covered by contexts?

## Individual Results

"""
        for i, individual in enumerate(result.individual_results, 1):
            report += f"### Sample {i}\n"
            for key, value in individual.items():
                if isinstance(value, float):
                    report += f"- {key}: {value:.3f}\n"
                else:
                    report += f"- {key}: {value}\n"
            report += "\n"
        
        return report
    
    def load_test_dataset(self, path: str) -> List[EvaluationSample]:
        """Load test dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return [EvaluationSample(**item) for item in data]
    
    def save_results(self, result: EvaluationResult, path: str) -> None:
        """Save evaluation results to JSON."""
        ...

Include default test dataset with 10 sample questions.
Output the complete file.
```

---

## PROMPT 1.11: FastAPI Application

```
Create the FastAPI application for Enterprise-RAG.

File: src/api/main.py

Requirements:
1. RESTful API with proper documentation
2. Async endpoints where beneficial
3. Error handling with proper HTTP codes
4. Request validation with Pydantic
5. Health checks

Implementation:

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time

# Pydantic models for requests/responses
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    filters: Optional[dict] = None

class Citation(BaseModel):
    source: str
    chunk_id: str
    content_preview: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    processing_time: float
    model_used: str

class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    status: str

class EvaluationRequest(BaseModel):
    samples: Optional[List[dict]] = None  # If None, use default test set

class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float

class HealthResponse(BaseModel):
    status: str
    components: dict
    timestamp: str

# App setup
app = FastAPI(
    title="Enterprise-RAG API",
    description="Production-grade RAG system with hybrid retrieval and evaluation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection for RAG components
# (Initialize on startup)

@app.on_event("startup")
async def startup():
    """Initialize RAG components on startup."""
    ...

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Returns an answer with citations from the knowledge base.
    """
    try:
        response = rag_chain.query(
            question=request.question,
            top_k_rerank=request.top_k,
            use_reranking=request.use_reranking,
            filters=request.filters
        )
        return QueryResponse(
            answer=response.answer,
            citations=[Citation(**c.__dict__) for c in response.citations],
            processing_time=response.processing_time,
            model_used=response.model_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and ingest a document into the knowledge base.
    
    Supports: PDF, DOCX, MD, TXT
    """
    ...

@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    ...

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base."""
    ...

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    Run RAGAS evaluation on the RAG system.
    
    If no samples provided, uses default test dataset.
    """
    ...

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all components."""
    components = {
        "vector_store": "healthy",
        "embedding_model": "healthy",
        "reranker": "healthy",
        "llm": "healthy"
    }
    
    # Check each component
    try:
        vector_store.get_stats()
    except:
        components["vector_store"] = "unhealthy"
    
    # ... check others
    
    return HealthResponse(
        status="healthy" if all(v == "healthy" for v in components.values()) else "degraded",
        components=components,
        timestamp=datetime.utcnow().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

Output the complete file with all endpoints implemented.
```

---

## PROMPT 1.12: Streamlit Demo UI

```
Create the Streamlit demo interface for Enterprise-RAG.

File: src/ui/app.py

Requirements:
1. Professional chat interface
2. Source display with scores
3. Document upload
4. Settings panel
5. Evaluation trigger

Implementation:

import streamlit as st
import requests
from typing import List
import time

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise-RAG Demo",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .source-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .score-bar {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
    }
    .score-fill {
        height: 100%;
        background-color: #4CAF50;
        border-radius: 4px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model settings
    st.subheader("Retrieval Settings")
    top_k = st.slider("Number of sources", 1, 10, 5)
    use_reranking = st.checkbox("Use cross-encoder reranking", value=True)
    
    st.divider()
    
    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "md", "txt"]
    )
    if uploaded_file and st.button("Upload"):
        with st.spinner("Processing document..."):
            files = {"file": uploaded_file}
            response = requests.post(f"{API_URL}/ingest", files=files)
            if response.ok:
                result = response.json()
                st.success(f"✅ Created {result['chunks_created']} chunks")
            else:
                st.error("Failed to upload document")
    
    st.divider()
    
    # Sample questions
    st.subheader("💡 Try These Questions")
    sample_questions = [
        "What is the main topic of the documents?",
        "Summarize the key points",
        "What are the important dates mentioned?",
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.sample_question = q
    
    st.divider()
    
    # Evaluation
    st.subheader("📊 Evaluation")
    if st.button("Run RAGAS Evaluation"):
        with st.spinner("Evaluating..."):
            response = requests.post(f"{API_URL}/evaluate", json={})
            if response.ok:
                result = response.json()
                st.metric("Faithfulness", f"{result['faithfulness']:.2f}")
                st.metric("Answer Relevancy", f"{result['answer_relevancy']:.2f}")
                st.metric("Context Precision", f"{result['context_precision']:.2f}")
                st.metric("Overall", f"{result['overall_score']:.2f}")

# Main chat area
st.title("🔍 Enterprise-RAG")
st.caption("Production-grade RAG system with hybrid retrieval")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": prompt,
                    "top_k": top_k,
                    "use_reranking": use_reranking
                }
            )
            
            if response.ok:
                result = response.json()
                st.markdown(result["answer"])
                
                # Show sources in expander
                with st.expander("📚 Sources", expanded=False):
                    for i, citation in enumerate(result["citations"], 1):
                        st.markdown(f"""
                        **[{i}] {citation['source']}**
                        
                        > {citation['content_preview']}
                        
                        Relevance: {citation['relevance_score']:.2%}
                        
                        ---
                        """)
                
                st.caption(f"⏱️ {result['processing_time']:.2f}s | 🤖 {result['model_used']}")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"]
                })
            else:
                st.error("Failed to get response")

# Handle sample question selection
if hasattr(st.session_state, 'sample_question'):
    st.session_state.messages.append({
        "role": "user",
        "content": st.session_state.sample_question
    })
    del st.session_state.sample_question
    st.rerun()

Output the complete file with professional styling.
```

---

## PROMPT 1.13: Docker Configuration

```
Create Docker configuration for Enterprise-RAG.

Files:
1. Dockerfile
2. docker-compose.yml
3. docker-compose.dev.yml
4. .dockerignore

Dockerfile:
- Multi-stage build
- Python 3.11-slim base
- Non-root user
- Health check
- Optimized layer caching

docker-compose.yml:
- rag-api service (FastAPI on port 8000)
- rag-ui service (Streamlit on port 8501)
- qdrant service (vector DB)
- Volumes for persistence
- Health checks
- Environment variables

docker-compose.dev.yml:
- Hot reload for development
- ChromaDB instead of Qdrant
- Debug ports

Output all files completely.
```

---

## PROMPT 1.14: Tests

```
Create comprehensive tests for Enterprise-RAG.

Files:
1. tests/conftest.py - Fixtures
2. tests/test_document_processor.py
3. tests/test_retrieval.py
4. tests/test_rag_chain.py
5. tests/test_api.py

conftest.py should include fixtures for:
- Sample documents (create temp files)
- Mock embedding service
- Mock LLM responses
- Test vector store (in-memory)
- Test RAG chain

Test coverage:
1. Document processing: file types, chunking, metadata
2. Retrieval: dense, sparse, hybrid, reranking
3. Generation: context building, citation extraction
4. API: all endpoints, error handling

Use pytest with:
- Async test support
- Parametrized tests
- Markers for slow tests

Output all files with at least 30 test cases total.
```

---

## PROMPT 1.15: README Documentation

```
Create comprehensive README.md for Enterprise-RAG.

Include:

1. Project Title and Badges
   - Python version
   - License
   - Tests passing
   - Code coverage

2. Overview
   - What it does
   - Key features
   - Architecture diagram (ASCII)

3. Features
   - Hybrid retrieval
   - Cross-encoder reranking
   - Multi-format ingestion
   - RAGAS evaluation
   - Production API

4. Quick Start
   - Prerequisites
   - Installation steps
   - Running locally
   - Running with Docker

5. API Documentation
   - Endpoints table
   - Request/response examples

6. Evaluation Results
   - RAGAS metrics on sample dataset
   - Comparison with/without reranking

7. Architecture
   - Component diagram
   - Data flow

8. Configuration
   - Environment variables
   - Customization options

9. Development
   - Running tests
   - Contributing

10. License

Output the complete README.md with proper Markdown formatting.
```

---

# PROJECT 1B: Multi-Modal RAG
## RAG System with Image and Table Understanding

### Project Overview

**What You'll Build**: A RAG system that handles:
- Text documents
- Images with descriptions
- Tables with structured extraction
- Combined multi-modal queries

**Why This Matters for Jobs**:
- Differentiates you from basic RAG implementations
- Shows understanding of real enterprise documents
- Demonstrates cutting-edge capabilities

**Time Estimate**: 7-10 days (after completing 1A)

---

## SESSION SETUP PROMPT

```
You are an expert Python developer helping me build a multi-modal RAG system.

PROJECT: Multi-Modal RAG
PURPOSE: Extend basic RAG to handle:
- Images with automatic captioning/OCR
- Tables with structured extraction
- Combined text + visual queries

TECH STACK:
- LlamaIndex multi-modal extensions
- GPT-4V or LLaVA for image understanding
- Unstructured.io for document parsing
- Tabula/Camelot for table extraction
- FastAPI + Streamlit
- Python 3.11+

This is an ADVANCED RAG project building on Enterprise-RAG (Project 1A).

RULES:
1. Generate complete, runnable code
2. Include all imports
3. Handle edge cases
4. Production-ready quality

Please confirm you understand, then we'll build this file by file.
```

---

## PROMPT 1B.1: Project Structure

```
Create the project structure for Multi-Modal RAG.

This extends Enterprise-RAG with additional modules:

New directories:
- src/multimodal/
- src/extraction/
- data/sample_images/
- data/sample_tables/

New files:
- src/multimodal/image_processor.py
- src/multimodal/table_extractor.py
- src/multimodal/multimodal_retriever.py
- src/multimodal/vision_llm.py

Additional requirements:
- unstructured[all-docs]
- tabula-py
- pillow
- transformers (for local vision models)

Output the updated requirements.txt and new directory structure.
```

---

## PROMPT 1B.2: Image Processing & Captioning

```
Create the image processing module.

File: src/multimodal/image_processor.py

Requirements:
1. Extract images from documents (PDF, DOCX)
2. Generate captions using vision LLM
3. OCR for text in images
4. Store image embeddings

Class: ImageProcessor
Methods:
- extract_images_from_pdf(pdf_path: Path) -> List[ImageData]
- extract_images_from_docx(docx_path: Path) -> List[ImageData]
- generate_caption(image: Image) -> str
- extract_text_ocr(image: Image) -> str
- get_image_embedding(image: Image) -> np.ndarray

ImageData:
- image_bytes: bytes
- caption: str
- ocr_text: str
- source_doc: str
- page_number: int
- embedding: np.ndarray

Support both:
- GPT-4V API for captions (cloud)
- LLaVA for local captioning (optional)

Output the complete file.
```

---

## PROMPT 1B.3: Table Extraction

```
Create the table extraction module.

File: src/multimodal/table_extractor.py

Requirements:
1. Extract tables from PDFs
2. Convert to structured format
3. Generate natural language descriptions
4. Index both structured and descriptions

Class: TableExtractor
Methods:
- extract_tables_from_pdf(pdf_path: Path) -> List[TableData]
- convert_to_dataframe(table: Any) -> pd.DataFrame
- generate_table_description(df: pd.DataFrame) -> str
- get_table_embedding(description: str) -> np.ndarray

TableData:
- dataframe: pd.DataFrame
- description: str
- source_doc: str
- page_number: int
- embedding: np.ndarray

Use tabula-py for extraction with Camelot as fallback.

Output the complete file.
```

---

## PROMPT 1B.4: Multi-Modal Retriever

```
Create the multi-modal retriever that searches across text, images, and tables.

File: src/multimodal/multimodal_retriever.py

Requirements:
1. Unified search across all content types
2. Content-type filtering
3. Combined scoring
4. Rich results with previews

Class: MultiModalRetriever
Methods:
- retrieve(query: str, content_types: List[str], top_k: int) -> List[MultiModalResult]
- _search_text(query: str, top_k: int) -> List[SearchResult]
- _search_images(query: str, top_k: int) -> List[ImageResult]
- _search_tables(query: str, top_k: int) -> List[TableResult]
- _merge_results(results: List[Any], top_k: int) -> List[MultiModalResult]

MultiModalResult:
- content_type: Literal["text", "image", "table"]
- content: Union[str, bytes, pd.DataFrame]
- preview: str
- metadata: dict
- score: float

Output the complete file.
```

---

## PROMPT 1B.5: Multi-Modal RAG Chain

```
Create the multi-modal RAG chain.

File: src/multimodal/multimodal_rag.py

Requirements:
1. Handle queries about images and tables
2. Include visual context in prompts
3. Generate responses with multi-modal citations

Class: MultiModalRAGChain
Methods:
- query(question: str, include_images: bool, include_tables: bool) -> MultiModalResponse
- _build_multimodal_context(results: List[MultiModalResult]) -> str
- _generate_with_vision(question: str, context: str, images: List[bytes]) -> str

MultiModalResponse:
- answer: str
- text_citations: List[Citation]
- image_citations: List[ImageCitation]
- table_citations: List[TableCitation]

Output the complete file.
```

---

## PROMPT 1B.6: Updated API & UI

```
Update the FastAPI and Streamlit to support multi-modal RAG.

File: src/api/multimodal_endpoints.py
- POST /query/multimodal - Query with content type filters
- POST /ingest/multimodal - Ingest with image/table extraction
- GET /images/{image_id} - Get image by ID
- GET /tables/{table_id} - Get table by ID

File: src/ui/multimodal_app.py
- Image preview in sources
- Table preview with formatting
- Content type filters
- Visual results display

Output both files.
```

---

# END OF PROJECT 1 (RAG) PROMPTS

---

## Summary: RAG Projects

| Project | Focus | Complexity | Time |
|---------|-------|------------|------|
| **1A: Enterprise-RAG** | Core RAG skills | Medium | 10-14 days |
| **1B: Multi-Modal RAG** | Advanced differentiation | High | 7-10 days |

**My Recommendation**: 
- Build 1A completely first
- Add 1B only if you have time OR if applying to jobs requiring multi-modal
- 1A alone is sufficient for 90% of job applications
