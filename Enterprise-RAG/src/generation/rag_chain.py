# ============================================================
# Enterprise-RAG: RAG Chain Orchestration
# ============================================================
"""
Main RAG chain orchestrating retrieval and generation.

This module provides the complete RAG pipeline:
1. Retrieve relevant documents (hybrid search)
2. Rerank for accuracy (cross-encoder)
3. Build context from results
4. Generate answer with LLM
5. Extract citations

Supports multiple LLM providers with streaming and conversation history.

Example:
    >>> from src.generation import RAGChain, LLMProvider
    >>> chain = RAGChain(
    ...     retriever=hybrid_retriever,
    ...     reranker=reranker,
    ...     llm_provider=LLMProvider.OPENAI,
    ...     model_name="gpt-4o-mini"
    ... )
    >>> response = chain.query("What is the refund policy?")
    >>> print(response.answer)
    >>> for citation in response.citations:
    ...     print(f"[{citation.source}]")
"""

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Optional, Tuple

from src.config import settings
from src.logging_config import get_logger
from src.retrieval.hybrid_retriever import HybridSearchResult
from src.retrieval.reranker import CrossEncoderReranker, RerankedSearchResult

# Initialize logger
logger = get_logger(__name__)


# ============================================================
# Enums
# ============================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GLM = "glm"


# ============================================================
# Data Classes
# ============================================================

@dataclass(frozen=True)
class Citation:
    """
    Citation reference for a source used in the answer.

    Attributes:
        source: Source document/path
        chunk_id: Unique chunk identifier
        content_preview: Preview of cited content
        relevance_score: Relevance score from retrieval
        page_number: Optional page number

    Example:
        >>> citation = Citation(
        ...     source="policy.pdf",
        ...     chunk_id="doc_123_chunk_0",
        ...     content_preview="Refunds are processed within...",
        ...     relevance_score=0.92,
        ...     page_number=5
        ... )
    """

    source: str
    chunk_id: str
    content_preview: str
    relevance_score: float
    page_number: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "content_preview": self.content_preview,
            "relevance_score": self.relevance_score,
            "page_number": self.page_number,
        }


@dataclass
class RAGResponse:
    """
    Complete response from RAG query.

    Attributes:
        answer: Generated answer text
        citations: List of citation references
        retrieval_results: All retrieval results used
        model_used: LLM model name
        provider_used: LLM provider
        processing_time: Total processing time in seconds
        token_usage: Optional token usage information
        retrieval_time: Time spent on retrieval
        rerank_time: Time spent on reranking
        generation_time: Time spent on generation

    Example:
        >>> response = chain.query("What is the refund policy?")
        >>> print(f"Answer: {response.answer}")
        >>> print(f"Sources: {len(response.citations)}")
        >>> print(f"Time: {response.processing_time:.2f}s")
    """

    answer: str
    citations: list[Citation]
    retrieval_results: list[HybridSearchResult]
    model_used: str
    provider_used: LLMProvider
    processing_time: float
    token_usage: Optional[dict[str, int]] = None
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    generation_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "num_sources": len(self.citations),
            "retrieval_results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": getattr(r, "rerank_score", None) or r.fused_score,
                    "content": r.content[:200] + "...",
                }
                for r in self.retrieval_results
            ],
            "model_used": self.model_used,
            "provider_used": self.provider_used.value,
            "processing_time": round(self.processing_time, 3),
            "retrieval_time": round(self.retrieval_time, 3),
            "rerank_time": round(self.rerank_time, 3),
            "generation_time": round(self.generation_time, 3),
            "token_usage": self.token_usage,
        }


@dataclass
class Message:
    """
    Chat message for conversation history.

    Attributes:
        role: Message role ("user", "assistant", "system")
        content: Message content
        timestamp: Message timestamp
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================
# RAG Chain
# ============================================================

class RAGChain:
    """
    Main RAG chain orchestrating retrieval and generation.

    Pipeline:
        1. Retrieve relevant documents (hybrid dense + sparse)
        2. Rerank for accuracy (cross-encoder)
        3. Build context from top results
        4. Generate answer with LLM
        5. Extract citations from answer

    Supports multiple LLM providers with streaming and conversation history.

    Args:
        retriever: HybridRetriever instance
        reranker: CrossEncoderReranker instance
        llm_provider: LLM provider to use
        model_name: Model name for the provider
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        top_k_retrieve: Number of documents to retrieve
        top_k_rerank: Number of documents after reranking
        system_prompt: Custom system prompt (optional)

    Example:
        >>> chain = RAGChain(
        ...     retriever=hybrid_retriever,
        ...     reranker=reranker,
        ...     llm_provider=LLMProvider.OPENAI,
        ...     model_name="gpt-4o-mini"
        ... )
        >>> response = chain.query("What is the refund policy?")
    """

    def __init__(
        self,
        retriever: Any,  # HybridRetriever
        reranker: Any,  # CrossEncoderReranker
        llm_provider: LLMProvider = LLMProvider.OPENAI,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the RAG chain."""
        # Components
        self.retriever = retriever
        self.reranker = reranker

        # LLM configuration
        self.llm_provider = llm_provider
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS

        # Retrieval configuration
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # System prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Lazy-loaded LLM client
        self._llm_client: Optional[Any] = None
        self._client_lock = threading.Lock()

        # Conversation history
        self.conversation_history: list[Message] = []

        logger.info(
            "RAGChain initialized",
            extra={
                "provider": llm_provider.value,
                "model": self.model_name,
                "temperature": self.temperature,
                "top_k_retrieve": top_k_retrieve,
                "top_k_rerank": top_k_rerank,
            },
        )

    # ============================================================
    # Properties
    # ============================================================

    @property
    def llm_client(self) -> Any:
        """Lazy load LLM client based on provider."""
        if self._llm_client is None:
            with self._client_lock:
                if self._llm_client is None:
                    self._load_llm_client()
        return self._llm_client

    def _load_llm_client(self) -> None:
        """Load the appropriate LLM client."""
        logger.info(f"Loading LLM client: {self.llm_provider.value}")

        if self.llm_provider == LLMProvider.OPENAI:
            self._load_openai_client()
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            self._load_anthropic_client()
        elif self.llm_provider == LLMProvider.OLLAMA:
            self._load_ollama_client()
        elif self.llm_provider == LLMProvider.GLM:
            self._load_glm_client()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        logger.info(f"LLM client loaded: {self.llm_provider.value}")

    def _load_openai_client(self) -> None:
        """Load OpenAI client."""
        try:
            from openai import OpenAI

            self._llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider")

    def _load_anthropic_client(self) -> None:
        """Load Anthropic client."""
        try:
            from anthropic import Anthropic

            api_key = settings.ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")

            self._llm_client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic provider")

    def _load_ollama_client(self) -> None:
        """Load Ollama client."""
        try:
            from openai import OpenAI

            # Ollama uses OpenAI-compatible API
            self._llm_client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # Required but not used
            )
        except ImportError:
            raise ImportError("openai package is required for Ollama provider")

    def _load_glm_client(self) -> None:
        """Load GLM (ZhipuAI) client."""
        try:
            from zhipuai import ZhipuAI

            api_key = settings.OPENAI_API_KEY  # Reuse or add GLM_API_KEY
            if not api_key:
                raise ValueError("API key is required for GLM provider")

            self._llm_client = ZhipuAI(api_key=api_key)
        except ImportError:
            raise ImportError("zhipuai package is required for GLM provider")

    # ============================================================
    # Query Methods
    # ============================================================

    def query(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        use_reranking: bool = True,
        filters: Optional[dict[str, Any]] = None,
        include_history: bool = False,
    ) -> RAGResponse:
        """
        Execute RAG query: retrieve, rerank, generate.

        Args:
            question: User's question
            top_k_retrieve: Number of documents to retrieve
            top_k_rerank: Number of documents after reranking
            use_reranking: Whether to apply cross-encoder reranking
            filters: Metadata filters for retrieval
            include_history: Whether to include conversation history

        Returns:
            RAGResponse with answer, citations, and metadata

        Example:
            >>> response = chain.query(
            ...     "What is the refund policy?",
            ...     top_k_retrieve=20,
            ...     top_k_rerank=5
            ... )
            >>> print(response.answer)
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        start_time = time.time()

        # Use configured defaults if not specified
        top_k_retrieve = top_k_retrieve or self.top_k_retrieve
        top_k_rerank = top_k_rerank or self.top_k_rerank

        logger.info(
            f"RAG query: {question[:100]}",
            extra={
                "top_k_retrieve": top_k_retrieve,
                "top_k_rerank": top_k_rerank,
                "use_reranking": use_reranking,
            },
        )

        try:
            # Step 1: Retrieve
            retrieval_start = time.time()
            retrieval_results = self.retriever.retrieve(
                query=question,
                top_k=top_k_retrieve,
                use_hybrid=True,
                filters=filters,
            )
            retrieval_time = time.time() - retrieval_start

            logger.info(f"Retrieved {len(retrieval_results)} documents in {retrieval_time:.3f}s")

            if not retrieval_results:
                # No results found
                return RAGResponse(
                    answer="I don't have enough information to answer this question. "
                    "No relevant documents were found.",
                    citations=[],
                    retrieval_results=[],
                    model_used=self.model_name,
                    provider_used=self.llm_provider,
                    processing_time=time.time() - start_time,
                    retrieval_time=retrieval_time,
                )

            # Step 2: Rerank
            rerank_time = 0.0
            final_results = retrieval_results

            if use_reranking:
                rerank_start = time.time()
                reranked_results = self.reranker.rerank(
                    query=question,
                    results=retrieval_results[:top_k_retrieve],
                    top_k=top_k_rerank,
                )
                rerank_time = time.time() - rerank_start

                # Use reranked results
                final_results = reranked_results
                logger.info(f"Reranked to {len(final_results)} documents in {rerank_time:.3f}s")

            # Limit to top_k_rerank
            final_results = final_results[:top_k_rerank]

            # Step 3: Build context
            context = self._build_context(final_results)

            # Step 4: Generate response
            generation_start = time.time()
            answer, token_usage = self._generate(
                question=question,
                context=context,
                include_history=include_history,
            )
            generation_time = time.time() - generation_start

            logger.info(f"Generated answer in {generation_time:.3f}s")

            # Step 5: Extract citations
            citations = self._extract_citations(answer, final_results)

            # Calculate total time
            processing_time = time.time() - start_time

            # Add to conversation history
            self.conversation_history.append(Message(role="user", content=question))
            self.conversation_history.append(Message(role="assistant", content=answer))

            response = RAGResponse(
                answer=answer,
                citations=citations,
                retrieval_results=final_results,
                model_used=self.model_name,
                provider_used=self.llm_provider,
                processing_time=processing_time,
                token_usage=token_usage,
                retrieval_time=retrieval_time,
                rerank_time=rerank_time,
                generation_time=generation_time,
            )

            logger.info(
                f"RAG query complete in {processing_time:.3f}s",
                extra={
                    "citations": len(citations),
                    "tokens": token_usage.get("total_tokens") if token_usage else None,
                },
            )

            return response

        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}", exc_info=True)
            raise

    async def query_stream(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        use_reranking: bool = True,
        filters: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the response for real-time display.

        Args:
            question: User's question
            top_k_retrieve: Number of documents to retrieve
            top_k_rerank: Number of documents after reranking
            use_reranking: Whether to apply reranking
            filters: Metadata filters

        Yields:
            Response text chunks as they're generated

        Example:
            >>> async for chunk in chain.query_stream("What is AI?"):
            ...     print(chunk, end="", flush=True)
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Perform retrieval and reranking (same as query)
        top_k_retrieve = top_k_retrieve or self.top_k_retrieve
        top_k_rerank = top_k_rerank or self.top_k_rerank

        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k_retrieve,
            use_hybrid=True,
            filters=filters,
        )

        if not retrieval_results:
            yield "I don't have enough information to answer this question."
            return

        if use_reranking:
            final_results = self.reranker.rerank(
                query=question,
                results=retrieval_results[:top_k_retrieve],
                top_k=top_k_rerank,
            )
        else:
            final_results = retrieval_results[:top_k_rerank]

        context = self._build_context(final_results)

        # Stream generation
        async for chunk in self._generate_stream(question, context):
            yield chunk

    # ============================================================
    # Context Building
    # ============================================================

    def _build_context(self, results: list[HybridSearchResult]) -> str:
        """
        Build context string from retrieval results.

        Args:
            results: List of search results

        Returns:
            Formatted context string

        Example:
            >>> context = chain._build_context(results)
            >>> print(context)
        """
        context_parts = []

        for i, result in enumerate(results, start=1):
            # Get source info
            source = result.metadata.get("source", "Unknown")
            page = result.metadata.get("page")
            page_str = f" (Page {page})" if page else ""

            # Build context entry
            entry = f"[{i}] Source: {source}{page_str}\n{result.content}"
            context_parts.append(entry)

        return "\n\n---\n\n".join(context_parts)

    def _get_default_system_prompt(self) -> str:
        """Return the default system prompt for RAG."""
        return """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only answer based on the provided context - do not use outside knowledge
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Cite your sources using [1], [2], etc. matching the context numbers
4. Be concise but thorough
5. If you're uncertain about something, express that uncertainty
6. Use bullet points when listing multiple items
7. Preserve important numbers, dates, and names from the context"""

    # ============================================================
    # Generation
    # ============================================================

    def _generate(
        self,
        question: str,
        context: str,
        include_history: bool = False,
    ) -> Tuple[str, Optional[dict[str, int]]]:
        """
        Generate response using LLM.

        Args:
            question: User's question
            context: Retrieved context
            include_history: Whether to include conversation history

        Returns:
            Tuple of (answer, token_usage)

        Example:
            >>> answer, usage = chain._generate(question, context)
            >>> print(answer)
        """
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add conversation history if requested
        if include_history and self.conversation_history:
            for msg in self.conversation_history[-6:]:  # Last 3 exchanges
                messages.append({"role": msg.role, "content": msg.content})

        # Add current question with context
        user_message = f"""Context:
{context}

Question: {question}

Answer:"""
        messages.append({"role": "user", "content": user_message})

        # Generate based on provider
        if self.llm_provider == LLMProvider.OPENAI:
            return self._generate_openai(messages)
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            return self._generate_anthropic(messages)
        elif self.llm_provider == LLMProvider.OLLAMA:
            return self._generate_ollama(messages)
        elif self.llm_provider == LLMProvider.GLM:
            return self._generate_glm(messages)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _generate_openai(
        self,
        messages: list[dict[str, str]],
    ) -> Tuple[str, Optional[dict[str, int]]]:
        """Generate using OpenAI."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return answer, token_usage

        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}", exc_info=True)
            raise

    def _generate_anthropic(
        self,
        messages: list[dict[str, str]],
    ) -> Tuple[str, Optional[dict[str, int]]]:
        """Generate using Anthropic."""
        try:
            # Anthropic uses different message format
            # System message should be separate
            system_message = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            response = self.llm_client.messages.create(
                model=self.model_name,
                system=system_message or self.system_prompt,
                messages=user_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            answer = response.content[0].text
            token_usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return answer, token_usage

        except Exception as e:
            logger.error(f"Anthropic generation failed: {str(e)}", exc_info=True)
            raise

    def _generate_ollama(
        self,
        messages: list[dict[str, str]],
    ) -> Tuple[str, Optional[dict[str, int]]]:
        """Generate using Ollama (local models)."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,  # e.g., "llama2", "mistral"
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )

            answer = response.choices[0].message.content
            # Ollama doesn't provide token usage
            return answer, None

        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}", exc_info=True)
            raise

    def _generate_glm(
        self,
        messages: list[dict[str, str]],
    ) -> Tuple[str, Optional[dict[str, int]]]:
        """Generate using GLM (ZhipuAI)."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,  # e.g., "glm-4"
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return answer, token_usage

        except Exception as e:
            logger.error(f"GLM generation failed: {str(e)}", exc_info=True)
            raise

    async def _generate_stream(
        self,
        question: str,
        context: str,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ]

        try:
            if self.llm_provider == LLMProvider.OPENAI:
                async for chunk in self._stream_openai(messages):
                    yield chunk
            elif self.llm_provider == LLMProvider.OLLAMA:
                async for chunk in self._stream_ollama(messages):
                    yield chunk
            else:
                # For providers without streaming support, fall back to non-streaming
                answer, _ = self._generate(question, context)
                yield answer

        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}", exc_info=True)
            raise

    async def _stream_openai(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Stream from OpenAI."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _sync_stream():
            return self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

        # Run in thread pool to avoid blocking
        stream = await loop.run_in_executor(None, _sync_stream)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_ollama(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Stream from Ollama."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _sync_stream():
            return self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )

        stream = await loop.run_in_executor(None, _sync_stream)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ============================================================
    # Citation Extraction
    # ============================================================

    def _extract_citations(
        self,
        answer: str,
        results: list[HybridSearchResult],
    ) -> list[Citation]:
        """
        Extract citation references from the answer.

        Looks for patterns like [1], [2], etc. and maps them to sources.

        Args:
            answer: Generated answer
            results: Retrieval results

        Returns:
            List of Citation objects

        Example:
            >>> citations = chain._extract_citations(answer, results)
            >>> for cit in citations:
            ...     print(f"{cit.source}: {cit.relevance_score:.2f}")
        """
        citation_pattern = r"\[(\d+)\]"
        cited_numbers = set(int(n) for n in re.findall(citation_pattern, answer))

        citations = []

        for num in sorted(cited_numbers):
            if 1 <= num <= len(results):
                result = results[num - 1]

                # Get page number if available
                page_number = result.metadata.get("page")

                citation = Citation(
                    source=result.metadata.get("source", "Unknown"),
                    chunk_id=result.chunk_id,
                    content_preview=result.content[:200] + "...",
                    relevance_score=getattr(result, "rerank_score", None)
                    or result.fused_score,
                    page_number=page_number,
                )
                citations.append(citation)

        return citations

    # ============================================================
    # Conversation Management
    # ============================================================

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt."""
        self.system_prompt = prompt
        logger.info("System prompt updated")

    # ============================================================
    # Statistics
    # ============================================================

    def get_stats(self) -> dict[str, Any]:
        """Get chain statistics."""
        return {
            "provider": self.llm_provider.value,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k_retrieve": self.top_k_retrieve,
            "top_k_rerank": self.top_k_rerank,
            "conversation_length": len(self.conversation_history),
        }


# ============================================================
# Utility Functions
# ============================================================

def create_rag_chain(
    retriever: Any,
    reranker: Any,
    llm_provider: str = "openai",
    **kwargs: Any,
) -> RAGChain:
    """
    Create a RAG chain configured from settings.

    Args:
        retriever: HybridRetriever instance
        reranker: CrossEncoderReranker instance
        llm_provider: LLM provider name
        **kwargs: Additional configuration

    Returns:
        Configured RAGChain instance

    Example:
        >>> chain = create_rag_chain(
        ...     retriever=hybrid_retriever,
        ...     reranker=reranker,
        ...     llm_provider="openai"
        ... )
    """
    provider_enum = LLMProvider(llm_provider.lower())

    return RAGChain(
        retriever=retriever,
        reranker=reranker,
        llm_provider=provider_enum,
        model_name=kwargs.get("model_name", settings.LLM_MODEL),
        temperature=kwargs.get("temperature", settings.LLM_TEMPERATURE),
        max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
    )


# Export public API
__all__ = [
    # Classes
    "RAGChain",
    "RAGResponse",
    "Citation",
    "Message",
    # Enums
    "LLMProvider",
    # Utilities
    "create_rag_chain",
]
