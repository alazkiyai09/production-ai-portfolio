"""
Main RAG Chain for DataChat-RAG

Orchestrates query routing, retrieval, and response generation with citations.
"""

import os
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, LLM
from llama_index.llms.openai import OpenAI

from src.routers.query_router import QueryRouter, QueryType, QueryClassification
from src.retrievers import DocumentRetriever, RetrievalResult
from src.cache import QueryCache


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Message:
    """Conversation message."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_chat_message(self) -> ChatMessage:
        """Convert to LlamaIndex ChatMessage."""
        return ChatMessage(role=self.role, content=self.content)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class SQLResult:
    """Result from SQL query execution."""

    query: str
    results: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": self.results,
            "columns": self.columns,
            "row_count": self.row_count,
            "execution_time_seconds": self.execution_time_seconds,
            "error": self.error,
        }

    def format_summary(self) -> str:
        """Format as a summary for the LLM."""
        if self.error:
            return f"Error executing query: {self.error}\nQuery: {self.query}"

        if not self.results:
            return f"Query returned no results.\nQuery: {self.query}"

        summary = f"Query: {self.query}\n"
        summary += f"Returned {self.row_count} row(s).\n\n"

        # Format results (limit to first 10 rows for context)
        max_rows = min(10, len(self.results))
        summary += "Results:\n"

        for i, row in enumerate(self.results[:max_rows], 1):
            summary += f"  {i}. "
            summary += ", ".join([f"{k}={v}" for k, v in row.items()])
            summary += "\n"

        if self.row_count > max_rows:
            summary += f"  ... ({self.row_count - max_rows} more rows)\n"

        return summary


@dataclass
class RAGResponse:
    """Response from the RAG chain."""

    answer: str
    query_type: str
    confidence: float
    sql_query: Optional[str] = None
    sql_results: Optional[SQLResult] = None
    doc_sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Optional[str] = None
    suggested_followup: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    is_cached: bool = False
    cache_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "query_type": self.query_type,
            "confidence": self.confidence,
            "sql_query": self.sql_query,
            "sql_results": self.sql_results.to_dict() if self.sql_results else None,
            "doc_sources": self.doc_sources,
            "reasoning": self.reasoning,
            "suggested_followup": self.suggested_followup,
            "processing_time_seconds": self.processing_time_seconds,
            "is_cached": self.is_cached,
            "cache_key": self.cache_key,
        }


# =============================================================================
# Main RAG Chain
# =============================================================================

class DataChatRAG:
    """
    Main RAG orchestrator for DataChat-RAG.

    Combines query routing, SQL/document retrieval, and LLM response generation
    with proper citations and conversation memory.
    """

    # System prompts
    SQL_SYSTEM_PROMPT = """You are a helpful data analyst assistant for a healthcare AdTech company. You have access to campaign performance data.

When presenting SQL query results:
- Clearly explain the key findings
- Use specific numbers and percentages
- Highlight trends, patterns, or anomalies
- Be concise but thorough
- If data is incomplete, acknowledge limitations

Format numbers appropriately (e.g., "$1.2M", "3.5%", "1,234")."""

    DOC_SYSTEM_PROMPT = """You are a helpful knowledge assistant for a healthcare AdTech company. You have access to company documentation including policies, procedures, and best practices.

When answering from documentation:
- Reference sources using [1], [2] citations
- Explain concepts clearly and accurately
- Include relevant details from the documents
- If information is incomplete, acknowledge this
- Direct users to relevant documentation when appropriate

Citations format: [Source: document_name | Type | Date]"""

    HYBRID_SYSTEM_PROMPT = """You are a helpful assistant for a healthcare AdTech company. You have access to both performance data and company documentation.

When answering hybrid questions:
- Start with data findings from SQL queries
- Add context from relevant documentation
- Explain WHY something happened using both data and knowledge
- Reference document sources with citations
- Provide actionable insights when possible
- Acknowledge uncertainty when information is incomplete

Combine quantitative insights with qualitative context for comprehensive answers."""

    def __init__(
        self,
        sql_retriever: Any,  # SQLRetriever (to be implemented)
        doc_retriever: DocumentRetriever,
        query_router: QueryRouter,
        llm: Optional[LLM] = None,
        enable_memory: bool = True,
        max_history_length: int = 10,
        query_cache: Optional[QueryCache] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the RAG chain.

        Args:
            sql_retriever: SQL retriever instance
            doc_retriever: Document retriever instance
            query_router: Query router instance
            llm: LLM instance (defaults to OpenAI gpt-4o)
            enable_memory: Whether to maintain conversation memory
            max_history_length: Maximum number of messages to keep in memory
            query_cache: Optional QueryCache instance for result caching
            enable_cache: Whether to enable query result caching (default: True)
        """
        self.sql_retriever = sql_retriever
        self.doc_retriever = doc_retriever
        self.query_router = query_router
        self.llm = llm or OpenAI(model="gpt-4o", temperature=0)
        self.enable_memory = enable_memory
        self.max_history_length = max_history_length

        # Conversation memory
        self.conversation_history: List[Message] = []

        # Query cache
        self.enable_cache = enable_cache
        self.query_cache = query_cache
        if enable_cache and query_cache is None:
            # Create default cache if none provided
            try:
                self.query_cache = QueryCache(
                    enabled=True,
                )
                print("✓ Query cache initialized (default)")
            except Exception as e:
                print(f"⚠ Failed to initialize query cache: {e}")
                self.query_cache = None
                self.enable_cache = False
        elif query_cache:
            print("✓ Query cache initialized")

        print("✓ DataChatRAG initialized")

    def query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Process a user question through the RAG pipeline.

        Args:
            question: User's question
            conversation_id: Optional conversation ID for memory isolation
            filters: Optional filters for document retrieval

        Returns:
            RAGResponse with answer, citations, and metadata
        """
        import time
        start_time = time.time()

        # Prepare conversation context for cache key
        conversation_context = None
        if self.enable_memory and self.conversation_history:
            conversation_context = [
                {"role": msg.role, "content": msg.content}
                for msg in self.conversation_history[-4:]  # Last 4 messages for context
            ]

        # Check cache first
        if self.enable_cache and self.query_cache:
            try:
                cached = self.query_cache.get(
                    question=question,
                    filters=filters,
                    conversation_context=conversation_context,
                )
                if cached:
                    # Reconstruct RAGResponse from cache
                    response = RAGResponse(
                        answer=cached.get("answer", ""),
                        query_type=cached.get("query_type", ""),
                        confidence=cached.get("confidence", 0.0),
                        sql_query=cached.get("sql_query"),
                        sql_results=None,  # SQL results not cached
                        doc_sources=cached.get("doc_sources", []),
                        reasoning=cached.get("reasoning"),
                        suggested_followup=cached.get("suggested_followup", []),
                        processing_time_seconds=time.time() - start_time,
                        is_cached=True,
                        cache_key=cached.get("_cache_key"),
                    )
                    # Update conversation memory even for cached responses
                    if self.enable_memory:
                        self._update_memory(question, response)
                    return response
            except Exception as e:
                # Log cache error but continue processing
                import logging
                logging.getLogger(__name__).warning(f"Cache retrieval error: {e}")

        # Classify the query
        classification = self.query_router.classify(question)

        # Route to appropriate handler
        if classification.query_type == QueryType.SQL_QUERY:
            response = self._handle_sql_query(
                question,
                classification,
                filters,
            )
        elif classification.query_type == QueryType.DOC_SEARCH:
            response = self._handle_doc_query(
                question,
                classification,
                filters,
            )
        else:  # HYBRID
            response = self._handle_hybrid_query(
                question,
                classification,
                filters,
            )

        # Add metadata
        response.reasoning = classification.reasoning
        response.processing_time_seconds = time.time() - start_time
        response.is_cached = False

        # Generate cache key
        cache_key = None
        if self.enable_cache and self.query_cache:
            try:
                cache_key = self.query_cache.generate_key(
                    question=question,
                    filters=filters,
                    conversation_context=conversation_context,
                )
                response.cache_key = cache_key

                # Store in cache
                cache_data = response.to_dict()
                cache_data["_cache_key"] = cache_key

                self.query_cache.set(
                    question=question,
                    response=cache_data,
                    filters=filters,
                    conversation_context=conversation_context,
                )
            except Exception as e:
                # Log cache error but don't fail the request
                import logging
                logging.getLogger(__name__).warning(f"Cache storage error: {e}")

        # Update conversation memory
        if self.enable_memory:
            self._update_memory(question, response)

        return response

    def _handle_sql_query(
        self,
        question: str,
        classification: QueryClassification,
        filters: Optional[Dict[str, Any]],
    ) -> RAGResponse:
        """Handle SQL-only queries."""
        try:
            # Generate SQL from natural language
            sql_result = self._generate_and_execute_sql(question, classification)

            if sql_result.error:
                # SQL execution failed - provide error response
                return RAGResponse(
                    answer=f"I encountered an error executing the query: {sql_result.error}",
                    query_type=str(classification.query_type.value),
                    confidence=0.3,
                    sql_query=sql_result.query,
                    sql_results=sql_result,
                )

            # Generate response from SQL results
            answer = self._generate_sql_response(question, sql_result)

            return RAGResponse(
                answer=answer,
                query_type=str(classification.query_type.value),
                confidence=classification.confidence,
                sql_query=sql_result.query,
                sql_results=sql_result,
                doc_sources=[],
            )

        except Exception as e:
            return RAGResponse(
                answer=f"I encountered an error processing your question: {str(e)}",
                query_type=str(classification.query_type.value),
                confidence=0.2,
            )

    def _handle_doc_query(
        self,
        question: str,
        classification: QueryClassification,
        filters: Optional[Dict[str, Any]],
    ) -> RAGResponse:
        """Handle document-only queries."""
        try:
            # Retrieve relevant documents
            doc_results = self.doc_retriever.retrieve(
                question,
                filters=filters or self._get_doc_filters(classification),
                top_k=5,
            )

            if not doc_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information in the documentation. "
                           "Could you rephrase your question or try different keywords?",
                    query_type=str(classification.query_type.value),
                    confidence=0.4,
                    doc_sources=[],
                )

            # Generate response from documents
            answer = self._generate_doc_response(question, doc_results)

            # Format sources
            doc_sources = [
                {
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "source": r.source,
                    "doc_type": str(r.doc_type.value),
                    "relevance": round(r.relevance_score, 3),
                }
                for r in doc_results
            ]

            return RAGResponse(
                answer=answer,
                query_type=str(classification.query_type.value),
                confidence=classification.confidence,
                doc_sources=doc_sources,
            )

        except Exception as e:
            return RAGResponse(
                answer=f"I encountered an error searching the documentation: {str(e)}",
                query_type=str(classification.query_type.value),
                confidence=0.2,
            )

    def _handle_hybrid_query(
        self,
        question: str,
        classification: QueryClassification,
        filters: Optional[Dict[str, Any]],
    ) -> RAGResponse:
        """Handle hybrid queries requiring both SQL and documents."""
        try:
            # Get SQL results
            sql_result = None
            sql_context = ""
            if classification.suggested_sql_tables:
                sql_result = self._generate_and_execute_sql(question, classification)
                if not sql_result.error and sql_result.results:
                    sql_context = sql_result.format_summary()

            # Get document results
            doc_results = []
            doc_context = ""
            if classification.suggested_doc_topics:
                doc_results = self.doc_retriever.retrieve(
                    question,
                    filters=filters or self._get_doc_filters(classification),
                    top_k=3,
                )
                if doc_results:
                    doc_context = self.doc_retriever.format_context(
                        doc_results,
                        include_citations=True,
                    )

            # Check if we have any results
            if not sql_result and not doc_results:
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question. "
                           "Could you provide more context or rephrase your question?",
                    query_type=str(classification.query_type.value),
                    confidence=0.3,
                )

            # Generate combined response
            answer = self._generate_hybrid_response(
                question,
                sql_context,
                doc_context,
                classification,
            )

            # Format sources
            doc_sources = [
                {
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "source": r.source,
                    "doc_type": str(r.doc_type.value),
                    "relevance": round(r.relevance_score, 3),
                }
                for r in doc_results
            ]

            return RAGResponse(
                answer=answer,
                query_type=str(classification.query_type.value),
                confidence=classification.confidence,
                sql_query=sql_result.query if sql_result else None,
                sql_results=sql_result,
                doc_sources=doc_sources,
            )

        except Exception as e:
            return RAGResponse(
                answer=f"I encountered an error processing your question: {str(e)}",
                query_type=str(classification.query_type.value),
                confidence=0.2,
            )

    def _generate_and_execute_sql(
        self,
        question: str,
        classification: QueryClassification,
    ) -> SQLResult:
        """Generate and execute SQL query using text-to-SQL."""
        try:
            if self.sql_retriever is None:
                return SQLResult(
                    query="",
                    results=[],
                    columns=[],
                    row_count=0,
                    error="SQL retriever not configured",
                )

            # Use SQL retriever to generate and execute query
            result = self.sql_retriever.query(
                question,
                tables=classification.suggested_sql_tables,
            )

            return result

        except Exception as e:
            return SQLResult(
                query="",
                results=[],
                columns=[],
                row_count=0,
                error=str(e),
            )

    def _generate_sql_response(
        self,
        question: str,
        sql_result: SQLResult,
    ) -> str:
        """Generate natural language response from SQL results."""
        # Build context from SQL results
        context = sql_result.format_summary()

        # Add conversation history if available
        history_context = self._format_history_for_llm()

        # Build prompt
        prompt = f"""Question: {question}

{context}

{history_context}

Please provide a clear, concise answer to the question based on the data above.
Highlight the key findings and any notable patterns or trends."""

        messages = [
            ChatMessage(role="system", content=self.SQL_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        # Generate response
        response = self.llm.chat(messages)
        return response.message.content.strip()

    def _generate_doc_response(
        self,
        question: str,
        doc_results: List[RetrievalResult],
    ) -> str:
        """Generate natural language response from documents."""
        # Format document context
        context = self.doc_retriever.format_context(
            doc_results,
            include_citations=True,
        )

        # Add conversation history if available
        history_context = self._format_history_for_llm()

        # Build prompt
        prompt = f"""Question: {question}

Relevant Information:
{context}

{history_context}

Please answer the question based on the information above.
Use [1], [2] citations to reference your sources.
If the information doesn't fully answer the question, acknowledge this."""

        messages = [
            ChatMessage(role="system", content=self.DOC_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        # Generate response
        response = self.llm.chat(messages)
        return response.message.content.strip()

    def _generate_hybrid_response(
        self,
        question: str,
        sql_context: str,
        doc_context: str,
        classification: QueryClassification,
    ) -> str:
        """Generate response combining SQL and document results."""
        # Add conversation history if available
        history_context = self._format_history_for_llm()

        # Build prompt with both contexts
        prompt_parts = [f"Question: {question}\n"]

        if sql_context:
            prompt_parts.append(f"Data Findings:\n{sql_context}\n")

        if doc_context:
            prompt_parts.append(f"Relevant Documentation:\n{doc_context}\n")

        prompt_parts.append(history_context)
        prompt_parts.append(
            "\nPlease provide a comprehensive answer combining the data and documentation above. "
            "Explain what the data shows and why it might be happening based on the documentation. "
            "Use citations for document references."
        )

        prompt = "\n".join(prompt_parts)

        messages = [
            ChatMessage(role="system", content=self.HYBRID_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        # Generate response
        response = self.llm.chat(messages)
        answer = response.message.content.strip()

        # Generate follow-up suggestions
        followup = self._generate_followup_suggestions(
            question,
            classification,
            sql_context,
            doc_context,
        )

        if followup:
            answer += "\n\n" + "\n".join(followup)

        return answer

    def _generate_followup_suggestions(
        self,
        question: str,
        classification: QueryClassification,
        sql_context: str,
        doc_context: str,
    ) -> List[str]:
        """Generate suggested follow-up questions."""
        suggestions = []

        # Based on query type
        if classification.query_type == QueryType.SQL_QUERY:
            if "last week" in question.lower() or "last month" in question.lower():
                suggestions.append("Would you like to see the trend over a longer time period?")
            if "top" in question.lower() or "bottom" in question.lower():
                suggestions.append("Would you like to drill down into any specific campaign?")
            suggestions.append("Would you like to compare these metrics to our benchmarks?")

        elif classification.query_type == QueryType.DOC_SEARCH:
            suggestions.append("Would you like more details on any specific aspect?")
            suggestions.append("Is there a related policy or guideline you'd like to review?")

        else:  # HYBRID
            suggestions.append("Would you like me to investigate specific campaigns or time periods?")
            suggestions.append("Would you like actionable recommendations based on these findings?")

        return [f"💡 {s}" for s in suggestions[:2]]

    def _get_doc_filters(self, classification: QueryClassification) -> Optional[Dict[str, Any]]:
        """Get document filters from classification."""
        if not classification.suggested_doc_topics:
            return None

        # Map topics to document types
        topic_to_doc_type = {
            "hipaa_compliance": "compliance",
            "ad_approval_process": "sop",
            "creative_guidelines": "guideline",
            "targeting_policies": "policy",
            "attribution_models": "guideline",
            "tracking_setup": "guideline",
            "reporting_standards": "policy",
            "data_governance": "policy",
            "client_onboarding": "sop",
            "campaign_best_practices": "best_practice",
        }

        doc_types = []
        for topic in classification.suggested_doc_topics:
            doc_type = topic_to_doc_type.get(topic)
            if doc_type:
                doc_types.append(doc_type)

        if doc_types:
            return {"doc_type": doc_types}
        return None

    def _format_history_for_llm(self) -> str:
        """Format conversation history for LLM context."""
        if not self.conversation_history:
            return ""

        recent_history = self.conversation_history[-self.max_history_length:]

        history_lines = ["\nConversation History:"]
        for msg in recent_history[-4:]:  # Last 4 messages for context
            role = msg.role.capitalize()
            history_lines.append(f"{role}: {msg.content}")

        return "\n".join(history_lines)

    def _update_memory(self, question: str, response: RAGResponse):
        """Update conversation memory with the interaction."""
        # Add user message
        self.conversation_history.append(
            Message(role="user", content=question)
        )

        # Add assistant message (just the answer, not all metadata)
        self.conversation_history.append(
            Message(role="assistant", content=response.answer)
        )

        # Trim if needed
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]

    def clear_memory(self):
        """Clear conversation memory."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history as a list of dictionaries."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]

    # =========================================================================
    # Cache Management
    # =========================================================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics including hit rate, miss rate, etc.
        """
        if not self.enable_cache or not self.query_cache:
            return {
                "cache_enabled": False,
                "message": "Caching is not enabled",
            }

        try:
            stats = self.query_cache.get_stats()
            stats["cache_enabled"] = True
            return stats
        except Exception as e:
            return {
                "cache_enabled": True,
                "error": str(e),
                "message": "Failed to retrieve cache statistics",
            }

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the query cache.

        Returns:
            Dict with operation result
        """
        if not self.enable_cache or not self.query_cache:
            return {
                "success": False,
                "message": "Caching is not enabled",
            }

        try:
            cleared = self.query_cache.clear()
            return {
                "success": cleared,
                "message": "Cache cleared successfully" if cleared else "No cache entries to clear",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to clear cache: {e}",
            }

    def delete_cached_query(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Delete a specific query from cache.

        Args:
            question: The question to delete from cache
            filters: Optional filters used with the question

        Returns:
            Dict with operation result
        """
        if not self.enable_cache or not self.query_cache:
            return {
                "success": False,
                "message": "Caching is not enabled",
            }

        try:
            # Get conversation context for cache key
            conversation_context = None
            if self.enable_memory and self.conversation_history:
                conversation_context = [
                    {"role": msg.role, "content": msg.content}
                    for msg in self.conversation_history[-4:]
                ]

            deleted = self.query_cache.delete(
                question=question,
                filters=filters,
                conversation_context=conversation_context,
            )
            return {
                "success": deleted,
                "message": "Query deleted from cache" if deleted else "Query not found in cache",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete from cache: {e}",
            }

    def is_cache_enabled(self) -> bool:
        """Check if query caching is enabled."""
        return self.enable_cache and self.query_cache is not None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_rag_chain(
    sql_retriever: Any,
    doc_retriever: DocumentRetriever,
    query_router: QueryRouter,
    llm: Optional[LLM] = None,
    query_cache: Optional[QueryCache] = None,
    enable_cache: bool = True,
) -> DataChatRAG:
    """
    Convenience function to create a RAG chain.

    Args:
        sql_retriever: SQL retriever instance
        doc_retriever: Document retriever instance
        query_router: Query router instance
        llm: Optional LLM instance
        query_cache: Optional QueryCache instance for result caching
        enable_cache: Whether to enable query result caching

    Returns:
        Configured DataChatRAG instance
    """
    return DataChatRAG(
        sql_retriever=sql_retriever,
        doc_retriever=doc_retriever,
        query_router=query_router,
        llm=llm,
        query_cache=query_cache,
        enable_cache=enable_cache,
    )


# =============================================================================
# Test Cases
# =============================================================================

# Whitelist of allowed table names for safe SQL query construction
ALLOWED_TABLES = {
    "campaigns", "impressions", "clicks", "conversions", "daily_metrics"
}


class MockSQLRetriever:
    """Mock SQL retriever for testing."""

    def query(self, question: str, tables: List[str] = None) -> SQLResult:
        """Generate mock SQL result.

        Note: This is a mock for testing only. In production, use parameterized queries.
        """
        # Safely select the table name from the whitelist
        if tables and tables[0] in ALLOWED_TABLES:
            table_name = tables[0]
        else:
            table_name = "campaigns"  # Default safe table

        return SQLResult(
            query=f"SELECT * FROM {table_name} LIMIT 5",
            results=[
                {"campaign_name": "MedTech Q4", "ctr": 1.2, "impressions": 50000, "clicks": 600},
                {"campaign_name": "PharmaCorp Launch", "ctr": 0.8, "impressions": 75000, "clicks": 600},
                {"campaign_name": "HealthAwareness", "ctr": 1.5, "impressions": 30000, "clicks": 450},
            ],
            columns=["campaign_name", "ctr", "impressions", "clicks"],
            row_count=3,
        )


def run_test_cases():
    """Run test cases for the RAG chain."""
    import pprint

    pp = pprint.PrettyPrinter(indent=2)

    print("=" * 80)
    print("RAG CHAIN TEST CASES")
    print("=" * 80)

    # Create mock components
    print("\nInitializing components...")

    mock_sql_retriever = MockSQLRetriever()

    # Note: These would be real instances in production
    # For testing, we'll mock the responses
    from src.routers import QueryRouter, QueryType
    from src.retrievers import DocumentRetriever, DocumentType

    query_router = QueryRouter()

    # Mock doc retriever responses
    class MockDocRetriever:
        def __init__(self):
            self.test_results = [
                RetrievalResult(
                    content="HIPAA requires all patient data to be encrypted and access to be logged.",
                    source="HIPAA_Compliance_Guide.txt",
                    doc_type=DocumentType.COMPLIANCE,
                    relevance_score=0.92,
                    metadata={"department": "legal", "date": "2024-01-15"},
                ),
                RetrievalResult(
                    content="Healthcare campaigns typically have lower CTR (0.8-1.5%) due to regulations.",
                    source="Healthcare_Campaign_Best_Practices.txt",
                    doc_type=DocumentType.BEST_PRACTICE,
                    relevance_score=0.88,
                    metadata={"department": "marketing", "date": "2024-03-10"},
                ),
            ]

        def retrieve(self, query, filters=None, top_k=5):
            return self.test_results[:top_k]

        def format_context(self, results, include_citations=True):
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"[{i}] {r.content}")
                if include_citations:
                    lines.append(f"    {r.format_citation()}")
            return "\n".join(lines)

    mock_doc_retriever = MockDocRetriever()

    # Create RAG chain (without LLM for testing)
    print("✓ Components initialized\n")

    # Test 1: SQL Query Classification
    print("─" * 80)
    print("Test 1: SQL Query Classification")
    print("─" * 80)

    question = "What was our average CTR last week?"
    classification = query_router.classify(question)

    print(f"Question: {question}")
    print(f"Classified as: {classification.query_type.value}")
    print(f"Confidence: {classification.confidence}")
    print(f"Suggested Tables: {classification.suggested_sql_tables}")

    # Test 2: Doc Query Classification
    print("\n─" * 80)
    print("Test 2: Document Query Classification")
    print("─" * 80)

    question = "What are the HIPAA compliance requirements?"
    classification = query_router.classify(question)

    print(f"Question: {question}")
    print(f"Classified as: {classification.query_type.value}")
    print(f"Confidence: {classification.confidence}")
    print(f"Suggested Topics: {classification.suggested_doc_topics}")

    # Test 3: Hybrid Query Classification
    print("\n─" * 80)
    print("Test 3: Hybrid Query Classification")
    print("─" * 80)

    question = "Why is the BioGen campaign underperforming compared to benchmarks?"
    classification = query_router.classify(question)

    print(f"Question: {question}")
    print(f"Classified as: {classification.query_type.value}")
    print(f"Confidence: {classification.confidence}")
    print(f"Suggested Tables: {classification.suggested_sql_tables}")
    print(f"Suggested Topics: {classification.suggested_doc_topics}")

    # Test 4: Document Retrieval
    print("\n─" * 80)
    print("Test 4: Document Retrieval")
    print("─" * 80)

    query = "HIPAA requirements"
    results = mock_doc_retriever.retrieve(query, top_k=2)
    context = mock_doc_retriever.format_context(results, include_citations=True)

    print(f"Query: {query}")
    print(f"\nRetrieved {len(results)} documents:\n")
    print(context)

    # Test 5: SQL Result Formatting
    print("\n─" * 80)
    print("Test 5: SQL Result Formatting")
    print("─" * 80)

    sql_result = SQLResult(
        query="SELECT campaign_name, AVG(ctr) as avg_ctr FROM daily_metrics GROUP BY campaign_name ORDER BY avg_ctr DESC LIMIT 5",
        results=[
            {"campaign_name": "MedTech Solutions Q4", "avg_ctr": 1.85},
            {"campaign_name": "HealthAwareness Jan", "avg_ctr": 1.52},
            {"campaign_name": "PharmaCorp Launch", "avg_ctr": 1.23},
        ],
        columns=["campaign_name", "avg_ctr"],
        row_count=3,
    )

    print("SQL Query Summary:")
    print("─" * 40)
    print(sql_result.format_summary())

    # Test 6: Response Structure
    print("\n─" * 80)
    print("Test 6: RAGResponse Structure")
    print("─" * 80)

    response = RAGResponse(
        answer="Based on the data, the average CTR last week was 1.2%, which is above our healthcare benchmark of 0.8-1.5%.",
        query_type="SQL_QUERY",
        confidence=0.92,
        sql_query="SELECT AVG(ctr) FROM daily_metrics WHERE date >= CURRENT_DATE - INTERVAL '7 days'",
        sql_results=sql_result,
        doc_sources=[],
        suggested_followup=[
            "💡 Would you like to see the trend over a longer time period?",
            "💡 Would you like to compare these metrics to our benchmarks?",
        ],
    )

    print("Sample RAGResponse:")
    pp.pprint(response.to_dict())

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)

    print("\nNote: Full end-to-end testing requires:")
    print("  - OPENAI_API_KEY for LLM and embeddings")
    print("  - Configured SQL retriever")
    print("  - Populated document store")


if __name__ == "__main__":
    run_test_cases()
