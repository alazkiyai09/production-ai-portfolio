"""
Intelligent Query Router for DataChat-RAG

Classifies incoming questions into SQL_QUERY, DOC_SEARCH, or HYBRID types
using LLM-based classification with confidence scoring.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.llms import ChatMessage, LLM
from llama_index.llms.openai import OpenAI


class QueryType(Enum):
    """Query classification types."""

    SQL_QUERY = "SQL_QUERY"
    DOC_SEARCH = "DOC_SEARCH"
    HYBRID = "HYBRID"

    def __str__(self) -> str:
        return self.value


@dataclass
class QueryClassification:
    """Result of query classification."""

    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggested_sql_tables: List[str] = field(default_factory=list)
    suggested_doc_topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    raw_llm_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_type": str(self.query_type),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_sql_tables": self.suggested_sql_tables,
            "suggested_doc_topics": self.suggested_doc_topics,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryClassification":
        """Create from dictionary."""
        return cls(
            query_type=QueryType(data["query_type"]),
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            suggested_sql_tables=data.get("suggested_sql_tables", []),
            suggested_doc_topics=data.get("suggested_doc_topics", []),
            keywords=data.get("keywords", []),
        )


# SQL-related keywords
SQL_KEYWORDS = [
    # Metrics and KPIs
    "ctr", "cvr", "cpa", "cpc", "cpm", "roi", "roas",
    "click-through rate", "conversion rate", "cost per acquisition",
    "impressions", "clicks", "conversions", "spend", "budget",
    "performance", "metrics", "analytics", "statistics",

    # Time-based queries
    "today", "yesterday", "last week", "last month", "this quarter",
    "year to date", "ytd", "last 7 days", "last 30 days",

    # Comparison and ranking
    "top", "bottom", "highest", "lowest", "best", "worst",
    "compared to", "vs", "versus", "ranking", "sorted by",

    # Aggregation
    "total", "average", "sum", "count", "median", "maximum", "minimum",
    "how many", "how much", "what was the", "show me",

    # Campaign and data entities
    "campaign", "campaigns", "client", "industry", "device type",
    "geo location", "ad placement", "table", "database",

    # Numbers and measurements
    "number of", "count of", "percentage", "rate", "ratio",
]

# Document-related keywords
DOC_KEYWORDS = [
    # Policies and guidelines
    "policy", "policies", "guideline", "guidelines", "standard", "standards",
    "procedure", "procedures", "process", "processes", "protocol", "protocols",

    # Compliance and regulations
    "hipaa", "compliance", "compliant", "regulation", "regulatory",
    "fda", "legal", "requirement", "requirements",

    # Knowledge and information
    "how do i", "how to", "what is the", "explain", "describe",
    "documentation", "docs", "guide", "handbook", "manual",

    # Company-specific
    "approval process", "workflow", "onboarding", "training",
    "best practices", "playbook", "sop", "standard operating procedure",

    # AdTech specific
    "ad approval", "creative guidelines", "targeting rules",
    "attribution model", "tracking", "pixel", "tag",
]

# Hybrid indicators
HYBRID_KEYWORDS = [
    "why", "reason", "cause", "explain why", "insight", "analysis",
    "underperforming", "overperforming", "unexpected", "unusual",
    "benchmark", "compared to our", "against our", "trend",
    "what happened", "what changed", "diagnose", "investigate",
    "context", "background", "understand the", "tell me about",
]


class QueryRouter:
    """
    Intelligent router that classifies queries using LLM analysis.

    Uses few-shot prompting with examples to classify queries into
    SQL_QUERY, DOC_SEARCH, or HYBRID types with confidence scoring.
    """

    # SQL schema information for the router
    SQL_TABLES = [
        "campaigns",
        "impressions",
        "clicks",
        "conversions",
        "daily_metrics",
    ]

    # Document topics for healthcare AdTech
    DOC_TOPICS = [
        "hipaa_compliance",
        "ad_approval_process",
        "creative_guidelines",
        "targeting_policies",
        "attribution_models",
        "tracking_setup",
        "reporting_standards",
        "data_governance",
        "client_onboarding",
        "campaign_best_practices",
        "healthcare_advertising_regulations",
        "pharma_marketing_guidelines",
        "medical_device_promotions",
    ]

    def __init__(self, llm: Optional[LLM] = None):
        """
        Initialize the query router.

        Args:
            llm: LLM instance for classification. Defaults to OpenAI gpt-4o.
        """
        self.llm = llm or OpenAI(model="gpt-4o", temperature=0)
        self._classification_cache: Dict[str, QueryClassification] = {}

    def get_routing_prompt(self) -> str:
        """
        Get the few-shot prompt for query classification.

        Returns:
            The system prompt with examples for classification.
        """
        return """You are a query classifier for a healthcare AdTech data platform. Your job is to determine whether a user question requires:

1. **SQL_QUERY**: Questions about metrics, numbers, campaign performance, or any quantitative data from the database.
2. **DOC_SEARCH**: Questions about company policies, procedures, guidelines, compliance, or documented knowledge.
3. **HYBRID**: Questions that need BOTH data AND contextual knowledge (insights, explanations, benchmarks).

---

## Available SQL Tables:
- campaigns (campaign metadata, budgets, clients, industries)
- impressions (ad impressions with device, geo, placement, cost)
- clicks (click events with timestamps, landing pages)
- conversions (conversion events with types and values)
- daily_metrics (aggregated metrics: impressions, clicks, conversions, spend, ctr, cvr, cpa, cpc, cpm)

## Available Document Topics:
- HIPAA compliance
- Ad approval process
- Creative guidelines
- Targeting policies
- Attribution models
- Tracking setup
- Reporting standards
- Data governance
- Client onboarding
- Campaign best practices
- Healthcare advertising regulations
- Pharma marketing guidelines
- Medical device promotions

---

## Classification Rules:

**SQL_QUERY** if the question asks for:
- Specific metrics or numbers (CTR, CVR, CPA, spend, impressions, etc.)
- Rankings, top/bottom lists, comparisons
- Time-based aggregations (last week, this month, YTD, etc.)
- Campaign, client, or performance data
- "How many", "how much", "what was the", "show me"

**DOC_SEARCH** if the question asks for:
- Policies, procedures, guidelines, or SOPs
- How to do something, process explanations
- Compliance or regulatory information
- Documentation, manuals, playbooks
- "How do I", "what is the policy", "explain the process"

**HYBRID** if the question asks for:
- Why something happened (causal analysis)
- Insights combining data with context
- Benchmarking against standards/best practices
- Unexpected performance or anomalies
- "Why is", "what caused", "explain why", "compared to benchmarks"

---

## Examples:

### Example 1:
Query: "What was our average CTR last week?"
Classification: SQL_QUERY
Confidence: 0.95
Reasoning: The question asks for a specific metric (CTR) aggregated over a time period (last week), which requires querying the daily_metrics table.
Tables: ["daily_metrics"]
Topics: []

### Example 2:
Query: "What's the process for getting a healthcare ad approved?"
Classification: DOC_SEARCH
Confidence: 0.98
Reasoning: The question asks about a process/workflow (ad approval), which is documented knowledge rather than quantitative data.
Tables: []
Topics: ["ad_approval_process"]

### Example 3:
Query: "Top 5 campaigns by spend this month"
Classification: SQL_QUERY
Confidence: 0.97
Reasoning: The question requests a ranking of campaigns based on a numeric metric (spend), requiring aggregation from the campaigns and daily_metrics tables.
Tables: ["campaigns", "daily_metrics"]
Topics: []

### Example 4:
Query: "HIPAA requirements for patient data in ads"
Classification: DOC_SEARCH
Confidence: 0.99
Reasoning: The question asks about compliance requirements (HIPAA), which is regulatory documentation.
Tables: []
Topics: ["hipaa_compliance"]

### Example 5:
Query: "Why is the BioGen campaign underperforming compared to our benchmarks?"
Classification: HYBRID
Confidence: 0.94
Reasoning: The question asks for causal analysis ("why") comparing actual performance to benchmarks, requiring both query data AND knowledge of industry benchmarks/best practices.
Tables: ["campaigns", "daily_metrics"]
Topics: ["campaign_best_practices"]

### Example 6:
Query: "How many conversions did we get from mobile devices yesterday?"
Classification: SQL_QUERY
Confidence: 0.96
Reasoning: The question asks for a count (conversions) filtered by specific criteria (mobile, yesterday), which is a database query.
Tables: ["conversions", "impressions"]
Topics: []

### Example 7:
Query: "Explain our attribution model for healthcare campaigns"
Classification: DOC_SEARCH
Confidence: 0.92
Reasoning: The question asks for an explanation of a documented concept (attribution model), which is knowledge retrieval.
Tables: []
Topics: ["attribution_models"]

### Example 8:
Query: "What caused the spike in CPC last Tuesday?"
Classification: HYBRID
Confidence: 0.91
Reasoning: The question asks for causal explanation ("what caused") about a metric anomaly, requiring both the data to identify the spike AND contextual knowledge to explain possible causes.
Tables: ["daily_metrics", "impressions"]
Topics: ["campaign_best_practices"]

### Example 9:
Query: "Show me the trending campaigns by impressions"
Classification: SQL_QUERY
Confidence: 0.93
Reasoning: The question requests campaign ranking by a numeric metric (impressions), which is a straightforward database query.
Tables: ["campaigns", "daily_metrics"]
Topics: []

### Example 10:
Query: "What are our targeting policies for pharmaceutical ads?"
Classification: DOC_SEARCH
Confidence: 0.97
Reasoning: The question asks about documented policies (targeting policies for pharma), which is knowledge retrieval.
Tables: []
Topics: ["targeting_policies", "pharma_marketing_guidelines"]

### Example 11:
Query: "How do I set up conversion tracking for a medical device client?"
Classification: DOC_SEARCH
Confidence: 0.95
Reasoning: The question asks "how do I" about a documented process (setting up tracking), which requires procedural knowledge.
Tables: []
Topics: ["tracking_setup", "medical_device_promotions"]

### Example 12:
Query: "Why is CPA higher for mobile compared to desktop?"
Classification: HYBRID
Confidence: 0.93
Reasoning: The question asks for causal analysis ("why is") comparing metrics across segments, requiring both query data AND knowledge about device performance patterns.
Tables: ["daily_metrics", "impressions"]
Topics: ["campaign_best_practices"]

---

## Your Task:

Analyze the user query and respond ONLY with valid JSON in this exact format:

```json
{
  "query_type": "SQL_QUERY" | "DOC_SEARCH" | "HYBRID",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of the classification",
  "suggested_sql_tables": ["table1", "table2"],
  "suggested_doc_topics": ["topic1", "topic2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}
```

Important:
- Be confident (0.8+) when the classification is clear
- Use lower confidence (0.5-0.7) for ambiguous queries
- suggested_sql_tables should only include relevant tables from the available list
- suggested_doc_topics should only include relevant topics from the available list
- keywords should be the key terms that influenced your decision (3-5 max)
- Respond with JSON ONLY, no additional text
"""

    def classify(self, query: str, use_cache: bool = True) -> QueryClassification:
        """
        Classify a query into SQL_QUERY, DOC_SEARCH, or HYBRID.

        Args:
            query: The user's question or query string
            use_cache: Whether to use cached classifications (default: True)

        Returns:
            QueryClassification with type, confidence, reasoning, and suggestions

        Raises:
            ValueError: If the query is empty or classification fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Normalize query
        query = query.strip()

        # Check cache
        if use_cache and query in self._classification_cache:
            return self._classification_cache[query]

        try:
            # Get classification from LLM
            classification = self._classify_with_llm(query)

            # Cache the result
            if use_cache:
                self._classification_cache[query] = classification

            return classification

        except Exception as e:
            # Fallback to keyword-based classification
            return self._classify_with_keywords(query)

    def _classify_with_llm(self, query: str) -> QueryClassification:
        """Use LLM to classify the query."""
        # Prepare messages
        messages = [
            ChatMessage(role="system", content=self.get_routing_prompt()),
            ChatMessage(role="user", content=f"Classify this query: {query}"),
        ]

        # Get response from LLM
        response = self.llm.chat(messages)
        response_text = response.message.content.strip()

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                return QueryClassification(
                    query_type=QueryType(data["query_type"]),
                    confidence=float(data["confidence"]),
                    reasoning=data["reasoning"],
                    suggested_sql_tables=data.get("suggested_sql_tables", []),
                    suggested_doc_topics=data.get("suggested_doc_topics", []),
                    keywords=data.get("keywords", []),
                    raw_llm_response=response_text,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # If JSON parsing fails, log and fall back to keyword-based
                print(f"Warning: Failed to parse LLM response as JSON: {e}")
                return self._classify_with_keywords(query)
        else:
            # No JSON found, fall back to keyword-based
            return self._classify_with_keywords(query)

    def _classify_with_keywords(self, query: str) -> QueryClassification:
        """
        Fallback: Classify using keyword matching.

        This is used when LLM classification fails or as a backup.
        """
        query_lower = query.lower()

        # Count keyword matches
        sql_matches = sum(1 for kw in SQL_KEYWORDS if kw in query_lower)
        doc_matches = sum(1 for kw in DOC_KEYWORDS if kw in query_lower)
        hybrid_matches = sum(1 for kw in HYBRID_KEYWORDS if kw in query_lower)

        # Extract matched keywords
        matched_keywords = []
        if sql_matches > 0:
            matched_keywords.extend([kw for kw in SQL_KEYWORDS if kw in query_lower][:3])
        if doc_matches > 0:
            matched_keywords.extend([kw for kw in DOC_KEYWORDS if kw in query_lower][:3])
        if hybrid_matches > 0:
            matched_keywords.extend([kw for kw in HYBRID_KEYWORDS if kw in query_lower][:3])

        # Determine type based on keyword counts
        if hybrid_matches > 0:
            # Hybrid questions have specific "why/analysis" indicators
            query_type = QueryType.HYBRID
            confidence = min(0.85, 0.5 + hybrid_matches * 0.1)
            reasoning = f"Contains hybrid analysis keywords: {', '.join([kw for kw in HYBRID_KEYWORDS if kw in query_lower])}"
        elif sql_matches >= doc_matches:
            query_type = QueryType.SQL_QUERY
            confidence = min(0.90, 0.5 + sql_matches * 0.05)
            reasoning = f"Contains {sql_matches} SQL-related keywords"
        else:
            query_type = QueryType.DOC_SEARCH
            confidence = min(0.90, 0.5 + doc_matches * 0.05)
            reasoning = f"Contains {doc_matches} document-related keywords"

        # Suggest tables/topics based on keywords
        suggested_tables = []
        suggested_topics = []

        if query_type in [QueryType.SQL_QUERY, QueryType.HYBRID]:
            # Always include campaigns and daily_metrics for SQL queries
            suggested_tables = ["campaigns", "daily_metrics"]
            if any(kw in query_lower for kw in ["click", "ctr", "cpc"]):
                suggested_tables.append("clicks")
            if any(kw in query_lower for kw in ["conversion", "cvr", "cpa"]):
                suggested_tables.append("conversions")
            if any(kw in query_lower for kw in ["impression", "cpm", "device", "geo"]):
                suggested_tables.append("impressions")

        if query_type in [QueryType.DOC_SEARCH, QueryType.HYBRID]:
            if any(kw in query_lower for kw in ["hipaa", "compliance", "patient", "data"]):
                suggested_topics.append("hipaa_compliance")
            if any(kw in query_lower for kw in ["approval", "review", "creative"]):
                suggested_topics.append("ad_approval_process")
            if any(kw in query_lower for kw in ["targeting", "audience"]):
                suggested_topics.append("targeting_policies")
            if any(kw in query_lower for kw in ["attribution", "tracking", "pixel"]):
                suggested_topics.append("attribution_models")
            if any(kw in query_lower for kw in ["benchmark", "best practice"]):
                suggested_topics.append("campaign_best_practices")

        return QueryClassification(
            query_type=query_type,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            suggested_sql_tables=suggested_tables,
            suggested_doc_topics=suggested_topics,
            keywords=list(set(matched_keywords))[:5],
        )

    def clear_cache(self):
        """Clear the classification cache."""
        self._classification_cache.clear()

    def batch_classify(self, queries: List[str]) -> List[QueryClassification]:
        """
        Classify multiple queries in batch.

        Args:
            queries: List of query strings

        Returns:
            List of QueryClassification results
        """
        return [self.classify(query) for query in queries]


# =============================================================================
# Convenience Functions
# =============================================================================

def classify_query(query: str, llm: Optional[LLM] = None) -> QueryClassification:
    """
    Convenience function to classify a single query.

    Args:
        query: The user's question
        llm: Optional LLM instance

    Returns:
        QueryClassification result
    """
    router = QueryRouter(llm)
    return router.classify(query)


# =============================================================================
# Test Cases
# =============================================================================

def run_test_cases():
    """Run test cases to demonstrate query classification."""
    import pprint

    pp = pprint.PrettyPrinter(indent=2)

    test_queries = [
        # SQL queries
        ("What was our average CTR last week?", QueryType.SQL_QUERY),
        ("Top 5 campaigns by spend this month", QueryType.SQL_QUERY),
        ("How many conversions did we get from mobile devices yesterday?", QueryType.SQL_QUERY),
        ("Show me the trending campaigns by impressions", QueryType.SQL_QUERY),
        ("What's the total spend for pharma campaigns this quarter?", QueryType.SQL_QUERY),

        # Document searches
        ("What's the process for getting a healthcare ad approved?", QueryType.DOC_SEARCH),
        ("HIPAA requirements for patient data in ads", QueryType.DOC_SEARCH),
        ("Explain our attribution model for healthcare campaigns", QueryType.DOC_SEARCH),
        ("What are our targeting policies for pharmaceutical ads?", QueryType.DOC_SEARCH),
        ("How do I set up conversion tracking for a medical device client?", QueryType.DOC_SEARCH),

        # Hybrid queries
        ("Why is the BioGen campaign underperforming compared to our benchmarks?", QueryType.HYBRID),
        ("What caused the spike in CPC last Tuesday?", QueryType.HYBRID),
        ("Why is CPA higher for mobile compared to desktop?", QueryType.HYBRID),
        ("Explain why MedTech Solutions campaign has lower CVR than industry average", QueryType.HYBRID),
        ("What's driving the performance difference between our healthcare and pharma campaigns?", QueryType.HYBRID),
    ]

    print("=" * 80)
    print("QUERY ROUTER TEST CASES")
    print("=" * 80)

    # Create router
    router = QueryRouter()

    results = []

    for query, expected_type in test_queries:
        print(f"\n{'─' * 80}")
        print(f"Query: {query}")
        print(f"Expected: {expected_type.value}")
        print(f"{'─' * 80}")

        classification = router.classify(query)
        results.append((query, expected_type, classification))

        # Check if classification matches expected
        match = "✓" if classification.query_type == expected_type else "✗"

        print(f"\nResult: {match}")
        print(f"  Type:       {classification.query_type.value}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Reasoning:  {classification.reasoning}")
        if classification.suggested_sql_tables:
            print(f"  Tables:     {', '.join(classification.suggested_sql_tables)}")
        if classification.suggested_doc_topics:
            print(f"  Topics:     {', '.join(classification.suggested_doc_topics)}")
        if classification.keywords:
            print(f"  Keywords:   {', '.join(classification.keywords)}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    correct = sum(1 for _, expected, result in results if result.query_type == expected)
    total = len(results)
    accuracy = correct / total * 100

    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

    # Average confidence by type
    sql_conf = [r.confidence for _, e, r in results if e == QueryType.SQL_QUERY]
    doc_conf = [r.confidence for _, e, r in results if e == QueryType.DOC_SEARCH]
    hybrid_conf = [r.confidence for _, e, r in results if e == QueryType.HYBRID]

    print(f"\nAverage Confidence by Type:")
    print(f"  SQL_QUERY:   {sum(sql_conf)/len(sql_conf):.2f}" if sql_conf else "  SQL_QUERY:   N/A")
    print(f"  DOC_SEARCH:  {sum(doc_conf)/len(doc_conf):.2f}" if doc_conf else "  DOC_SEARCH:  N/A")
    print(f"  HYBRID:      {sum(hybrid_conf)/len(hybrid_conf):.2f}" if hybrid_conf else "  HYBRID:      N/A")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    # Run test cases when executed directly
    run_test_cases()
