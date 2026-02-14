"""
Query Router - Intelligent Query Classification and Routing
============================================================

THE BRAIN OF THE SYSTEM.

Routes queries to the appropriate service:
- Analytics queries → Analytics Service (fast, cheap)
- Search queries → Search only (medium cost)
- Insight queries → RAG Service (slow, expensive)

This is CRITICAL for cost optimization and user experience.

System Design Philosophy:
------------------------
NOT ALL QUERIES ARE EQUAL.

Query Types:
1. "How many calls?" → Analytics (0 LLM calls, <100ms, $0)
2. "Find calls about X" → Search (0 LLM calls, <500ms, $0.001)
3. "What are customers saying about X?" → RAG (1-2 LLM calls, 2-5s, $0.02-0.10)

Smart routing saves 80%+ of costs by avoiding LLM when not needed.
"""

from typing import Dict, Any
from datetime import datetime

from models.schemas import (
    UserQuery,
    QueryClassification,
    QueryType,
    QueryResponse,
    ProcessingMode,
)
from services.analytics_service import AnalyticsService
from services.rag_service import RAGService
from services.indexing_service import IndexingService
from utils.logger import get_logger, track_performance, metrics_collector
from utils.config_loader import config

logger = get_logger(__name__)


class QueryRouter:
    """
    Query router with classification and intelligent routing.

    Architecture Pattern: Strategy Pattern

    Each query type has its own processing strategy:
    - AGGREGATION → SQL + precomputed metrics
    - SEARCH → Vector/keyword search
    - INSIGHT → Full RAG pipeline
    - HYBRID → Combine multiple approaches
    """

    def __init__(self):
        self.analytics_service = AnalyticsService()
        self.rag_service = RAGService()
        self.indexing_service = IndexingService()

        self.classification_threshold = config.get(
            "services.query_router.classification_threshold", 0.6
        )

        # Classification patterns
        self._init_patterns()

        logger.info("query_router_initialized")

    def _init_patterns(self):
        """Initialize query classification patterns"""

        # Analytics query patterns
        self.analytics_patterns = [
            # Count queries
            r"\bhow many\b",
            r"\bcount\b",
            r"\bnumber of\b",
            r"\btotal\b",
            # Percentage queries
            r"\bpercentage\b",
            r"\bpercent\b",
            r"\bratio\b",
            r"\bproportion\b",
            # Trend queries
            r"\btrend\b",
            r"\bover time\b",
            r"\bhistory\b",
            r"\blast (week|month|year)\b",
            # Top/ranking queries
            r"\btop\b",
            r"\bmost common\b",
            r"\bhighest\b",
            r"\blowest\b",
            r"\brank\b",
            # Average/statistics
            r"\baverage\b",
            r"\bmean\b",
            r"\bmedian\b",
        ]

        # Search query patterns
        self.search_patterns = [
            r"\bfind\b",
            r"\bshow me\b",
            r"\blist\b",
            r"\bget\b",
            r"\bwhich calls\b",
            r"\blook for\b",
        ]

        # Insight query patterns
        self.insight_patterns = [
            r"\bwhat are.*saying\b",
            r"\bsummarize\b",
            r"\bexplain\b",
            r"\bwhy\b",
            r"\binsight\b",
            r"\banalysis\b",
            r"\bpattern\b",
            r"\btheme\b",
            r"\bcommon.*issue\b",
            r"\bmain.*concern\b",
        ]

    def classify_query(self, query_text: str) -> QueryClassification:
        """
        Classify query into type.

        Algorithm:
        ----------
        1. Pattern matching on query text
        2. Score each query type
        3. Select highest scoring type
        4. Determine routing

        This is a simple rule-based classifier.
        For production, consider:
        - Small BERT-based classifier
        - Few-shot LLM classification
        - Hybrid approach
        """
        import re

        query_lower = query_text.lower()

        # Score each type
        analytics_score = sum(
            1 for pattern in self.analytics_patterns if re.search(pattern, query_lower)
        )

        search_score = sum(
            1 for pattern in self.search_patterns if re.search(pattern, query_lower)
        )

        insight_score = sum(
            1 for pattern in self.insight_patterns if re.search(pattern, query_lower)
        )

        # Normalize scores
        total = max(analytics_score + search_score + insight_score, 1)
        analytics_conf = analytics_score / total
        search_conf = search_score / total
        insight_conf = insight_score / total

        # Determine primary type
        scores = {
            "analytics": analytics_conf,
            "search": search_conf,
            "insight": insight_conf,
        }

        primary_type = max(scores, key=scores.get)
        confidence = scores[primary_type]

        # Map to QueryType
        if primary_type == "analytics":
            query_type = QueryType.AGGREGATION
            route_to_analytics = True
            route_to_search = False
            route_to_rag = False
        elif primary_type == "search":
            query_type = QueryType.SEARCH
            route_to_analytics = False
            route_to_search = True
            route_to_rag = False
        else:
            query_type = QueryType.INSIGHT
            route_to_analytics = False
            route_to_search = False
            route_to_rag = True

        # If confidence is low, use hybrid approach
        if confidence < self.classification_threshold:
            query_type = QueryType.HYBRID
            route_to_analytics = analytics_conf > 0.3
            route_to_search = search_conf > 0.3
            route_to_rag = insight_conf > 0.3

        # Extract entities (simple keyword extraction)
        entities = self._extract_entities(query_text)

        classification = QueryClassification(
            query_id=query_text[:50],  # Use first 50 chars as ID
            query_type=query_type,
            confidence=confidence,
            route_to_analytics=route_to_analytics,
            route_to_search=route_to_search,
            route_to_rag=route_to_rag,
            intent=primary_type,
            entities=entities,
        )

        logger.info(
            "query_classified",
            query=query_text[:100],
            type=query_type.value,
            confidence=confidence,
        )

        metrics_collector.increment(f"query_type_{query_type.value}")

        return classification

    def _extract_entities(self, query_text: str) -> Dict[str, list]:
        """
        Extract entities from query text.

        Simple keyword-based extraction.
        For production, use NER model.
        """
        entities = {"products": [], "intents": [], "sentiments": [], "timeframes": []}

        query_lower = query_text.lower()

        # Products
        product_keywords = {
            "credit_card": ["credit card", "card"],
            "mortgage": ["mortgage", "home loan"],
            "retirement_fund": ["retirement", "401k", "pension"],
            "etf": ["etf", "fund"],
            "savings": ["savings"],
            "loan": ["loan"],
        }

        for product, keywords in product_keywords.items():
            if any(kw in query_lower for kw in keywords):
                entities["products"].append(product)

        # Intents
        if any(
            word in query_lower
            for word in ["complaint", "complain", "issue", "problem"]
        ):
            entities["intents"].append("complaint")

        if any(word in query_lower for word in ["question", "inquiry", "ask"]):
            entities["intents"].append("inquiry")

        # Sentiments
        if "negative" in query_lower or "unhappy" in query_lower:
            entities["sentiments"].append("negative")
        elif "positive" in query_lower or "happy" in query_lower:
            entities["sentiments"].append("positive")

        # Timeframes
        import re

        if re.search(r"\blast (day|week|month|year)\b", query_lower):
            match = re.search(r"\blast (\w+)\b", query_lower)
            if match:
                entities["timeframes"].append(match.group(1))

        return entities

    async def route_query(self, user_query: UserQuery) -> QueryResponse:
        """
        Route query to appropriate service(s).

        Process:
        -------
        1. Classify query
        2. Route to service(s)
        3. Merge results if hybrid
        4. Return response
        """
        with track_performance("route_query"):
            start_time = datetime.now()

            # Classify
            classification = self.classify_query(user_query.query_text)

            # Route based on classification
            if classification.query_type == QueryType.AGGREGATION:
                response = await self._route_to_analytics(user_query, classification)

            elif classification.query_type == QueryType.SEARCH:
                response = await self._route_to_search(user_query, classification)

            elif classification.query_type == QueryType.INSIGHT:
                response = await self._route_to_rag(user_query, classification)

            elif classification.query_type == QueryType.HYBRID:
                response = await self._route_hybrid(user_query, classification)

            else:
                response = self._create_error_response(user_query, "Unknown query type")

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            response.processing_time_ms = duration_ms

            logger.info(
                "query_routed",
                query_type=classification.query_type.value,
                duration_ms=duration_ms,
                cost=response.total_cost_usd,
            )

            return response

    async def _route_to_analytics(
        self, user_query: UserQuery, classification: QueryClassification
    ) -> QueryResponse:
        """
        Route to analytics service.

        NO LLM CALLS - Pure SQL.
        Fastest and cheapest option.
        """
        # Get analytics answer
        result = self.analytics_service.answer_natural_language_query(
            user_query.query_text
        )

        if result.get("success"):
            answer = result.get("answer", "")

            # Format data nicely
            if "data" in result:
                data = result["data"]
                if isinstance(data, list) and len(data) > 0:
                    answer += "\n\nTop results:\n"
                    for item in data[:5]:
                        if isinstance(item, dict):
                            if "category" in item and "count" in item:
                                answer += f"- {item['category']}: {item['count']}\n"
        else:
            answer = result.get("message", "Could not process analytics query.")

        return QueryResponse(
            query_id=user_query.query_id or "unknown",
            query_text=user_query.query_text,
            answer=answer,
            confidence=0.9 if result.get("success") else 0.3,
            sources=[],
            citations=[],
            analytics=result.get("data"),
            processing_time_ms=0,  # Set by caller
            total_cost_usd=0.0,  # No LLM cost
            model_used="none",
            query_type=QueryType.AGGREGATION,
            processing_mode=ProcessingMode.CHEAP,
            timestamp=datetime.now(),
        )

    async def _route_to_search(
        self, user_query: UserQuery, classification: QueryClassification
    ) -> QueryResponse:
        """
        Route to search service.

        Uses vector/keyword search only.
        No LLM synthesis.
        """
        # Perform search
        results = self.indexing_service.hybrid_search(
            query=user_query.query_text, top_k=10, filters=user_query.filters
        )

        # Format answer
        if results:
            answer = f"Found {len(results)} relevant call segments:\n\n"
            for i, result in enumerate(results[:5]):
                answer += f"{i+1}. {result.text[:150]}...\n"
                answer += f"   (Relevance: {result.score:.2f})\n\n"
        else:
            answer = "No relevant calls found for your query."

        return QueryResponse(
            query_id=user_query.query_id or "unknown",
            query_text=user_query.query_text,
            answer=answer,
            confidence=0.7 if results else 0.0,
            sources=results,
            citations=[],
            analytics=None,
            processing_time_ms=0,
            total_cost_usd=0.001,  # Minimal embedding cost
            model_used="search_only",
            query_type=QueryType.SEARCH,
            processing_mode=ProcessingMode.CHEAP,
            timestamp=datetime.now(),
        )

    async def _route_to_rag(
        self, user_query: UserQuery, classification: QueryClassification
    ) -> QueryResponse:
        """
        Route to RAG service.

        Full RAG pipeline with LLM synthesis.
        Most expensive but most capable.
        """
        return await self.rag_service.query(user_query)

    async def _route_hybrid(
        self, user_query: UserQuery, classification: QueryClassification
    ) -> QueryResponse:
        """
        Hybrid approach - combine multiple services.

        Use when query classification is uncertain.
        """
        # For now, prefer RAG for hybrid
        # In production, could run multiple in parallel and merge
        return await self._route_to_rag(user_query, classification)

    def _create_error_response(
        self, user_query: UserQuery, message: str
    ) -> QueryResponse:
        """Create error response"""
        return QueryResponse(
            query_id=user_query.query_id or "unknown",
            query_text=user_query.query_text,
            answer=f"Error: {message}",
            confidence=0.0,
            sources=[],
            citations=[],
            analytics=None,
            processing_time_ms=0,
            total_cost_usd=0.0,
            model_used="none",
            query_type=QueryType.INSIGHT,
            processing_mode=user_query.processing_mode,
            timestamp=datetime.now(),
        )


async def main():
    """Test query router"""
    router = QueryRouter()

    # Test queries
    test_queries = [
        "How many calls about credit cards?",
        "Find calls mentioning refunds",
        "What are customers saying about retirement fund performance?",
    ]

    for query_text in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query_text}")
        print("=" * 60)

        # Classify
        classification = router.classify_query(query_text)
        print(f"Type: {classification.query_type.value}")
        print(f"Confidence: {classification.confidence:.2f}")

        # Route
        query = UserQuery(
            query_text=query_text,
            query_id=f"test_{hash(query_text)}",
            processing_mode=ProcessingMode.BALANCED,
        )

        response = await router.route_query(query)
        print(f"\nAnswer: {response.answer[:200]}...")
        print(f"Cost: ${response.total_cost_usd:.4f}")
        print(f"Time: {response.processing_time_ms:.0f}ms")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
