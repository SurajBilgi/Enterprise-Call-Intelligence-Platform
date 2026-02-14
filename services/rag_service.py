"""
RAG Service - Retrieval Augmented Generation
=============================================

Handles complex semantic queries that require:
1. Finding relevant call transcripts (Retrieval)
2. Synthesizing insights from multiple calls (Generation)
3. Providing evidence-based answers with citations

Examples:
- "What are customers saying about retirement fund performance?"
- "Summarize common issues with credit card access"
- "What concerns do premium customers have about fees?"

System Design:
-------------
RAG Pipeline Stages:
1. Query Understanding
2. Hybrid Retrieval (keyword + semantic)
3. Re-ranking
4. Context Construction
5. LLM Synthesis
6. Citation Generation

Cost Optimization:
-----------------
- Use retrieval to filter before LLM
- Only send relevant context to LLM
- Use context compression
- Cache similar queries
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.schemas import (
    UserQuery,
    QueryResponse,
    QueryType,
    ProcessingMode,
    SearchResult,
    RetrievalContext,
)
from services.indexing_service import IndexingService
from services.enrichment_service import LLMProvider
from utils.logger import get_logger, track_performance, cost_tracker, metrics_collector
from utils.config_loader import config

logger = get_logger(__name__)


class RAGService:
    """
    RAG service for complex semantic queries.

    Architecture Pattern: Retrieval-Augmented Generation

    Key Innovation:
    --------------
    Instead of fine-tuning a model on all data (expensive, slow):
    1. Retrieve relevant documents dynamically
    2. Feed to general-purpose LLM
    3. Get contextualized answers

    Benefits:
    - Always up-to-date (no retraining)
    - Transparent (can show sources)
    - Cost-effective (only process relevant data)
    """

    def __init__(self):
        self.indexing_service = IndexingService()
        self.llm = LLMProvider()

        self.retrieval_top_k = config.get("services.rag.retrieval_top_k", 10)
        self.rerank_top_k = config.get("services.rag.rerank_top_k", 5)
        self.enable_hybrid = config.get("services.rag.enable_hybrid_search", True)
        self.max_context_tokens = config.get("services.rag.max_context_tokens", 4000)

        logger.info("rag_service_initialized")

    async def query(self, user_query: UserQuery) -> QueryResponse:
        """
        Process a RAG query end-to-end.

        Pipeline:
        1. Retrieve relevant segments
        2. Rerank for relevance
        3. Construct context
        4. Generate answer with LLM
        5. Extract citations
        """
        with track_performance(f"rag_query_{user_query.query_id}"):
            start_time = datetime.now()

            # 1. RETRIEVAL
            retrieval_context = await self._retrieve(user_query)

            if len(retrieval_context.results) == 0:
                # No relevant results found
                return self._create_empty_response(user_query, start_time)

            # 2. RE-RANKING (optional, for now just take top-k)
            ranked_results = retrieval_context.results[: self.rerank_top_k]

            # 3. CONTEXT CONSTRUCTION
            context = self._construct_context(ranked_results)

            # 4. LLM SYNTHESIS
            answer, model_used, tokens_used, cost = await self._synthesize_answer(
                query=user_query.query_text,
                context=context,
                mode=user_query.processing_mode,
            )

            # 5. EXTRACT CITATIONS
            citations = self._generate_citations(ranked_results)

            # Calculate total time
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Build response
            response = QueryResponse(
                query_id=user_query.query_id or "unknown",
                query_text=user_query.query_text,
                answer=answer,
                confidence=self._calculate_confidence(ranked_results),
                sources=ranked_results,
                citations=citations,
                analytics=None,
                processing_time_ms=duration_ms,
                total_cost_usd=cost,
                model_used=model_used,
                query_type=QueryType.INSIGHT,
                processing_mode=user_query.processing_mode,
                timestamp=datetime.now(),
            )

            logger.info(
                "rag_query_completed",
                query_id=user_query.query_id,
                num_sources=len(ranked_results),
                cost=cost,
                duration_ms=duration_ms,
            )

            metrics_collector.increment("rag_queries")

            return response

    async def _retrieve(self, user_query: UserQuery) -> RetrievalContext:
        """
        Retrieve relevant segments.

        Strategy:
        --------
        - Use hybrid search (keyword + semantic)
        - Apply filters if provided
        - Return top-k results
        """
        # Extract filters
        filters = user_query.filters or {}

        # Perform search
        if self.enable_hybrid:
            results = self.indexing_service.hybrid_search(
                query=user_query.query_text, top_k=self.retrieval_top_k, filters=filters
            )
            retrieval_method = "hybrid"
        else:
            results = self.indexing_service.semantic_search(
                query=user_query.query_text, top_k=self.retrieval_top_k, filters=filters
            )
            retrieval_method = "semantic"

        logger.info(
            "retrieval_completed",
            query=user_query.query_text,
            num_results=len(results),
            method=retrieval_method,
        )

        return RetrievalContext(
            results=results,
            total_retrieved=len(results),
            retrieval_method=retrieval_method,
            retrieval_timestamp=datetime.now(),
        )

    def _construct_context(self, results: List[SearchResult]) -> str:
        """
        Construct context for LLM from retrieved results.

        Context Engineering:
        -------------------
        - Include most relevant segments
        - Add metadata (product, sentiment, etc.)
        - Format for clarity
        - Stay within token limits
        """
        context_parts = []
        current_tokens = 0

        for i, result in enumerate(results):
            # Format segment
            segment_text = f"""
[Call {i+1}]
Product: {result.metadata.get('product', 'N/A')}
Intent: {result.metadata.get('intent', 'N/A')}
Sentiment: {result.metadata.get('sentiment', 'N/A')}
Relevance: {result.score:.2f}

Transcript:
{result.text}
---
"""

            # Estimate tokens (rough: 1 token ~= 4 chars)
            estimated_tokens = len(segment_text) // 4

            if current_tokens + estimated_tokens > self.max_context_tokens:
                break

            context_parts.append(segment_text)
            current_tokens += estimated_tokens

        return "\n".join(context_parts)

    async def _synthesize_answer(
        self, query: str, context: str, mode: ProcessingMode
    ) -> tuple[str, str, int, float]:
        """
        Generate answer using LLM.

        Prompt Engineering:
        ------------------
        - Clear instructions
        - Structured output
        - Request citations
        - Emphasize accuracy

        Returns:
            (answer, model_used, tokens_used, cost)
        """
        # Select model based on mode
        if mode == ProcessingMode.DEEP:
            model = config.get("services.enrichment.expensive_model")
        else:
            model = config.get("services.enrichment.cheap_model")

        # Construct prompt
        prompt = f"""You are an AI assistant analyzing customer support call transcripts.

User Question:
{query}

Relevant Call Transcripts:
{context}

Instructions:
1. Answer the user's question based ONLY on the provided transcripts
2. Be specific and cite evidence from the transcripts
3. If the transcripts don't contain enough information, say so
4. Organize your answer clearly
5. Highlight key insights and patterns

Answer:"""

        # Generate
        result = await self.llm.generate(prompt=prompt, model=model, max_tokens=500)

        return (
            result["output"],
            result["model"],
            result["total_tokens"],
            result["cost"],
        )

    def _generate_citations(self, results: List[SearchResult]) -> List[str]:
        """Generate citation strings"""
        citations = []

        for i, result in enumerate(results):
            citation = (
                f"[{i+1}] Call {result.call_id} - "
                f"{result.metadata.get('product', 'N/A')} - "
                f"Score: {result.score:.2f}"
            )
            citations.append(citation)

        return citations

    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """
        Calculate confidence score based on retrieval quality.

        Factors:
        - Number of results
        - Average score
        - Score variance (consistency)
        """
        if not results:
            return 0.0

        # Average score
        avg_score = sum(r.score for r in results) / len(results)

        # Number of results factor
        result_factor = min(len(results) / 5, 1.0)

        # Combined confidence
        confidence = avg_score * 0.7 + result_factor * 0.3

        return min(confidence, 1.0)

    def _create_empty_response(
        self, user_query: UserQuery, start_time: datetime
    ) -> QueryResponse:
        """Create response when no results found"""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return QueryResponse(
            query_id=user_query.query_id or "unknown",
            query_text=user_query.query_text,
            answer="I couldn't find any relevant call transcripts matching your query. Try rephrasing or broadening your search.",
            confidence=0.0,
            sources=[],
            citations=[],
            analytics=None,
            processing_time_ms=duration_ms,
            total_cost_usd=0.0,
            model_used="none",
            query_type=QueryType.INSIGHT,
            processing_mode=user_query.processing_mode,
            timestamp=datetime.now(),
        )


async def main():
    """Test RAG service"""
    from models.schemas import UserQuery, ProcessingMode

    service = RAGService()

    # Test query
    query = UserQuery(
        query_text="What are customers complaining about?",
        query_id="test_001",
        processing_mode=ProcessingMode.BALANCED,
    )

    print(f"Query: {query.query_text}")
    print("Processing...")

    response = await service.query(query)

    print(f"\nAnswer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Sources: {len(response.sources)}")
    print(f"Cost: ${response.total_cost_usd:.4f}")
    print(f"Time: {response.processing_time_ms:.0f}ms")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
