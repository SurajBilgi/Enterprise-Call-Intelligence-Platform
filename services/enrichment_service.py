"""
Enrichment Service - Selective LLM Usage for Cost Optimization
===============================================================

This service demonstrates INTELLIGENT LLM USAGE:
- Only processes calls flagged by preprocessing
- Uses cheap models for simple tasks
- Uses expensive models for complex analysis
- Tracks costs meticulously

Cost Strategy:
--------------
1. Preprocessing filters 85% of calls (no LLM cost)
2. For remaining 15%:
   - Use GPT-3.5 for summarization (~$0.002/call)
   - Use GPT-4 only for critical issues (~$0.05/call)
3. Result: ~$5K for 1M calls vs $50K without filtering

This service is where the REAL money is spent.
Every call here should be justified by preprocessing.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

import tiktoken
import duckdb

from models.schemas import EnrichmentResult, ImportanceLevel, IntentCategory
from utils.logger import get_logger, track_performance, cost_tracker, metrics_collector
from utils.config_loader import config

logger = get_logger(__name__)


class LLMProvider:
    """
    Mock LLM Provider for demonstration.

    In production, replace with real OpenAI client.
    This mock simulates:
    - API calls
    - Token counting
    - Cost calculation
    - Different model capabilities
    """

    def __init__(self):
        self.cheap_model = config.get("llm.models.cheap.name", "gpt-3.5-turbo")
        self.expensive_model = config.get(
            "llm.models.expensive.name", "gpt-4-turbo-preview"
        )

        self.cheap_input_cost = config.get(
            "llm.models.cheap.cost_per_1k_tokens", 0.0015
        )
        self.cheap_output_cost = config.get(
            "llm.models.cheap.cost_per_1k_output_tokens", 0.002
        )

        self.expensive_input_cost = config.get(
            "llm.models.expensive.cost_per_1k_tokens", 0.01
        )
        self.expensive_output_cost = config.get(
            "llm.models.expensive.cost_per_1k_output_tokens", 0.03
        )

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(
            "llm_provider_initialized",
            cheap_model=self.cheap_model,
            expensive_model=self.expensive_model,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    async def generate(
        self, prompt: str, model: str = None, max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate completion (MOCKED for demo).

        In production, replace with:
        ```python
        import openai
        response = await openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        ```
        """
        if model is None:
            model = self.cheap_model

        # Count input tokens
        input_tokens = self.count_tokens(prompt)

        # Simulate output (in production, this comes from API)
        output = self._mock_generation(prompt, model)
        output_tokens = self.count_tokens(output)

        # Calculate cost
        if "gpt-4" in model:
            cost = (input_tokens / 1000) * self.expensive_input_cost + (
                output_tokens / 1000
            ) * self.expensive_output_cost
        else:
            cost = (input_tokens / 1000) * self.cheap_input_cost + (
                output_tokens / 1000
            ) * self.cheap_output_cost

        # Track cost
        cost_tracker.track_llm_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_per_1k_input=(
                self.expensive_input_cost if "gpt-4" in model else self.cheap_input_cost
            ),
            cost_per_1k_output=(
                self.expensive_output_cost
                if "gpt-4" in model
                else self.cheap_output_cost
            ),
        )

        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "model": model,
        }

    def _mock_generation(self, prompt: str, model: str) -> str:
        """
        Mock LLM generation for demo purposes.

        In production, remove this and use real API.
        """
        prompt_lower = prompt.lower()

        # Detect task type from prompt
        if "summarize" in prompt_lower:
            return (
                "Customer contacted regarding account issues. "
                "Main concerns: billing discrepancy and access problems. "
                "Agent provided resolution steps. Status: Resolved."
            )

        elif "key issues" in prompt_lower or "extract issues" in prompt_lower:
            return json.dumps(
                ["Billing discrepancy", "Account access problem", "Service delay"]
            )

        elif "action items" in prompt_lower:
            return json.dumps(
                [
                    "Refund processing",
                    "Account verification",
                    "Follow-up call scheduled",
                ]
            )

        elif "semantic tags" in prompt_lower:
            return json.dumps(
                [
                    "billing",
                    "customer_dissatisfaction",
                    "refund_request",
                    "access_issue",
                ]
            )

        else:
            return "Analysis completed successfully."


class EnrichmentService:
    """
    Selective LLM enrichment service.

    KEY PRINCIPLE: Only enrich what's important.

    Architecture Pattern: Tiered Processing
    - Tier 1: Cheap model for all flagged calls (summarization)
    - Tier 2: Expensive model for critical calls only (deep analysis)
    """

    def __init__(self):
        self.db_path = config.storage_paths["structured_db"]
        self.llm = LLMProvider()

        self.enable_selective = config.get(
            "services.enrichment.enable_selective_enrichment", True
        )
        self.enrich_threshold = config.get("services.enrichment.enrich_threshold", 0.4)
        self.use_cheap_for_triage = config.get(
            "services.enrichment.use_cheap_model_for_triage", True
        )

        self.cheap_model = config.get(
            "services.enrichment.cheap_model", "gpt-3.5-turbo"
        )
        self.expensive_model = config.get(
            "services.enrichment.expensive_model", "gpt-4-turbo-preview"
        )

        self._init_database()

        logger.info(
            "enrichment_service_initialized",
            selective_enrichment=self.enable_selective,
            threshold=self.enrich_threshold,
        )

    def _init_database(self):
        """Initialize enrichment results table"""
        conn = duckdb.connect(str(self.db_path))

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enrichments (
                enrichment_id VARCHAR PRIMARY KEY,
                call_id VARCHAR,
                segment_id VARCHAR,
                summary TEXT,
                key_issues TEXT,
                action_items TEXT,
                semantic_tags TEXT,
                model_used VARCHAR,
                tokens_used INTEGER,
                cost_usd DOUBLE,
                enrichment_timestamp TIMESTAMP,
                confidence_score DOUBLE
            )
        """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_enrichment_call ON enrichments(call_id)"
        )

        conn.close()

    async def enrich_call(
        self, call_id: str, force_expensive: bool = False
    ) -> EnrichmentResult:
        """
        Enrich a single call with LLM.

        COST DECISION TREE:
        -------------------
        1. Check importance level
        2. If CRITICAL or force_expensive:
           -> Use GPT-4 (expensive but thorough)
        3. Else if HIGH/MEDIUM:
           -> Use GPT-3.5 (cheap and fast)
        4. Else:
           -> Skip enrichment (shouldn't reach here if preprocessing works)

        Args:
            call_id: Call to enrich
            force_expensive: Force expensive model (for critical cases)
        """
        with track_performance(f"enrich_call_{call_id}"):
            # Load preprocessing result
            preprocessing = self._load_preprocessing(call_id)
            if not preprocessing:
                raise ValueError(f"No preprocessing result for {call_id}")

            # Check if enrichment is needed
            if not preprocessing["requires_enrichment"]:
                logger.warning(
                    "enrichment_not_required",
                    call_id=call_id,
                    reason="Preprocessing did not flag for enrichment",
                )
                # Return minimal enrichment
                return self._create_minimal_enrichment(call_id)

            # Select model based on importance
            importance = preprocessing["overall_importance"]

            if force_expensive or importance == "critical":
                model = self.expensive_model
                logger.info(
                    "using_expensive_model",
                    call_id=call_id,
                    reason="Critical importance",
                )
            else:
                model = self.cheap_model
                logger.info(
                    "using_cheap_model", call_id=call_id, reason="Standard importance"
                )

            # Load full transcript
            transcript = self._load_transcript(call_id)

            # Generate enrichment
            result = await self._generate_enrichment(
                call_id=call_id,
                transcript=transcript,
                preprocessing=preprocessing,
                model=model,
            )

            # Store result
            self._store_enrichment(result)

            # Mark as enriched
            conn = duckdb.connect(str(self.db_path))
            conn.execute(
                "UPDATE calls SET enriched = TRUE WHERE call_id = ?", [call_id]
            )
            conn.close()

            logger.info(
                "call_enriched",
                call_id=call_id,
                model=model,
                cost=result.cost_usd,
                tokens=result.tokens_used,
            )

            metrics_collector.increment("calls_enriched")
            metrics_collector.increment(f"enrichment_{model.replace('-', '_')}")

            return result

    def _load_preprocessing(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Load preprocessing result"""
        conn = duckdb.connect(str(self.db_path))

        result = conn.execute(
            "SELECT * FROM preprocessed_calls WHERE call_id = ?", [call_id]
        ).fetchone()

        conn.close()

        if not result:
            return None

        columns = [
            "call_id",
            "overall_intent",
            "overall_sentiment",
            "overall_importance",
            "importance_score",
            "requires_enrichment",
            "enrichment_reason",
            "total_segments",
            "high_priority_segments",
            "processing_timestamp",
        ]

        return dict(zip(columns, result))

    def _load_transcript(self, call_id: str) -> str:
        """Load and format transcript for LLM"""
        conn = duckdb.connect(str(self.db_path))

        segments = conn.execute(
            "SELECT speaker, text FROM segments WHERE call_id = ? ORDER BY segment_id",
            [call_id],
        ).fetchall()

        conn.close()

        # Format as conversation
        transcript_lines = []
        for speaker, text in segments:
            transcript_lines.append(f"{speaker.upper()}: {text}")

        return "\n".join(transcript_lines)

    async def _generate_enrichment(
        self, call_id: str, transcript: str, preprocessing: Dict[str, Any], model: str
    ) -> EnrichmentResult:
        """
        Generate enrichment using LLM.

        Prompt Engineering for Cost Efficiency:
        - Clear, concise instructions
        - Structured output (JSON)
        - No unnecessary elaboration
        """
        # Task 1: Summarization
        summary_prompt = f"""Summarize this customer support call in 2-3 sentences.
Focus on the main issue and resolution.

Call transcript:
{transcript}

Provide a concise summary:"""

        summary_result = await self.llm.generate(
            summary_prompt, model=model, max_tokens=150
        )

        # Task 2: Extract key issues
        issues_prompt = f"""Extract the key issues from this call.
Return as a JSON array of strings.

Call transcript:
{transcript}

Key issues (JSON array):"""

        issues_result = await self.llm.generate(
            issues_prompt, model=model, max_tokens=100
        )

        try:
            key_issues = json.loads(issues_result["output"])
            if not isinstance(key_issues, list):
                key_issues = [issues_result["output"]]
        except:
            key_issues = [issues_result["output"]]

        # Task 3: Extract action items
        actions_prompt = f"""Extract action items from this call.
Return as a JSON array of strings.

Call transcript:
{transcript}

Action items (JSON array):"""

        actions_result = await self.llm.generate(
            actions_prompt, model=model, max_tokens=100
        )

        try:
            action_items = json.loads(actions_result["output"])
            if not isinstance(action_items, list):
                action_items = [actions_result["output"]]
        except:
            action_items = [actions_result["output"]]

        # Task 4: Generate semantic tags
        tags_prompt = f"""Generate 3-5 semantic tags for this call for search and categorization.
Return as a JSON array of lowercase strings.

Call transcript:
{transcript}

Semantic tags (JSON array):"""

        tags_result = await self.llm.generate(tags_prompt, model=model, max_tokens=50)

        try:
            semantic_tags = json.loads(tags_result["output"])
            if not isinstance(semantic_tags, list):
                semantic_tags = [tags_result["output"]]
        except:
            semantic_tags = [tags_result["output"]]

        # Calculate total cost and tokens
        total_tokens = (
            summary_result["total_tokens"]
            + issues_result["total_tokens"]
            + actions_result["total_tokens"]
            + tags_result["total_tokens"]
        )

        total_cost = (
            summary_result["cost"]
            + issues_result["cost"]
            + actions_result["cost"]
            + tags_result["cost"]
        )

        # Calculate confidence score (mock)
        confidence = 0.85 if model == self.expensive_model else 0.75

        return EnrichmentResult(
            call_id=call_id,
            segment_id=None,
            summary=summary_result["output"],
            key_issues=key_issues,
            action_items=action_items,
            semantic_tags=semantic_tags,
            model_used=model,
            tokens_used=total_tokens,
            cost_usd=round(total_cost, 4),
            enrichment_timestamp=datetime.now(),
            confidence_score=confidence,
        )

    def _create_minimal_enrichment(self, call_id: str) -> EnrichmentResult:
        """Create minimal enrichment for low-priority calls"""
        return EnrichmentResult(
            call_id=call_id,
            segment_id=None,
            summary="General inquiry call. No critical issues.",
            key_issues=[],
            action_items=[],
            semantic_tags=["general", "routine"],
            model_used="none",
            tokens_used=0,
            cost_usd=0.0,
            enrichment_timestamp=datetime.now(),
            confidence_score=0.5,
        )

    def _store_enrichment(self, result: EnrichmentResult):
        """Store enrichment result"""
        conn = duckdb.connect(str(self.db_path))

        enrichment_id = f"{result.call_id}_ENRICH"

        conn.execute(
            """
            INSERT OR REPLACE INTO enrichments (
                enrichment_id, call_id, segment_id, summary, key_issues,
                action_items, semantic_tags, model_used, tokens_used,
                cost_usd, enrichment_timestamp, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                enrichment_id,
                result.call_id,
                result.segment_id,
                result.summary,
                json.dumps(result.key_issues),
                json.dumps(result.action_items),
                json.dumps(result.semantic_tags),
                result.model_used,
                result.tokens_used,
                result.cost_usd,
                result.enrichment_timestamp,
                result.confidence_score,
            ],
        )

        conn.close()

    async def enrich_batch(
        self, call_ids: List[str], batch_size: int = 10
    ) -> List[EnrichmentResult]:
        """
        Enrich a batch of calls with rate limiting.

        System Design Note:
        ------------------
        Batch processing with concurrency control:
        - Prevents API rate limits
        - Controls cost burn rate
        - Maintains system stability
        """
        results = []

        for i in range(0, len(call_ids), batch_size):
            batch = call_ids[i : i + batch_size]

            batch_results = await asyncio.gather(
                *[self.enrich_call(cid) for cid in batch], return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("enrichment_error", error=str(result))
                    metrics_collector.increment("enrichment_errors")
                else:
                    results.append(result)

            # Rate limiting pause
            await asyncio.sleep(0.1)

        return results

    def get_enrichment_statistics(self) -> Dict[str, Any]:
        """Get enrichment cost and usage statistics"""
        conn = duckdb.connect(str(self.db_path))

        stats = {}

        # Total enrichments
        stats["total_enrichments"] = conn.execute(
            "SELECT COUNT(*) FROM enrichments"
        ).fetchone()[0]

        # Total cost
        stats["total_cost_usd"] = (
            conn.execute("SELECT SUM(cost_usd) FROM enrichments").fetchone()[0] or 0.0
        )

        # By model
        stats["by_model"] = (
            conn.execute(
                """
            SELECT model_used, COUNT(*) as count, SUM(cost_usd) as total_cost
            FROM enrichments
            GROUP BY model_used
        """
            )
            .fetchdf()
            .to_dict("records")
        )

        # Average cost
        if stats["total_enrichments"] > 0:
            stats["avg_cost_per_call"] = (
                stats["total_cost_usd"] / stats["total_enrichments"]
            )
        else:
            stats["avg_cost_per_call"] = 0.0

        conn.close()

        return stats


async def main():
    """Run enrichment service"""
    service = EnrichmentService()

    from services.preprocessing_service import PreprocessingService

    preprocessing = PreprocessingService()

    # Get calls requiring enrichment
    calls_to_enrich = preprocessing.get_calls_requiring_enrichment(limit=5)

    print(f"Enriching {len(calls_to_enrich)} calls...")

    for call_id in calls_to_enrich:
        result = await service.enrich_call(call_id)
        print(f"  {call_id}: ${result.cost_usd:.4f} ({result.tokens_used} tokens)")

    # Show statistics
    stats = service.get_enrichment_statistics()
    print(f"\nEnrichment Statistics:")
    print(f"  Total enrichments: {stats['total_enrichments']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"  Average cost: ${stats['avg_cost_per_call']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
