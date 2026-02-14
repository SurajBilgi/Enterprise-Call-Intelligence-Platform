"""
Preprocessing Service - THE MOST CRITICAL SERVICE FOR COST OPTIMIZATION
========================================================================

This service answers the KEY QUESTION:
"Which calls and which parts of speech are important?"

Responsibilities:
- Segment conversations
- Cheap NLP triage (NO LLM usage here - critical for cost)
- Intent classification
- Sentiment analysis
- Keyword spotting
- Importance scoring
- Triage decision (which calls need expensive LLM enrichment)

System Design Philosophy:
-------------------------
For MILLIONS of calls:
1. We CANNOT run LLM on every call (too expensive)
2. We MUST use cheap NLP to filter/score first
3. Only HIGH-IMPORTANCE calls get LLM enrichment
4. This service determines processing cost

Cost Impact Example:
-------------------
- Without triage: 1M calls × $0.05/call = $50,000
- With triage (15% enrichment): 150K calls × $0.05/call = $7,500
- Savings: $42,500 (85% reduction)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import asyncio

import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from models.schemas import (
    IntentCategory,
    SentimentScore,
    ImportanceLevel,
    SegmentAnalysis,
    PreprocessingResult,
)
from utils.logger import get_logger, track_performance, metrics_collector
from utils.config_loader import config

logger = get_logger(__name__)


class PreprocessingService:
    """
    Preprocessing and triage service.

    ARCHITECTURE PATTERN: Rule-based + Statistical NLP

    Why NOT use LLM here?
    - LLMs are expensive ($0.0015/1K tokens minimum)
    - For 1M calls, preprocessing with LLM = $50K+
    - Rule-based + stats can achieve 80%+ accuracy at <$100

    This is the GATEWAY that protects expensive LLM operations.
    """

    def __init__(self):
        self.db_path = config.storage_paths["structured_db"]
        self.raw_transcripts_path = config.storage_paths["raw_transcripts"]

        # Load thresholds from config
        self.high_priority_threshold = config.get(
            "services.preprocessing.importance_thresholds.high_priority", 0.7
        )
        self.medium_priority_threshold = config.get(
            "services.preprocessing.importance_thresholds.medium_priority", 0.4
        )

        # Load keywords
        self.high_priority_keywords = set(
            config.get("services.preprocessing.high_priority_keywords", [])
        )
        self.medium_priority_keywords = set(
            config.get("services.preprocessing.medium_priority_keywords", [])
        )

        # Initialize preprocessed_calls table
        self._init_database()

        # Compile regex patterns for efficiency
        self._compile_patterns()

        logger.info(
            "preprocessing_service_initialized",
            high_threshold=self.high_priority_threshold,
            medium_threshold=self.medium_priority_threshold,
            high_keywords=len(self.high_priority_keywords),
            medium_keywords=len(self.medium_priority_keywords),
        )

    def _init_database(self):
        """Initialize database tables for preprocessing results"""
        conn = duckdb.connect(str(self.db_path))

        # Preprocessing results table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preprocessed_calls (
                call_id VARCHAR PRIMARY KEY,
                overall_intent VARCHAR,
                overall_sentiment VARCHAR,
                overall_importance VARCHAR,
                importance_score DOUBLE,
                requires_enrichment BOOLEAN,
                enrichment_reason VARCHAR,
                total_segments INTEGER,
                high_priority_segments INTEGER,
                processing_timestamp TIMESTAMP
            )
        """
        )

        # Segment analysis table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                segment_id VARCHAR PRIMARY KEY,
                call_id VARCHAR,
                speaker VARCHAR,
                text VARCHAR,
                intent VARCHAR,
                sentiment VARCHAR,
                importance_score DOUBLE,
                requires_llm_enrichment BOOLEAN,
                is_critical BOOLEAN,
                high_priority_keywords VARCHAR,
                products_mentioned VARCHAR,
                issues_mentioned VARCHAR
            )
        """
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_segment_call ON segments(call_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_segment_importance ON segments(importance_score)"
        )

        conn.close()

    def _compile_patterns(self):
        """Compile regex patterns for fast matching"""
        # Complaint indicators
        self.complaint_pattern = re.compile(
            r"\b(complaint|complain|terrible|awful|horrible|frustrated|angry|"
            r"unacceptable|disappointed|terrible|awful|refund|cancel|lawsuit|fraud)\b",
            re.IGNORECASE,
        )

        # Issue indicators
        self.issue_pattern = re.compile(
            r"\b(issue|problem|error|bug|broken|not working|failed|wrong|"
            r"mistake|incorrect|dispute)\b",
            re.IGNORECASE,
        )

        # Question indicators
        self.question_pattern = re.compile(
            r"\b(how|what|when|where|why|can you|could you|would you|"
            r"question|help|explain|understand)\b",
            re.IGNORECASE,
        )

        # Negative sentiment
        self.negative_pattern = re.compile(
            r"\b(hate|terrible|awful|bad|horrible|worst|disappointed|"
            r"frustrated|annoyed|upset|angry)\b",
            re.IGNORECASE,
        )

        # Positive sentiment
        self.positive_pattern = re.compile(
            r"\b(great|excellent|perfect|thank|thanks|appreciate|wonderful|"
            r"fantastic|amazing|love|happy)\b",
            re.IGNORECASE,
        )

    async def preprocess_call(self, call_id: str) -> PreprocessingResult:
        """
        Preprocess a single call.

        CORE ALGORITHM:
        ---------------
        1. Load transcript
        2. Segment into speaker turns
        3. For each segment:
           a. Classify intent (rule-based)
           b. Analyze sentiment (pattern matching)
           c. Detect keywords (set matching)
           d. Calculate importance score (weighted formula)
        4. Aggregate to call level
        5. Make triage decision

        NO LLM CALLS IN THIS FUNCTION - This is the whole point!
        """
        with track_performance(f"preprocess_call_{call_id}"):
            # Load transcript
            transcript_data = self._load_transcript(call_id)
            if not transcript_data:
                raise ValueError(f"Transcript not found: {call_id}")

            # Analyze each segment
            segments = []
            for idx, turn in enumerate(transcript_data["transcript"]):
                segment = self._analyze_segment(
                    call_id=call_id,
                    segment_id=f"{call_id}_SEG_{idx:03d}",
                    speaker=turn["speaker"],
                    text=turn["text"],
                )
                segments.append(segment)

            # Aggregate to call level
            result = self._aggregate_segments(call_id, segments)

            # Store results
            self._store_preprocessing_result(result, segments)

            # Update call status
            conn = duckdb.connect(str(self.db_path))
            conn.execute(
                "UPDATE calls SET preprocessed = TRUE WHERE call_id = ?", [call_id]
            )
            conn.close()

            logger.info(
                "call_preprocessed",
                call_id=call_id,
                importance=result.overall_importance.value,
                requires_enrichment=result.requires_enrichment,
                segments=len(segments),
            )

            metrics_collector.increment("calls_preprocessed")
            metrics_collector.increment(f"importance_{result.overall_importance.value}")

            return result

    def _load_transcript(self, call_id: str) -> Dict[str, Any]:
        """Load raw transcript from file"""
        transcript_file = self.raw_transcripts_path / f"{call_id}.json"

        if not transcript_file.exists():
            return None

        with open(transcript_file, "r") as f:
            return json.load(f)

    def _analyze_segment(
        self, call_id: str, segment_id: str, speaker: str, text: str
    ) -> SegmentAnalysis:
        """
        Analyze a single conversation segment.

        This is where IMPORTANCE SCORING happens.

        Scoring Algorithm:
        ------------------
        Base score = 0.3 (neutral)

        Intent multipliers:
        - Complaint: +0.4
        - Issue: +0.3
        - Question: +0.2
        - General: +0.0

        Sentiment adjustments:
        - Very Negative: +0.2
        - Negative: +0.1
        - Neutral: +0.0
        - Positive: -0.1

        Keyword bonuses:
        - High-priority keyword: +0.15 each (max 0.3)
        - Medium-priority keyword: +0.05 each (max 0.1)

        Final score clamped to [0, 1]
        """
        text_lower = text.lower()

        # 1. INTENT CLASSIFICATION (rule-based)
        intent = self._classify_intent(text)

        # 2. SENTIMENT ANALYSIS (pattern matching)
        sentiment = self._analyze_sentiment(text)

        # 3. KEYWORD DETECTION
        high_keywords = [
            kw for kw in self.high_priority_keywords if kw.lower() in text_lower
        ]
        medium_keywords = [
            kw for kw in self.medium_priority_keywords if kw.lower() in text_lower
        ]

        # 4. ENTITY EXTRACTION (simple pattern matching)
        products = self._extract_products(text)
        issues = self._extract_issues(text)

        # 5. IMPORTANCE SCORE CALCULATION
        importance_score = self._calculate_importance_score(
            intent=intent,
            sentiment=sentiment,
            high_keywords=len(high_keywords),
            medium_keywords=len(medium_keywords),
            is_customer=speaker.lower() == "customer",
        )

        # 6. TRIAGE DECISIONS
        requires_llm = importance_score >= self.medium_priority_threshold
        is_critical = importance_score >= self.high_priority_threshold

        return SegmentAnalysis(
            segment_id=segment_id,
            text=text,
            speaker=speaker,
            intent=intent,
            sentiment=sentiment,
            importance_score=importance_score,
            high_priority_keywords_found=high_keywords,
            medium_priority_keywords_found=medium_keywords,
            products_mentioned=products,
            issues_mentioned=issues,
            requires_llm_enrichment=requires_llm,
            is_critical=is_critical,
        )

    def _classify_intent(self, text: str) -> IntentCategory:
        """
        Classify intent using rule-based approach.

        Why rule-based?
        - Fast (microseconds vs seconds for LLM)
        - Cheap (no API costs)
        - Predictable (no hallucinations)
        - Good enough for triage (80%+ accuracy)
        """
        text_lower = text.lower()

        # Check for complaints
        if self.complaint_pattern.search(text):
            return IntentCategory.COMPLAINT

        # Check for technical issues
        if self.issue_pattern.search(text):
            return IntentCategory.TECHNICAL_ISSUE

        # Check for billing disputes
        if any(word in text_lower for word in ["charge", "bill", "fee", "payment"]):
            if any(word in text_lower for word in ["wrong", "incorrect", "dispute"]):
                return IntentCategory.BILLING_DISPUTE

        # Check for inquiries
        if self.question_pattern.search(text) or "?" in text:
            return IntentCategory.INQUIRY

        # Check for account management
        if any(
            word in text_lower for word in ["update", "change", "modify", "account"]
        ):
            return IntentCategory.ACCOUNT_MANAGEMENT

        # Default to general
        return IntentCategory.GENERAL

    def _analyze_sentiment(self, text: str) -> SentimentScore:
        """
        Analyze sentiment using pattern matching.

        Could be replaced with a small local model (e.g., distilbert-sentiment)
        for better accuracy without API costs.
        """
        negative_count = len(self.negative_pattern.findall(text))
        positive_count = len(self.positive_pattern.findall(text))

        # Simple scoring
        if negative_count >= 2:
            return SentimentScore.VERY_NEGATIVE
        elif negative_count >= 1:
            return SentimentScore.NEGATIVE
        elif positive_count >= 2:
            return SentimentScore.VERY_POSITIVE
        elif positive_count >= 1:
            return SentimentScore.POSITIVE
        else:
            return SentimentScore.NEUTRAL

    def _extract_products(self, text: str) -> List[str]:
        """Extract product mentions"""
        products = []
        text_lower = text.lower()

        product_keywords = {
            "credit card": ["credit card", "card"],
            "mortgage": ["mortgage", "home loan"],
            "retirement fund": ["retirement", "401k", "pension", "ira"],
            "etf": ["etf", "index fund", "mutual fund"],
            "savings": ["savings account", "savings"],
            "loan": ["loan", "lending"],
        }

        for product, keywords in product_keywords.items():
            if any(kw in text_lower for kw in keywords):
                products.append(product)

        return products

    def _extract_issues(self, text: str) -> List[str]:
        """Extract issue mentions"""
        issues = []
        text_lower = text.lower()

        issue_keywords = [
            "access",
            "login",
            "password",
            "fee",
            "charge",
            "error",
            "bug",
            "delay",
            "late",
            "declined",
            "unauthorized",
            "fraud",
            "dispute",
        ]

        for issue in issue_keywords:
            if issue in text_lower:
                issues.append(issue)

        return issues

    def _calculate_importance_score(
        self,
        intent: IntentCategory,
        sentiment: SentimentScore,
        high_keywords: int,
        medium_keywords: int,
        is_customer: bool,
    ) -> float:
        """
        Calculate importance score (0-1).

        THIS IS THE HEART OF THE TRIAGE SYSTEM.

        The formula determines which calls get expensive LLM processing.
        Tune these weights based on:
        - Business priorities
        - Cost constraints
        - Quality requirements
        """
        # Base score
        score = 0.3

        # Intent multipliers
        intent_weights = {
            IntentCategory.COMPLAINT: 0.4,
            IntentCategory.TECHNICAL_ISSUE: 0.3,
            IntentCategory.BILLING_DISPUTE: 0.35,
            IntentCategory.INQUIRY: 0.15,
            IntentCategory.ACCOUNT_MANAGEMENT: 0.1,
            IntentCategory.GENERAL: 0.0,
            IntentCategory.FEEDBACK: 0.05,
        }
        score += intent_weights.get(intent, 0)

        # Sentiment adjustments
        sentiment_weights = {
            SentimentScore.VERY_NEGATIVE: 0.2,
            SentimentScore.NEGATIVE: 0.1,
            SentimentScore.NEUTRAL: 0.0,
            SentimentScore.POSITIVE: -0.05,
            SentimentScore.VERY_POSITIVE: -0.1,
        }
        score += sentiment_weights.get(sentiment, 0)

        # Keyword bonuses (capped)
        score += min(high_keywords * 0.15, 0.3)
        score += min(medium_keywords * 0.05, 0.1)

        # Customer speech is more important than agent speech
        if is_customer:
            score += 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _aggregate_segments(
        self, call_id: str, segments: List[SegmentAnalysis]
    ) -> PreprocessingResult:
        """
        Aggregate segment-level analysis to call level.

        Call importance = weighted average of segment importance
        Higher weight for customer segments
        """
        if not segments:
            raise ValueError("No segments to aggregate")

        # Calculate weighted importance
        customer_segments = [s for s in segments if s.speaker.lower() == "customer"]
        agent_segments = [s for s in segments if s.speaker.lower() == "agent"]

        customer_importance = (
            sum(s.importance_score for s in customer_segments) / len(customer_segments)
            if customer_segments
            else 0.0
        )
        agent_importance = (
            sum(s.importance_score for s in agent_segments) / len(agent_segments)
            if agent_segments
            else 0.0
        )

        # Weight customer speech more heavily
        overall_importance_score = customer_importance * 0.7 + agent_importance * 0.3

        # Determine importance level
        if overall_importance_score >= self.high_priority_threshold:
            importance_level = ImportanceLevel.CRITICAL
        elif overall_importance_score >= self.medium_priority_threshold:
            importance_level = ImportanceLevel.HIGH
        elif overall_importance_score >= 0.2:
            importance_level = ImportanceLevel.MEDIUM
        else:
            importance_level = ImportanceLevel.LOW

        # Overall intent (most severe)
        intent_severity = {
            IntentCategory.COMPLAINT: 3,
            IntentCategory.BILLING_DISPUTE: 3,
            IntentCategory.TECHNICAL_ISSUE: 2,
            IntentCategory.INQUIRY: 1,
            IntentCategory.ACCOUNT_MANAGEMENT: 1,
            IntentCategory.GENERAL: 0,
            IntentCategory.FEEDBACK: 0,
        }
        overall_intent = max(
            segments, key=lambda s: intent_severity.get(s.intent, 0)
        ).intent

        # Overall sentiment (most negative)
        sentiment_severity = {
            SentimentScore.VERY_NEGATIVE: 2,
            SentimentScore.NEGATIVE: 1,
            SentimentScore.NEUTRAL: 0,
            SentimentScore.POSITIVE: -1,
            SentimentScore.VERY_POSITIVE: -2,
        }
        overall_sentiment = max(
            segments, key=lambda s: sentiment_severity.get(s.sentiment, 0)
        ).sentiment

        # Triage decision
        high_priority_segments = sum(1 for s in segments if s.is_critical)
        requires_enrichment = (
            importance_level in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH]
            or high_priority_segments >= 2
        )

        enrichment_reason = None
        if requires_enrichment:
            if importance_level == ImportanceLevel.CRITICAL:
                enrichment_reason = "Critical importance score"
            elif overall_intent == IntentCategory.COMPLAINT:
                enrichment_reason = "Complaint detected"
            elif high_priority_segments >= 2:
                enrichment_reason = f"{high_priority_segments} high-priority segments"

        return PreprocessingResult(
            call_id=call_id,
            segments=segments,
            overall_intent=overall_intent,
            overall_sentiment=overall_sentiment,
            overall_importance=importance_level,
            requires_enrichment=requires_enrichment,
            enrichment_reason=enrichment_reason,
            total_segments=len(segments),
            high_priority_segments=high_priority_segments,
            processing_timestamp=datetime.now(),
        )

    def _store_preprocessing_result(
        self, result: PreprocessingResult, segments: List[SegmentAnalysis]
    ):
        """Store preprocessing results in database"""
        conn = duckdb.connect(str(self.db_path))

        # Store call-level result
        conn.execute(
            """
            INSERT OR REPLACE INTO preprocessed_calls (
                call_id, overall_intent, overall_sentiment, overall_importance,
                importance_score, requires_enrichment, enrichment_reason,
                total_segments, high_priority_segments, processing_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                result.call_id,
                result.overall_intent.value,
                result.overall_sentiment.value,
                result.overall_importance.value,
                sum(s.importance_score for s in segments) / len(segments),
                result.requires_enrichment,
                result.enrichment_reason,
                result.total_segments,
                result.high_priority_segments,
                result.processing_timestamp,
            ],
        )

        # Store segment-level analysis
        for segment in segments:
            conn.execute(
                """
                INSERT OR REPLACE INTO segments (
                    segment_id, call_id, speaker, text, intent, sentiment,
                    importance_score, requires_llm_enrichment, is_critical,
                    high_priority_keywords, products_mentioned, issues_mentioned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    segment.segment_id,
                    result.call_id,
                    segment.speaker,
                    segment.text,
                    segment.intent.value,
                    segment.sentiment.value,
                    segment.importance_score,
                    segment.requires_llm_enrichment,
                    segment.is_critical,
                    json.dumps(segment.high_priority_keywords_found),
                    json.dumps(segment.products_mentioned),
                    json.dumps(segment.issues_mentioned),
                ],
            )

        conn.close()

    async def preprocess_batch(self, call_ids: List[str]) -> List[PreprocessingResult]:
        """Preprocess a batch of calls"""
        results = []

        for call_id in call_ids:
            try:
                result = await self.preprocess_call(call_id)
                results.append(result)
            except Exception as e:
                logger.error("preprocessing_failed", call_id=call_id, error=str(e))
                metrics_collector.increment("preprocessing_errors")

        return results

    def get_calls_requiring_enrichment(self, limit: Optional[int] = None) -> List[str]:
        """Get call IDs that require LLM enrichment"""
        conn = duckdb.connect(str(self.db_path))

        query = """
            SELECT call_id FROM preprocessed_calls
            WHERE requires_enrichment = TRUE
            AND call_id NOT IN (SELECT call_id FROM calls WHERE enriched = TRUE)
        """

        if limit:
            query += f" LIMIT {limit}"

        result = conn.execute(query).fetchall()
        conn.close()

        return [row[0] for row in result]


async def main():
    """Run preprocessing service"""
    service = PreprocessingService()

    # Get unprocessed calls
    from services.ingestion_service import IngestionService

    ingestion = IngestionService()
    unprocessed = ingestion.get_unprocessed_calls(limit=10)

    print(f"Preprocessing {len(unprocessed)} calls...")

    for call in unprocessed:
        result = await service.preprocess_call(call["call_id"])
        print(
            f"  {call['call_id']}: {result.overall_importance.value} "
            f"(enrichment: {result.requires_enrichment})"
        )

        ingestion.mark_processed(call["call_id"])


if __name__ == "__main__":
    asyncio.run(main())
