"""
Data Models and Schemas for Enterprise Call Intelligence Platform
==================================================================

Core data structures used across all services.
Emphasizes:
- Type safety
- Clear separation of concerns
- Metadata for governance and observability
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ClientSegment(str, Enum):
    """Customer segment classification"""

    RETAIL = "retail"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    VIP = "vip"


class ProductCategory(str, Enum):
    """Product categories in financial services"""

    RETIREMENT_FUND = "retirement_fund"
    CREDIT_CARD = "credit_card"
    MORTGAGE = "mortgage"
    ETF = "etf"
    SAVINGS_ACCOUNT = "savings_account"
    LOAN = "loan"
    INSURANCE = "insurance"
    INVESTMENT = "investment"


class IntentCategory(str, Enum):
    """Call intent classification - CRITICAL for importance scoring"""

    COMPLAINT = "complaint"  # High importance
    TECHNICAL_ISSUE = "technical_issue"  # High importance
    BILLING_DISPUTE = "billing_dispute"  # High importance
    INQUIRY = "inquiry"  # Medium importance
    ACCOUNT_MANAGEMENT = "account_management"  # Medium importance
    GENERAL = "general"  # Low importance
    FEEDBACK = "feedback"  # Low importance


class SentimentScore(str, Enum):
    """Sentiment classification"""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ImportanceLevel(str, Enum):
    """Importance level for prioritization - KEY for cost optimization"""

    CRITICAL = "critical"  # Requires immediate attention & LLM enrichment
    HIGH = "high"  # Requires LLM enrichment
    MEDIUM = "medium"  # Selective enrichment
    LOW = "low"  # Analytics only, skip LLM


class QueryType(str, Enum):
    """Query classification for routing"""

    AGGREGATION = "aggregation"  # Route to analytics engine
    SEARCH = "search"  # Route to search + vector DB
    INSIGHT = "insight"  # Route to RAG pipeline
    HYBRID = "hybrid"  # Use multiple approaches


class ProcessingMode(str, Enum):
    """Processing mode for cost control"""

    CHEAP = "cheap"  # Analytics only, no LLM
    BALANCED = "balanced"  # Selective LLM usage
    DEEP = "deep"  # Full RAG with expensive models


# ============================================================
# RAW TRANSCRIPT DATA MODEL
# ============================================================


class SpeakerTurn(BaseModel):
    """Individual speaker turn in a conversation"""

    speaker: str  # "agent" or "customer"
    text: str
    start_time: Optional[float] = None  # Seconds from start
    end_time: Optional[float] = None
    duration: Optional[float] = None


class RawTranscript(BaseModel):
    """Raw call transcript from ingestion"""

    call_id: str
    transcript: List[SpeakerTurn]
    metadata: Dict[str, Any]
    ingestion_timestamp: datetime


# ============================================================
# CALL METADATA & STRUCTURED DATA
# ============================================================


class CallMetadata(BaseModel):
    """Structured metadata for a call - used for analytics"""

    call_id: str
    client_id: str
    client_segment: ClientSegment
    product: ProductCategory
    call_duration_seconds: float
    call_timestamp: datetime
    region: str
    agent_id: Optional[str] = None
    call_outcome: Optional[str] = None  # resolved, escalated, pending


# ============================================================
# PREPROCESSING OUTPUTS - CRITICAL FOR IMPORTANCE SCORING
# ============================================================


class SegmentAnalysis(BaseModel):
    """Analysis of a conversation segment"""

    segment_id: str
    text: str
    speaker: str

    # Importance indicators
    intent: IntentCategory
    sentiment: SentimentScore
    importance_score: float = Field(
        ge=0.0, le=1.0, description="0-1 score indicating importance"
    )

    # Keyword detection
    high_priority_keywords_found: List[str] = []
    medium_priority_keywords_found: List[str] = []

    # Entity extraction (cheap NLP)
    products_mentioned: List[str] = []
    issues_mentioned: List[str] = []

    # Flags for triage
    requires_llm_enrichment: bool
    is_critical: bool


class PreprocessingResult(BaseModel):
    """Complete preprocessing output for a call"""

    call_id: str

    # Segment-level analysis
    segments: List[SegmentAnalysis]

    # Call-level aggregations
    overall_intent: IntentCategory
    overall_sentiment: SentimentScore
    overall_importance: ImportanceLevel

    # Triage decision - CRITICAL for cost optimization
    requires_enrichment: bool
    enrichment_reason: Optional[str] = None

    # Statistics
    total_segments: int
    high_priority_segments: int
    processing_timestamp: datetime

    model_config = ConfigDict(use_enum_values=False)


# ============================================================
# ENRICHMENT OUTPUTS - EXPENSIVE LLM OPERATIONS
# ============================================================


class EnrichmentResult(BaseModel):
    """LLM-generated enrichment for important calls/segments"""

    call_id: str
    segment_id: Optional[str] = None

    # LLM outputs
    summary: str
    key_issues: List[str]
    action_items: List[str]
    semantic_tags: List[str]

    # Metadata
    model_used: str
    tokens_used: int
    cost_usd: float
    enrichment_timestamp: datetime
    confidence_score: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(protected_namespaces=())


# ============================================================
# SEARCH & RETRIEVAL
# ============================================================


class SearchResult(BaseModel):
    """Search result from keyword or vector search"""

    call_id: str
    segment_id: Optional[str] = None
    text: str
    score: float
    metadata: Dict[str, Any]

    # For hybrid search
    keyword_score: Optional[float] = None
    semantic_score: Optional[float] = None
    combined_score: Optional[float] = None


class RetrievalContext(BaseModel):
    """Retrieved context for RAG"""

    results: List[SearchResult]
    total_retrieved: int
    retrieval_method: str  # "keyword", "vector", "hybrid"
    retrieval_timestamp: datetime


# ============================================================
# QUERY PROCESSING
# ============================================================


class UserQuery(BaseModel):
    """User query input"""

    query_text: str
    query_id: Optional[str] = None
    processing_mode: ProcessingMode = ProcessingMode.BALANCED

    # Filters
    filters: Optional[Dict[str, Any]] = None
    date_range: Optional[Dict[str, datetime]] = None

    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class QueryClassification(BaseModel):
    """Query classification result for routing"""

    query_id: str
    query_type: QueryType
    confidence: float

    # Routing decision
    route_to_analytics: bool
    route_to_search: bool
    route_to_rag: bool

    # Extracted intent
    intent: str
    entities: Dict[str, List[str]]


class QueryResponse(BaseModel):
    """Final response to user query"""

    query_id: str
    query_text: str

    # Answer
    answer: str
    confidence: float

    # Evidence
    sources: List[SearchResult]
    citations: List[str]

    # Analytics (if applicable)
    analytics: Optional[Dict[str, Any]] = None

    # Cost & Performance
    processing_time_ms: float
    total_cost_usd: float
    model_used: Optional[str] = None

    # Metadata
    query_type: QueryType
    processing_mode: ProcessingMode
    timestamp: datetime

    model_config = ConfigDict(protected_namespaces=())


# ============================================================
# ANALYTICS MODELS
# ============================================================


class TrendData(BaseModel):
    """Time series trend data"""

    metric_name: str
    time_period: str
    data_points: List[Dict[str, Any]]


class CategoryCount(BaseModel):
    """Category aggregation"""

    category: str
    count: int
    percentage: float


class AnalyticsSummary(BaseModel):
    """Precomputed analytics summary"""

    total_calls: int
    date_range: Dict[str, datetime]

    top_complaints: List[CategoryCount]
    top_products: List[CategoryCount]
    sentiment_distribution: Dict[str, int]

    trends: List[TrendData]

    computed_at: datetime


# ============================================================
# OBSERVABILITY & GOVERNANCE
# ============================================================


class CostMetrics(BaseModel):
    """Cost tracking for a request"""

    request_id: str

    # LLM costs
    llm_calls: int
    total_tokens: int
    llm_cost_usd: float

    # Other costs (simulated)
    vector_search_cost_usd: float
    storage_cost_usd: float

    total_cost_usd: float
    timestamp: datetime


class ProcessingTrace(BaseModel):
    """End-to-end trace for observability"""

    trace_id: str
    request_type: str

    # Stages
    stages: List[Dict[str, Any]]

    # Performance
    total_duration_ms: float

    # Cost
    cost_metrics: CostMetrics

    # Quality
    confidence_score: Optional[float] = None

    timestamp: datetime


class PIIDetectionResult(BaseModel):
    """PII detection result for governance"""

    text_id: str
    contains_pii: bool
    pii_types_found: List[str]
    masked_text: Optional[str] = None
