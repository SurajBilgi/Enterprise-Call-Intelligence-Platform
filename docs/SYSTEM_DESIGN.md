# System Design: Enterprise Call Intelligence Platform

**Interview-Ready System Design Document**

*This document walks through the complete system design process as you would present it in a staff-level interview.*

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Requirements Gathering](#2-requirements-gathering)
3. [Capacity Estimation](#3-capacity-estimation)
4. [High-Level Design](#4-high-level-design)
5. [Detailed Component Design](#5-detailed-component-design)
6. [Data Model Design](#6-data-model-design)
7. [API Design](#7-api-design)
8. [Algorithm Deep Dive](#8-algorithm-deep-dive)
9. [Scalability & Performance](#9-scalability--performance)
10. [Reliability & Fault Tolerance](#10-reliability--fault-tolerance)
11. [Security & Compliance](#11-security--compliance)
12. [Monitoring & Observability](#12-monitoring--observability)
13. [Cost Optimization](#13-cost-optimization)
14. [Trade-offs & Alternatives](#14-trade-offs--alternatives)
15. [Future Enhancements](#15-future-enhancements)

---

## 1. Problem Statement

### The Core Challenge

**Business Context:**
A large enterprise receives **millions of customer support phone calls** monthly. They need to:
- Understand what customers are complaining about
- Identify trending issues quickly
- Get insights from natural language queries
- Do this cost-effectively (GenAI APIs are expensive)

**Key Question:** *"Which calls are important, and which parts of speech matter most?"*

### Why This is Hard

1. **Scale:** Processing 1M calls/month with LLMs = $50,000+ (prohibitively expensive)
2. **Latency:** Users expect answers in seconds, not minutes
3. **Variety:** Queries range from simple counts to complex semantic analysis
4. **Quality:** Answers must be accurate, verifiable, and cite sources

---

## 2. Requirements Gathering

### Functional Requirements

**Core Use Cases:**
1. **Natural Language Queries**
   - "How many calls about product X?"
   - "What are customers complaining about?"
   - "Show me trends for the last 30 days"

2. **Multi-Modal Query Support**
   - Aggregation queries (counts, percentages)
   - Search queries (find specific calls)
   - Insight queries (deep semantic analysis)

3. **Evidence-Based Answers**
   - Show source call transcripts
   - Provide confidence scores
   - Enable verification

### Non-Functional Requirements

**Performance:**
- Query latency: <100ms (analytics), <5s (insights)
- Ingestion throughput: 10K calls/hour
- Support concurrent queries: 100+ QPS

**Scalability:**
- Handle 1M+ calls/month
- Linear scaling with data volume
- Geographic distribution support

**Cost:**
- Budget: <$10K/month for 1M calls
- 90%+ cost reduction vs naive approach
- Transparent cost tracking

**Reliability:**
- 99.9% uptime
- Data durability: 99.999999999%
- Graceful degradation

### Non-Goals (Out of Scope)

- Real-time voice transcription
- Sentiment analysis of audio tone
- Multi-language support (Phase 1)
- Call routing/IVR integration

---

## 3. Capacity Estimation

### Traffic Estimates

**Assumptions:**
- 1M calls/month = ~33K calls/day
- Average call duration: 5 minutes
- Average transcript length: 1500 words = ~2000 tokens
- Query load: 10K queries/day = ~0.1 QPS avg, ~10 QPS peak

### Storage Estimates

**Per Call:**
- Raw transcript JSON: ~10 KB
- Metadata + preprocessing: ~2 KB
- Embeddings (768d): ~3 KB
- Total: ~15 KB/call

**For 1M calls:**
- Raw data: 10 GB
- Structured data: 2 GB
- Vector embeddings: 3 GB
- Total: ~15 GB/month
- With 12 months retention: ~180 GB

### Bandwidth Estimates

**Ingestion:**
- 33K calls/day × 10 KB = 330 MB/day
- Peak: 100 calls/min × 10 KB = 1 MB/min

**Query:**
- 10K queries/day × 50 KB response = 500 MB/day
- Negligible compared to ingestion

### Cost Estimates

**Without Optimization (Naive Approach):**
```
LLM Processing: 1M calls × $0.05 = $50,000/month
Storage: ~$5/month (S3)
Compute: ~$500/month (EC2)
Total: ~$50,500/month
```

**With Optimization (Our Approach):**
```
Preprocessing: 1M calls × $0 = $0
LLM (15% of calls): 150K × $0.003 = $450/month
Embeddings (local): ~$100/month (compute)
Storage: ~$5/month
Compute: ~$500/month
Total: ~$1,055/month

SAVINGS: 97.9% ($49,445/month)
```

---

## 4. High-Level Design

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Web UI     │  │  Mobile App  │  │  API Clients │          │
│  │  (Streamlit) │  │   (Future)   │  │   (REST)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Gateway (Load Balanced)                         │   │
│  │  - Authentication & Authorization                        │   │
│  │  - Rate Limiting                                         │   │
│  │  - Request Validation                                    │   │
│  │  - Response Caching                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   QUERY PROCESSING LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             QUERY ROUTER (Intelligent Routing)           │   │
│  │  - Classifies query type                                 │   │
│  │  - Routes to appropriate service                         │   │
│  │  - Cost-aware decision making                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│          │                    │                    │             │
│          ▼                    ▼                    ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  ANALYTICS   │  │    SEARCH    │  │  RAG SERVICE │         │
│  │   SERVICE    │  │   SERVICE    │  │   (GenAI)    │         │
│  │              │  │              │  │              │         │
│  │ • SQL Aggs   │  │ • Hybrid     │  │ • Retrieval  │         │
│  │ • Metrics    │  │   Search     │  │ • Synthesis  │         │
│  │ • Fast       │  │ • Filters    │  │ • Citations  │         │
│  │              │  │              │  │              │         │
│  │ Cost: $0     │  │ Cost: $0.001 │  │ Cost: $0.05  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   DuckDB     │  │    FAISS     │  │      S3      │         │
│  │ (Structured) │  │  (Vectors)   │  │  (Raw Data)  │         │
│  │              │  │              │  │              │         │
│  │ • Analytics  │  │ • Semantic   │  │ • Transcripts│         │
│  │ • Metadata   │  │   Search     │  │ • Backups    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │    Redis     │  │  TF-IDF      │                           │
│  │   (Cache)    │  │  (Keyword)   │                           │
│  └──────────────┘  └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 OFFLINE PROCESSING PIPELINE                      │
│                                                                  │
│  ┌───────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐     │
│  │Ingest │──▶│Preprocess│──▶│  Enrich    │──▶│  Index   │     │
│  │       │   │(Triage)  │   │(Selective) │   │(Multi)   │     │
│  └───────┘   └──────────┘   └────────────┘   └──────────┘     │
│                    │                                             │
│                    └──────▶ IMPORTANCE SCORING ◀────────────    │
│                            (The Key Innovation)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Tiered Processing Architecture**
- **Why:** Not all queries need the same resources
- **How:** Route based on query complexity
- **Impact:** 80% cost reduction on query processing

**2. Importance-Based Triage**
- **Why:** Can't afford LLM on every call
- **How:** Cheap NLP scores importance first
- **Impact:** 85% cost reduction on enrichment

**3. Hybrid Search**
- **Why:** No single search method is optimal
- **How:** Combine keyword + semantic search
- **Impact:** Better recall and precision

**4. Separation of Batch vs Real-time**
- **Why:** Different SLAs and resource needs
- **How:** Offline pipeline + online API
- **Impact:** Better resource utilization

---

## 5. Detailed Component Design

### 5.1 Ingestion Service

**Responsibility:** Load and validate raw transcripts

**Design:**
```python
class IngestionService:
    """
    ETL Pattern: Extract → Transform → Load
    
    Input:  S3 events (new transcript files)
    Output: Validated records in DuckDB
    """
    
    def ingest_batch(self, file_paths: List[str]):
        # 1. Extract
        transcripts = self.read_from_s3(file_paths)
        
        # 2. Transform
        validated = self.validate_and_normalize(transcripts)
        
        # 3. Load
        self.bulk_insert_to_db(validated)
```

**Key Decisions:**
- **Batch Processing:** Handle 100 files at once
- **Idempotency:** Use UPSERT to handle duplicates
- **Async I/O:** Non-blocking file operations

**Scalability:**
- Horizontal: Multiple workers consume from SQS
- Vertical: Increase batch size
- Checkpointing: Resume from failure

---

### 5.2 Preprocessing Service ⭐ **MOST CRITICAL**

**Responsibility:** Score importance WITHOUT using LLM

**The Core Algorithm:**

```python
def calculate_importance_score(segment) -> float:
    """
    Multi-Signal Importance Scoring
    
    Combines:
    - Intent classification (complaint > inquiry > general)
    - Sentiment analysis (negative > neutral > positive)
    - Keyword detection (high-priority terms)
    - Speaker role (customer > agent)
    
    Returns: 0.0 to 1.0 (higher = more important)
    """
    score = 0.3  # Base score
    
    # Signal 1: Intent (40% weight)
    if intent == COMPLAINT:
        score += 0.4
    elif intent == TECHNICAL_ISSUE:
        score += 0.3
    elif intent == BILLING_DISPUTE:
        score += 0.35
    elif intent == INQUIRY:
        score += 0.15
    
    # Signal 2: Sentiment (20% weight)
    if sentiment == VERY_NEGATIVE:
        score += 0.2
    elif sentiment == NEGATIVE:
        score += 0.1
    
    # Signal 3: Keywords (30% weight)
    # High-priority: refund, cancel, lawsuit, fraud, error
    score += min(num_high_priority_keywords * 0.15, 0.3)
    
    # Signal 4: Speaker (10% weight)
    if speaker == CUSTOMER:
        score += 0.1
    
    return clamp(score, 0.0, 1.0)
```

**Intent Classification (Rule-Based):**
```python
def classify_intent(text: str) -> IntentCategory:
    """
    Pattern matching for intent classification
    
    Why rule-based instead of ML?
    - 80% accuracy is sufficient for triage
    - Zero latency (no API calls)
    - Zero cost
    - Interpretable and debuggable
    """
    if re.search(r'\b(complaint|terrible|awful|refund)\b', text):
        return COMPLAINT
    
    if re.search(r'\b(error|bug|broken|not working)\b', text):
        return TECHNICAL_ISSUE
    
    if re.search(r'\b(how|what|question|help)\b', text):
        return INQUIRY
    
    return GENERAL
```

**Triage Decision:**
```python
def make_triage_decision(importance_score: float) -> bool:
    """
    Decide if call needs expensive LLM enrichment
    
    Thresholds tuned based on:
    - Cost budget
    - Quality requirements
    - Historical data
    """
    HIGH_THRESHOLD = 0.7    # Critical issues
    MEDIUM_THRESHOLD = 0.4  # Important issues
    
    if importance_score >= HIGH_THRESHOLD:
        return True, "gpt-4"  # Use expensive model
    
    elif importance_score >= MEDIUM_THRESHOLD:
        return True, "gpt-3.5-turbo"  # Use cheap model
    
    else:
        return False, None  # Skip LLM
```

**Why This Approach?**

| Approach | Accuracy | Latency | Cost | Pros | Cons |
|----------|----------|---------|------|------|------|
| Rule-based (Ours) | 80% | 50ms | $0 | Fast, cheap, debuggable | Lower accuracy |
| Small ML model | 90% | 100ms | $0 | Better accuracy | Needs training |
| LLM classification | 95% | 2s | $0.001 | Best accuracy | Expensive, slow |

For **triage**, 80% accuracy is sufficient because:
- False positives: We enrich a call unnecessarily (~$0.003 wasted)
- False negatives: We miss enriching an important call (acceptable <20% miss rate)

---

### 5.3 Enrichment Service

**Responsibility:** Selective LLM usage for important calls

**Design Pattern:** Cost-Aware Processing

```python
async def enrich_call(call_id: str) -> EnrichmentResult:
    """
    Selective enrichment based on importance
    
    Flow:
    1. Check preprocessing result
    2. Skip if not important
    3. Select appropriate model
    4. Generate enrichment
    5. Track cost
    """
    # Load preprocessing
    prep = load_preprocessing(call_id)
    
    if not prep.requires_enrichment:
        return create_minimal_enrichment()  # No LLM, $0
    
    # Model selection
    if prep.importance == CRITICAL:
        model = "gpt-4"  # $0.01/1K input
    else:
        model = "gpt-3.5-turbo"  # $0.0015/1K input
    
    # Generate enrichment (multiple tasks)
    summary = await llm.generate(summarize_prompt)  # ~100 tokens
    issues = await llm.generate(extract_issues_prompt)  # ~50 tokens
    actions = await llm.generate(action_items_prompt)  # ~50 tokens
    tags = await llm.generate(tagging_prompt)  # ~30 tokens
    
    # Track cost
    total_cost = calculate_cost(total_tokens, model)
    cost_tracker.log(call_id, total_cost)
    
    return EnrichmentResult(...)
```

**Batch Optimization:**
```python
async def enrich_batch(call_ids: List[str]):
    """
    Batch processing with rate limiting
    
    Considerations:
    - OpenAI rate limits: 3,500 RPM
    - Cost burn rate: Monitor and throttle
    - Error handling: Retry with backoff
    """
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
    
    async def process_with_limit(call_id):
        async with semaphore:
            return await enrich_call(call_id)
    
    tasks = [process_with_limit(cid) for cid in call_ids]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

---

### 5.4 Indexing Service

**Responsibility:** Build multi-modal search infrastructure

**Three Indices:**

**1. Vector Index (FAISS)**
```python
class VectorStore:
    """
    Semantic similarity search using embeddings
    
    Why FAISS?
    - Handles billions of vectors
    - Fast: <10ms for 1M vectors
    - Memory efficient
    - GPU support
    """
    
    def __init__(self, dimension=384):
        # L2 index with normalization = cosine similarity
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_vectors(self, embeddings, metadata):
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, query_vector, top_k=10):
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        
        # Convert L2 distance to similarity (0-1)
        similarities = 1 - (distances / 2)
        return similarities, indices
```

**2. Keyword Index (TF-IDF)**
```python
class KeywordIndex:
    """
    Exact term matching using TF-IDF
    
    Why TF-IDF?
    - Fast: <5ms for 1M documents
    - Works for product names, IDs
    - No API costs
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2)  # Unigrams + bigrams
        )
    
    def build_index(self, documents):
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def search(self, query, top_k=10):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        return top_k_indices(similarities, top_k)
```

**3. Structured Index (DuckDB)**
```sql
-- Optimized for OLAP queries
CREATE TABLE calls (
    call_id VARCHAR PRIMARY KEY,
    product VARCHAR,
    timestamp TIMESTAMP,
    importance_score DOUBLE,
    -- ... more columns
);

-- Column-oriented indexes
CREATE INDEX idx_product ON calls(product);
CREATE INDEX idx_timestamp ON calls(timestamp);
CREATE INDEX idx_importance ON calls(importance_score);
```

**Hybrid Search Algorithm:**
```python
def hybrid_search(query: str, top_k: int = 10):
    """
    Combine keyword + semantic search
    
    Why hybrid?
    - Keyword: Good for exact matches (product names)
    - Semantic: Good for conceptual matches
    - Combined: Best of both worlds
    """
    # Parallel search
    keyword_results = keyword_index.search(query, top_k * 2)
    semantic_results = vector_store.search(embed(query), top_k * 2)
    
    # Merge and rerank
    combined = {}
    for result in keyword_results:
        combined[result.id] = {
            'keyword_score': result.score,
            'semantic_score': 0.0,
            'data': result
        }
    
    for result in semantic_results:
        if result.id in combined:
            combined[result.id]['semantic_score'] = result.score
        else:
            combined[result.id] = {
                'keyword_score': 0.0,
                'semantic_score': result.score,
                'data': result
            }
    
    # Weighted combination
    for item in combined.values():
        item['final_score'] = (
            item['keyword_score'] * 0.3 +
            item['semantic_score'] * 0.7
        )
    
    # Sort and return top-k
    return sorted(combined.values(), 
                 key=lambda x: x['final_score'], 
                 reverse=True)[:top_k]
```

---

### 5.5 Query Router

**Responsibility:** Classify and route queries intelligently

**Classification Logic:**

```python
class QueryRouter:
    """
    Pattern-based query classification
    
    Three query types:
    1. AGGREGATION: "How many?" → SQL ($0)
    2. SEARCH: "Find calls..." → Vector search ($0.001)
    3. INSIGHT: "What are customers saying?" → RAG ($0.05)
    """
    
    AGGREGATION_PATTERNS = [
        r'\bhow many\b',
        r'\bcount\b',
        r'\bpercentage\b',
        r'\btop\b',
        r'\btrend\b'
    ]
    
    SEARCH_PATTERNS = [
        r'\bfind\b',
        r'\bshow me\b',
        r'\blist\b',
        r'\bwhich calls\b'
    ]
    
    INSIGHT_PATTERNS = [
        r'\bwhat are.*saying\b',
        r'\bsummarize\b',
        r'\bwhy\b',
        r'\bexplain\b'
    ]
    
    def classify(self, query: str) -> QueryType:
        query_lower = query.lower()
        
        # Score each type
        agg_score = sum(1 for p in self.AGGREGATION_PATTERNS 
                       if re.search(p, query_lower))
        search_score = sum(1 for p in self.SEARCH_PATTERNS 
                          if re.search(p, query_lower))
        insight_score = sum(1 for p in self.INSIGHT_PATTERNS 
                           if re.search(p, query_lower))
        
        # Return highest scoring type
        scores = {
            'aggregation': agg_score,
            'search': search_score,
            'insight': insight_score
        }
        
        return max(scores, key=scores.get)
```

**Routing Decision Tree:**

```
User Query
    │
    ▼
Is it counting/aggregation?
    │
    ├─YES─▶ Analytics Service
    │       • Use SQL
    │       • Return in <100ms
    │       • Cost: $0
    │
    └─NO───▶ Contains specific entities?
            │
            ├─YES─▶ Search Service
            │       • Use hybrid search
            │       • Return in <500ms
            │       • Cost: $0.001
            │
            └─NO───▶ RAG Service
                    • Full pipeline
                    • Return in 2-5s
                    • Cost: $0.02-0.10
```

---

### 5.6 RAG Service

**Responsibility:** Deep insights with LLM synthesis

**RAG Pipeline:**

```python
async def rag_query(query: str) -> QueryResponse:
    """
    Retrieval-Augmented Generation Pipeline
    
    5 Stages:
    1. Query Understanding
    2. Retrieval
    3. Re-ranking
    4. Context Construction
    5. LLM Synthesis
    """
    
    # Stage 1: Query Understanding
    # (Extract intent, entities, filters)
    query_intent = extract_intent(query)
    
    # Stage 2: Retrieval (Hybrid Search)
    candidates = hybrid_search(
        query=query,
        top_k=20,  # Over-retrieve
        filters=query_intent.filters
    )
    
    # Stage 3: Re-ranking
    # (Optional: Use cross-encoder for better ranking)
    reranked = rerank(query, candidates, top_k=5)
    
    # Stage 4: Context Construction
    context = construct_context(reranked, max_tokens=4000)
    
    # Stage 5: LLM Synthesis
    prompt = f"""You are analyzing customer support calls.

User Question: {query}

Relevant Call Transcripts:
{context}

Instructions:
1. Answer based ONLY on provided transcripts
2. Cite specific calls for evidence
3. If information is insufficient, say so
4. Be specific and actionable

Answer:"""
    
    response = await llm.generate(
        prompt=prompt,
        model="gpt-3.5-turbo",  # Or gpt-4 for critical
        max_tokens=500
    )
    
    # Build response with citations
    return QueryResponse(
        answer=response.text,
        sources=reranked,
        citations=build_citations(reranked),
        confidence=calculate_confidence(reranked),
        cost=response.cost
    )
```

**Context Construction:**

```python
def construct_context(results: List[SearchResult], 
                     max_tokens: int = 4000) -> str:
    """
    Build context for LLM within token limits
    
    Considerations:
    - Token limits (4K for GPT-3.5, 8K for GPT-4)
    - Relevant information density
    - Clear formatting
    """
    context_parts = []
    current_tokens = 0
    
    for i, result in enumerate(results):
        segment = f"""
[Call {i+1}]
Product: {result.metadata['product']}
Sentiment: {result.metadata['sentiment']}
Importance: {result.metadata['importance_score']:.2f}

Transcript:
{result.text}

---
"""
        
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        segment_tokens = len(segment) // 4
        
        if current_tokens + segment_tokens > max_tokens:
            break
        
        context_parts.append(segment)
        current_tokens += segment_tokens
    
    return "\n".join(context_parts)
```

---

### 5.7 Analytics Service

**Responsibility:** Fast aggregations without LLM

**SQL-Based Analytics:**

```python
class AnalyticsService:
    """
    OLAP queries on structured data
    
    Why not use LLM for this?
    - SQL is faster (10-100ms vs 2-5s)
    - SQL is free (no API costs)
    - SQL is deterministic (no hallucinations)
    """
    
    def get_call_counts_by_product(self):
        return self.db.query("""
            SELECT 
                product,
                COUNT(*) as call_count,
                AVG(importance_score) as avg_importance
            FROM calls
            GROUP BY product
            ORDER BY call_count DESC
        """)
    
    def get_trend_over_time(self, window='daily'):
        return self.db.query(f"""
            SELECT 
                DATE_TRUNC('{window}', call_timestamp) as period,
                COUNT(*) as call_count,
                AVG(importance_score) as avg_importance
            FROM calls
            WHERE call_timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY period
            ORDER BY period
        """)
```

**Natural Language to SQL (Simple):**

```python
def nl_to_sql(query: str) -> str:
    """
    Simple pattern matching for common queries
    
    For production, consider:
    - LangChain SQL agent
    - Text-to-SQL model
    - Pre-defined query templates
    """
    query_lower = query.lower()
    
    # "How many calls about X?"
    if match := re.search(r'how many.*about (\w+)', query_lower):
        product = match.group(1)
        return f"""
            SELECT COUNT(*) 
            FROM calls 
            WHERE LOWER(product) LIKE '%{product}%'
        """
    
    # "What are the top complaints?"
    if 'top complaint' in query_lower:
        return """
            SELECT overall_intent, COUNT(*) as count
            FROM preprocessed_calls
            WHERE overall_intent IN ('complaint', 'technical_issue')
            GROUP BY overall_intent
            ORDER BY count DESC
            LIMIT 10
        """
    
    return None  # Fall back to RAG
```

---

## 6. Data Model Design

### 6.1 Raw Transcript Storage

**Format:** JSON in S3

```json
{
  "call_id": "CALL_123456",
  "timestamp": "2024-01-15T14:30:00Z",
  "duration_seconds": 342,
  "transcript": [
    {
      "speaker": "customer",
      "text": "I've been trying to cancel my subscription...",
      "start_time": 0.0,
      "end_time": 5.2
    },
    {
      "speaker": "agent",
      "text": "I understand. Let me help you with that...",
      "start_time": 5.5,
      "end_time": 8.1
    }
  ],
  "metadata": {
    "client_id": "CLIENT_789",
    "product": "subscription_service",
    "region": "US-West",
    "agent_id": "AGT_456"
  }
}
```

### 6.2 Structured Database (DuckDB/Snowflake)

**Calls Table:**
```sql
CREATE TABLE calls (
    call_id VARCHAR PRIMARY KEY,
    client_id VARCHAR,
    client_segment VARCHAR,  -- retail, premium, enterprise
    product VARCHAR,
    call_duration_seconds DOUBLE,
    call_timestamp TIMESTAMP,
    region VARCHAR,
    agent_id VARCHAR,
    
    -- Processing flags
    processed BOOLEAN DEFAULT FALSE,
    preprocessed BOOLEAN DEFAULT FALSE,
    enriched BOOLEAN DEFAULT FALSE,
    
    -- Indexes for common queries
    INDEX idx_timestamp(call_timestamp),
    INDEX idx_product(product),
    INDEX idx_segment(client_segment)
);
```

**Preprocessing Results:**
```sql
CREATE TABLE preprocessed_calls (
    call_id VARCHAR PRIMARY KEY,
    
    -- Aggregated scores
    overall_intent VARCHAR,
    overall_sentiment VARCHAR,
    overall_importance VARCHAR,  -- critical, high, medium, low
    importance_score DOUBLE,     -- 0.0 to 1.0
    
    -- Triage decision
    requires_enrichment BOOLEAN,
    enrichment_reason VARCHAR,
    
    -- Statistics
    total_segments INTEGER,
    high_priority_segments INTEGER,
    
    processing_timestamp TIMESTAMP
);
```

**Segments Table:**
```sql
CREATE TABLE segments (
    segment_id VARCHAR PRIMARY KEY,
    call_id VARCHAR,
    
    -- Segment data
    speaker VARCHAR,
    text VARCHAR,
    
    -- Analysis
    intent VARCHAR,
    sentiment VARCHAR,
    importance_score DOUBLE,
    
    -- Flags
    requires_llm_enrichment BOOLEAN,
    is_critical BOOLEAN,
    
    -- Extracted features
    high_priority_keywords JSON,
    products_mentioned JSON,
    issues_mentioned JSON,
    
    INDEX idx_call(call_id),
    INDEX idx_importance(importance_score)
);
```

**Enrichments Table:**
```sql
CREATE TABLE enrichments (
    enrichment_id VARCHAR PRIMARY KEY,
    call_id VARCHAR,
    
    -- LLM outputs
    summary TEXT,
    key_issues JSON,
    action_items JSON,
    semantic_tags JSON,
    
    -- Metadata
    model_used VARCHAR,
    tokens_used INTEGER,
    cost_usd DOUBLE,
    enrichment_timestamp TIMESTAMP,
    confidence_score DOUBLE,
    
    INDEX idx_call(call_id)
);
```

### 6.3 Vector Storage (FAISS/Pinecone)

**Structure:**
```python
{
    "vector_id": "CALL_123_SEG_001",
    "embedding": [0.123, -0.456, ...],  # 384 or 768 dimensions
    "metadata": {
        "call_id": "CALL_123",
        "segment_id": "SEG_001",
        "text": "Original segment text...",
        "importance_score": 0.75,
        "product": "subscription_service"
    }
}
```

---

## 7. API Design

### 7.1 Query Endpoint

```http
POST /query
Content-Type: application/json

{
  "query": "What are customers complaining about?",
  "mode": "balanced",  // cheap | balanced | deep
  "filters": {
    "product": "subscription_service",
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-31"
    },
    "importance_min": 0.5
  }
}

Response:
{
  "query_id": "QRY_789",
  "answer": "Based on 47 relevant calls, customers are primarily concerned about...",
  "confidence": 0.85,
  "sources": [
    {
      "call_id": "CALL_123",
      "segment_id": "SEG_002",
      "text": "I've been trying to cancel...",
      "relevance_score": 0.92,
      "metadata": {...}
    }
  ],
  "citations": [
    "[1] Call CALL_123 - subscription_service - Score: 0.92",
    "[2] Call CALL_456 - subscription_service - Score: 0.88"
  ],
  "analytics": {
    "total_matching_calls": 47,
    "sentiment_distribution": {
      "negative": 32,
      "neutral": 10,
      "positive": 5
    }
  },
  "processing": {
    "query_type": "insight",
    "processing_mode": "balanced",
    "duration_ms": 2840,
    "cost_usd": 0.0342,
    "model_used": "gpt-3.5-turbo"
  },
  "timestamp": "2024-01-15T14:35:22Z"
}
```

### 7.2 Analytics Endpoint

```http
GET /analytics/summary?start_date=2024-01-01&end_date=2024-01-31

Response:
{
  "total_calls": 45230,
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  },
  "top_complaints": [
    {"category": "technical_issue", "count": 3421, "percentage": 7.6},
    {"category": "billing_dispute", "count": 2145, "percentage": 4.7}
  ],
  "top_products": [
    {"product": "subscription_service", "count": 12340, "percentage": 27.3},
    {"product": "mobile_app", "count": 8920, "percentage": 19.7}
  ],
  "sentiment_distribution": {
    "very_negative": 2341,
    "negative": 5623,
    "neutral": 28456,
    "positive": 7234,
    "very_positive": 1576
  },
  "importance_distribution": {
    "critical": 1234,
    "high": 4567,
    "medium": 12345,
    "low": 27084
  }
}
```

### 7.3 Call Details Endpoint

```http
GET /calls/{call_id}

Response:
{
  "call_id": "CALL_123",
  "metadata": {
    "timestamp": "2024-01-15T10:23:45Z",
    "duration_seconds": 342,
    "product": "subscription_service",
    "client_segment": "premium"
  },
  "preprocessing": {
    "overall_intent": "complaint",
    "overall_sentiment": "negative",
    "overall_importance": "high",
    "importance_score": 0.78,
    "requires_enrichment": true
  },
  "enrichment": {
    "summary": "Customer frustrated with cancellation process...",
    "key_issues": [
      "Difficulty finding cancellation option",
      "Multiple required steps",
      "No confirmation email"
    ],
    "action_items": [
      "Simplify cancellation flow",
      "Add confirmation emails",
      "Provide clear instructions"
    ],
    "cost_usd": 0.0034
  },
  "segments": [
    {
      "speaker": "customer",
      "text": "I've been trying to cancel...",
      "importance_score": 0.82,
      "intent": "complaint",
      "sentiment": "negative"
    }
  ]
}
```

---

## 8. Algorithm Deep Dive

### 8.1 Importance Scoring Algorithm

**Mathematical Formulation:**

```
importance_score(s) = clamp(Σ w_i × f_i(s), 0, 1)

where:
  s = segment
  w_i = weight for feature i
  f_i(s) = feature function i
  
Features:
  f_intent(s) = intent_score(s)           w = 0.4
  f_sentiment(s) = sentiment_score(s)     w = 0.2
  f_keywords(s) = keyword_score(s)        w = 0.3
  f_speaker(s) = speaker_score(s)         w = 0.1
```

**Feature Functions:**

```python
def intent_score(segment) -> float:
    """Map intent to score"""
    scores = {
        'complaint': 1.0,
        'technical_issue': 0.85,
        'billing_dispute': 0.9,
        'inquiry': 0.4,
        'general': 0.0
    }
    return scores.get(segment.intent, 0.0)

def sentiment_score(segment) -> float:
    """Map sentiment to score"""
    scores = {
        'very_negative': 1.0,
        'negative': 0.5,
        'neutral': 0.0,
        'positive': -0.25,
        'very_positive': -0.5
    }
    return scores.get(segment.sentiment, 0.0)

def keyword_score(segment) -> float:
    """Score based on keyword matches"""
    high_priority = segment.high_priority_keywords_found
    medium_priority = segment.medium_priority_keywords_found
    
    # Capped contribution
    return min(len(high_priority) * 0.5, 1.0) + \
           min(len(medium_priority) * 0.25, 0.5)

def speaker_score(segment) -> float:
    """Customer speech weighted higher"""
    return 1.0 if segment.speaker == 'customer' else 0.0
```

**Threshold Tuning:**

```python
# Confusion Matrix Analysis
def evaluate_thresholds(predictions, ground_truth):
    """
    Tune thresholds based on business requirements
    
    Metrics:
    - Precision: Of calls we enrich, how many are actually important?
    - Recall: Of important calls, how many do we catch?
    - Cost: How much do we spend?
    """
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        flagged = predictions >= threshold
        
        precision = precision_score(ground_truth, flagged)
        recall = recall_score(ground_truth, flagged)
        enrichment_rate = flagged.sum() / len(flagged)
        
        cost = enrichment_rate * LLM_COST_PER_CALL
        
        print(f"Threshold: {threshold:.1f}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  Enrichment Rate: {enrichment_rate:.2%}")
        print(f"  Cost (1M calls): ${cost * 1_000_000:.2f}")
        print()

# Example output:
# Threshold: 0.4
#   Precision: 82%
#   Recall: 95%
#   Enrichment Rate: 18%
#   Cost (1M calls): $9,000.00
```

---

### 8.2 Hybrid Search Algorithm

**Reciprocal Rank Fusion:**

```python
def reciprocal_rank_fusion(
    keyword_results: List[Result],
    semantic_results: List[Result],
    k: int = 60
) -> List[Result]:
    """
    Combine two ranked lists using RRF
    
    RRF score = Σ 1 / (k + rank_i)
    
    Why RRF?
    - Doesn't require score normalization
    - Robust to score scale differences
    - Well-studied in IR literature
    """
    rrf_scores = {}
    
    # Add keyword scores
    for rank, result in enumerate(keyword_results):
        rrf_scores[result.id] = 1 / (k + rank + 1)
    
    # Add semantic scores
    for rank, result in enumerate(semantic_results):
        if result.id in rrf_scores:
            rrf_scores[result.id] += 1 / (k + rank + 1)
        else:
            rrf_scores[result.id] = 1 / (k + rank + 1)
    
    # Sort by RRF score
    return sorted(rrf_scores.items(), 
                 key=lambda x: x[1], 
                 reverse=True)
```

---

## 9. Scalability & Performance

### 9.1 Horizontal Scaling

**Ingestion Pipeline:**
```
S3 → SQS → [Worker 1, Worker 2, ..., Worker N] → DuckDB

Scaling strategy:
- Add more workers based on queue depth
- Each worker processes batches independently
- Auto-scaling: Target queue depth < 100
```

**Query Processing:**
```
Load Balancer → [API 1, API 2, ..., API N] → Cache → Services

Scaling strategy:
- Stateless API servers
- Auto-scale based on CPU (>70%) or latency (>500ms p99)
- Target: 10ms per request overhead
```

**Enrichment:**
```
Rate Limited Queue → [Enrichment Worker 1, ..., Worker N] → LLM API

Scaling strategy:
- Respect OpenAI rate limits (3,500 RPM)
- N workers × calls/min ≤ 3,500
- Priority queue: Critical calls first
```

### 9.2 Vertical Scaling

**Database:**
- Current: Single DuckDB instance
- Scale up: Increase memory for larger working set
- Limit: ~1TB of data before horizontal split needed

**Vector Search:**
- Current: Single FAISS instance in memory
- Scale up: GPU acceleration for faster search
- Alternative: Shard vectors across multiple indices

### 9.3 Caching Strategy

**Multi-Level Cache:**

```python
# L1: Application cache (in-memory)
@lru_cache(maxsize=1000)
def get_preprocessed_call(call_id):
    return db.query(f"SELECT * FROM preprocessed_calls WHERE call_id = '{call_id}'")

# L2: Redis cache (distributed)
@redis_cache(ttl=3600)
def get_query_result(query_hash):
    return execute_query(query_hash)

# L3: CDN cache (edge)
@cdn_cache(ttl=86400)
def get_analytics_summary():
    return analytics_service.get_summary()
```

**Cache Invalidation:**
```python
# Event-driven invalidation
def on_call_processed(call_id):
    cache.delete(f"call:{call_id}")
    cache.delete_pattern("analytics:*")  # Invalidate aggregate caches
    
# Time-based invalidation
# - Preprocessed calls: 1 hour TTL
# - Query results: 15 minutes TTL
# - Analytics: 24 hours TTL
```

### 9.4 Performance Optimizations

**Database Query Optimization:**
```sql
-- Bad: Full table scan
SELECT * FROM calls WHERE LOWER(product) LIKE '%subscription%';

-- Good: Index scan
CREATE INDEX idx_product_lower ON calls(LOWER(product));
SELECT * FROM calls WHERE LOWER(product) = 'subscription_service';

-- Best: Materialized view
CREATE MATERIALIZED VIEW call_stats AS
SELECT product, COUNT(*) as count, AVG(importance_score) as avg_importance
FROM calls
GROUP BY product;
REFRESH MATERIALIZED VIEW call_stats;  -- Periodic refresh
```

**Embedding Generation:**
```python
# Bad: Sequential processing
for document in documents:
    embedding = model.encode(document)  # 50ms each

# Good: Batch processing
embeddings = model.encode(
    documents,
    batch_size=32,  # Process 32 at once
    show_progress_bar=False
)  # 1ms per document
```

**Vector Search:**
```python
# Bad: Flat index (exhaustive search)
index = faiss.IndexFlatL2(dimension)  # O(n) search time

# Good: IVF index (approximate search)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(training_vectors)  # One-time training
# Search time: O(n/nlist) ≈ O(n/100)

# Better: HNSW index (graph-based)
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
# Search time: O(log n)
```

---

## 10. Reliability & Fault Tolerance

### 10.1 Failure Modes

**1. Database Unavailable**
```python
@retry(tries=3, delay=2, backoff=2)
def query_database(sql):
    try:
        return db.execute(sql)
    except DatabaseConnectionError:
        logger.error("db_connection_failed", retry_count=retry.count)
        if retry.count >= 3:
            # Fallback to read replica
            return read_replica.execute(sql)
        raise
```

**2. LLM API Failure**
```python
class LLMProvider:
    async def generate_with_fallback(self, prompt):
        """
        Fallback strategy:
        1. Try primary model (GPT-3.5)
        2. If fails, try GPT-4 (might have different rate limits)
        3. If still fails, return cached similar query
        4. If no cache, return error with partial results
        """
        try:
            return await self.gpt35.generate(prompt)
        except RateLimitError:
            logger.warning("rate_limit_hit", model="gpt-3.5")
            return await self.gpt4.generate(prompt)
        except APIError as e:
            logger.error("llm_api_error", error=str(e))
            # Try cache
            similar_query = self.find_similar_cached_query(prompt)
            if similar_query:
                return similar_query.result
            raise
```

**3. Vector Index Corruption**
```python
class VectorStore:
    def __init__(self):
        self.primary_index = self.load_or_create("primary.index")
        self.backup_index = self.load_or_create("backup.index")
    
    def search(self, query_vector, top_k):
        try:
            return self.primary_index.search(query_vector, top_k)
        except IndexCorruptionError:
            logger.error("primary_index_corrupted")
            # Fallback to backup
            results = self.backup_index.search(query_vector, top_k)
            # Trigger async rebuild of primary
            asyncio.create_task(self.rebuild_primary())
            return results
```

### 10.2 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """
    Prevent cascading failures
    
    States:
    - CLOSED: Normal operation
    - OPEN: All requests fail fast
    - HALF_OPEN: Test if service recovered
    """
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"
        self.last_failure_time = None
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error("circuit_breaker_opened", service=func.__name__)
            
            raise

# Usage
llm_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

@llm_circuit_breaker.call
async def call_llm(prompt):
    return await llm.generate(prompt)
```

### 10.3 Data Durability

**Backup Strategy:**
```
┌─────────────────────────────────────────────┐
│ Real-time Replication                       │
├─────────────────────────────────────────────┤
│ Primary DB ──streaming──> Replica DB        │
│ (Every write)           (Read queries)      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Periodic Snapshots                          │
├─────────────────────────────────────────────┤
│ Every 6 hours → S3 Glacier                  │
│ Keep: Last 7 days (hourly)                  │
│       Last 4 weeks (daily)                  │
│       Last 12 months (monthly)              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Point-in-Time Recovery                      │
├─────────────────────────────────────────────┤
│ Transaction logs → S3                       │
│ Retention: 7 days                           │
│ Recovery time: < 1 hour                     │
└─────────────────────────────────────────────┘
```

---

## 11. Security & Compliance

### 11.1 Authentication & Authorization

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    JWT token verification
    
    Token contains:
    - user_id
    - roles: [admin, analyst, viewer]
    - permissions: [read_all, write_calls, manage_users]
    """
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return User(
            id=payload["user_id"],
            roles=payload["roles"],
            permissions=payload["permissions"]
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Usage
@app.post("/query")
async def query(request: QueryRequest, user: User = Depends(verify_token)):
    # Check permissions
    if "read_calls" not in user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return await process_query(request)
```

### 11.2 PII Detection & Redaction

```python
class PIIDetector:
    """
    Detect and redact PII in transcripts
    
    PII Types:
    - Names
    - Phone numbers
    - Email addresses
    - Credit card numbers
    - SSN
    - Addresses
    """
    
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\d{3}[-.]?)?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    }
    
    def detect(self, text: str) -> List[PIIMatch]:
        matches = []
        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end()
                ))
        return matches
    
    def redact(self, text: str) -> str:
        matches = self.detect(text)
        
        # Replace in reverse order to maintain positions
        for match in reversed(matches):
            redacted = self.REDACTION_TOKENS[match.type]
            text = text[:match.start] + redacted + text[match.end:]
        
        return text

# Usage in pipeline
def preprocess_call(call_transcript):
    # Detect PII
    pii_matches = pii_detector.detect(call_transcript)
    
    if pii_matches:
        logger.warning("pii_detected", call_id=call.id, types=[m.type for m in pii_matches])
        
        # Redact for processing
        redacted_transcript = pii_detector.redact(call_transcript)
        
        # Store both (encrypted)
        store_original(call_transcript, encrypted=True)
        return redacted_transcript
    
    return call_transcript
```

### 11.3 Audit Logging

```python
class AuditLogger:
    """
    Log all sensitive operations
    
    Compliance requirements:
    - GDPR: Log data access
    - SOC 2: Log configuration changes
    - HIPAA: Log PHI access
    """
    
    def log_query(self, user_id, query, results):
        self.db.insert("audit_log", {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "action": "query",
            "query_text": query,
            "num_results": len(results),
            "calls_accessed": [r.call_id for r in results],
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent")
        })
    
    def log_data_access(self, user_id, call_id):
        self.db.insert("audit_log", {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "action": "data_access",
            "resource_type": "call",
            "resource_id": call_id,
            "ip_address": request.client.host
        })
```

---

## 12. Monitoring & Observability

### 12.1 Metrics to Track

**Business Metrics:**
```python
# Daily active users
metrics.gauge("dau", count_unique_users_today())

# Query volume by type
metrics.counter("queries.total")
metrics.counter("queries.aggregation")
metrics.counter("queries.search")
metrics.counter("queries.insight")

# User satisfaction (explicit feedback)
metrics.gauge("satisfaction.score", avg_rating)
```

**System Metrics:**
```python
# API latency
metrics.histogram("api.latency", duration_ms, tags=["endpoint", "status"])

# Database query time
metrics.histogram("db.query_time", duration_ms, tags=["query_type"])

# LLM API latency
metrics.histogram("llm.latency", duration_ms, tags=["model", "prompt_type"])

# Cache hit rate
metrics.gauge("cache.hit_rate", hits / (hits + misses))
```

**Cost Metrics:**
```python
# LLM costs
metrics.counter("llm.cost", cost_usd, tags=["model"])
metrics.counter("llm.tokens", token_count, tags=["model", "type"])

# Processing costs
metrics.counter("processing.calls_enriched")
metrics.gauge("processing.enrichment_rate", enriched / total)

# Daily burn rate
metrics.gauge("cost.daily_burn", get_daily_cost())
```

**Quality Metrics:**
```python
# Importance scoring accuracy
metrics.gauge("preprocessing.accuracy", accuracy_score)

# RAG quality
metrics.gauge("rag.confidence", avg_confidence)
metrics.gauge("rag.citation_rate", queries_with_citations / total_queries)
```

### 12.2 Alerting Rules

```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 5%
    for: 5m
    severity: critical
    action: page_on_call
  
  - name: HighLatency
    condition: p99_latency > 10s
    for: 5m
    severity: warning
    action: slack_alert
  
  - name: CostSpike
    condition: hourly_cost > $100
    for: 1h
    severity: warning
    action: email_team
  
  - name: LowCacheHitRate
    condition: cache_hit_rate < 30%
    for: 30m
    severity: info
    action: slack_alert
  
  - name: HighEnrichmentRate
    condition: enrichment_rate > 25%
    for: 1h
    severity: warning
    action: email_team
    description: "More calls being enriched than expected - check preprocessing"
```

### 12.3 Distributed Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("process_query")
async def process_query(query: str):
    with tracer.start_as_current_span("classify_query"):
        query_type = classify_query(query)
    
    with tracer.start_as_current_span(f"route_to_{query_type}"):
        if query_type == "aggregation":
            with tracer.start_as_current_span("sql_query"):
                result = await analytics_service.query(query)
        elif query_type == "search":
            with tracer.start_as_current_span("hybrid_search"):
                result = await search_service.search(query)
        else:
            with tracer.start_as_current_span("rag_pipeline"):
                with tracer.start_as_current_span("retrieval"):
                    documents = await retrieve(query)
                
                with tracer.start_as_current_span("llm_synthesis"):
                    result = await llm.generate(documents)
    
    return result
```

**Trace Visualization:**
```
Query Request (2.8s total)
├─ classify_query (50ms)
├─ route_to_insight
│  └─ rag_pipeline (2.7s)
│     ├─ retrieval (200ms)
│     │  ├─ embed_query (50ms)
│     │  ├─ vector_search (80ms)
│     │  └─ keyword_search (70ms)
│     └─ llm_synthesis (2.5s)
│        ├─ construct_context (10ms)
│        └─ openai_api_call (2.49s)
```

---

## 13. Cost Optimization

### 13.1 Cost Breakdown

**For 1M Calls/Month:**

| Component | Usage | Unit Cost | Monthly Cost | % of Total |
|-----------|-------|-----------|--------------|------------|
| Preprocessing | 1M calls | $0 | $0 | 0% |
| Enrichment (GPT-3.5) | 135K calls | $0.003 | $405 | 38% |
| Enrichment (GPT-4) | 15K calls | $0.012 | $180 | 17% |
| Embeddings | 1M calls | $0.0001 | $100 | 9% |
| Storage (S3) | 180 GB | $0.023/GB | $4 | <1% |
| Database (DuckDB/Snowflake) | - | - | $100 | 9% |
| Compute (EC2) | 24/7 | - | $300 | 28% |
| **Total** | | | **$1,089** | **100%** |

**Compare to Naive:**
- LLM on every call: $50,000
- **Savings: 97.8% ($48,911)**

### 13.2 Cost Control Strategies

**1. Importance Threshold Tuning:**
```python
# Current: Enrich 15% of calls
HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4

# More aggressive: Enrich 10% (save 33%)
HIGH_THRESHOLD = 0.8
MEDIUM_THRESHOLD = 0.5

# More conservative: Enrich 20% (spend 33% more, better quality)
HIGH_THRESHOLD = 0.6
MEDIUM_THRESHOLD = 0.3
```

**2. Model Selection:**
```python
# Cost by model (per 1K tokens)
GPT_4 = 0.03          # Best quality
GPT_3_5_TURBO = 0.002 # 15x cheaper
GPT_3_5 = 0.0015      # 20x cheaper

# Strategy: Use cheapest model that meets quality bar
if importance == CRITICAL:
    model = GPT_4
elif importance == HIGH:
    model = GPT_3_5_TURBO
else:
    model = GPT_3_5  # Or skip LLM entirely
```

**3. Context Length Optimization:**
```python
# Bad: Send full transcript (2000 tokens)
prompt = f"Summarize: {full_transcript}"  # $0.003

# Good: Send key segments only (500 tokens)
important_segments = filter_by_importance(transcript, threshold=0.6)
prompt = f"Summarize: {important_segments}"  # $0.0007 (4x cheaper)
```

**4. Query Routing Optimization:**
```python
# Cost by query type
AGGREGATION = $0         # SQL
SEARCH = $0.001          # Embeddings
INSIGHT = $0.03          # LLM

# Optimize routing to prefer cheaper paths
def optimize_query(query):
    # Try to answer with SQL first
    sql_answer = try_sql_query(query)
    if sql_answer:
        return sql_answer  # $0
    
    # Try search next
    search_results = try_search(query)
    if is_sufficient(search_results):
        return format_results(search_results)  # $0.001
    
    # Fall back to LLM
    return rag_query(query)  # $0.03
```

**5. Caching:**
```python
# Cache expensive operations
@cache(ttl=3600)  # 1 hour
def get_llm_response(query_hash):
    return llm.generate(query)

# Cache hit rate optimization
# At 40% hit rate: Save $0.03 × 0.4 × queries/day
# At 10K queries/day: Save $120/day = $3,600/month
```

---

## 14. Trade-offs & Alternatives

### 14.1 Key Trade-offs

**1. Rule-Based vs ML Importance Scoring**

| Aspect | Rule-Based (Chosen) | ML Model |
|--------|---------------------|----------|
| Accuracy | 80% | 90-95% |
| Latency | 50ms | 100-500ms |
| Cost | $0 | $0 (training), $0.0001/call (inference) |
| Maintenance | Easy (edit rules) | Hard (retrain model) |
| Interpretability | High | Low |

**Decision:** Rule-based
- **Why:** 80% accuracy sufficient for triage, 10% error acceptable
- **Benefit:** Zero cost, instant, debuggable

**2. Local vs Cloud Embeddings**

| Aspect | Local (Chosen) | OpenAI |
|--------|----------------|--------|
| Quality | Good (768d) | Best (1536d) |
| Latency | 20ms | 200ms (API) |
| Cost | $0 | $0.0001/call |
| Dependencies | None | OpenAI API |

**Decision:** Local (sentence-transformers)
- **Why:** Quality sufficient, massive cost savings
- **Benefit:** $100/month vs $100K/month for 1M calls

**3. DuckDB vs Snowflake**

| Aspect | DuckDB (Demo) | Snowflake (Production) |
|--------|---------------|------------------------|
| Cost | $0 | $1,000-5,000/month |
| Scale | <1TB | Petabytes |
| Ops | Self-managed | Managed |
| Performance | Excellent | Excellent |

**Decision:** DuckDB for demo, Snowflake for production
- **Why:** DuckDB perfect for <100M rows, Snowflake for scale

**4. Batch vs Streaming Ingestion**

| Aspect | Batch (Chosen) | Streaming |
|--------|----------------|-----------|
| Latency | Minutes | Seconds |
| Complexity | Low | High |
| Cost | Low | High |
| Use case | Analytics | Real-time alerts |

**Decision:** Batch for now, add streaming later
- **Why:** Most queries are historical, batch is simpler

### 14.2 Alternative Architectures

**Alternative 1: Fine-Tuned Model Instead of RAG**

```
Pros:
- Lower latency (no retrieval)
- Lower cost per query (no LLM synthesis)

Cons:
- High upfront cost (fine-tuning)
- Stale knowledge (need retraining)
- No citations (black box)

Verdict: RAG better for our use case (evidence-based, always fresh)
```

**Alternative 2: Pure SQL Analytics (No GenAI)**

```
Pros:
- Lowest cost ($0)
- Fastest (<100ms)
- Most reliable

Cons:
- Can't handle semantic queries
- Requires predefined reports
- Limited flexibility

Verdict: Hybrid approach (SQL + RAG) gives best of both
```

**Alternative 3: Always Use GPT-4**

```
Pros:
- Best quality
- Simplest code

Cons:
- 15x more expensive ($15K vs $1K)
- Not necessary for most queries

Verdict: Tiered approach (3.5 by default, 4 for critical) is better ROI
```

---

## 15. Future Enhancements

### Phase 2 (3-6 months)

**1. Real-Time Processing**
```
Add streaming pipeline:
Kafka → Flink → [Real-time Importance Scoring] → Alerts

Use case: Alert when critical call detected
Latency: <1 minute from call end
```

**2. Feedback Loop**
```
User feedback → Improve importance scoring

Track:
- Which calls were actually important?
- Which enrichments were useful?
- Which queries got good answers?

Use feedback to:
- Tune importance thresholds
- Improve intent classification
- Retrain models
```

**3. Multi-Language Support**
```
Add language detection + translation:
1. Detect language (fasttext)
2. Translate to English (NLLB)
3. Process normally
4. Translate results back

Cost: +$0.0001/call for translation
```

### Phase 3 (6-12 months)

**4. Custom Embedding Model**
```
Fine-tune on domain data:
- Better semantic understanding
- Smaller dimension (384 → 256)
- Faster search

Expected improvement:
- 10% better search quality
- 30% faster search
```

**5. Automated Root Cause Analysis**
```
Cluster related calls:
1. Find spikes in volume
2. Cluster affected calls
3. Identify common patterns
4. Generate RCA report

Use case: "Why are complaints up 20% this week?"
```

**6. Predictive Analytics**
```
Forecast future issues:
- Predict churn risk from call content
- Predict escalation likelihood
- Predict agent performance

Models: LSTM/Transformer on call sequences
```

### Phase 4 (12+ months)

**7. Agent Assist (Real-Time)**
```
During live call:
1. Transcribe in real-time
2. Detect customer issue
3. Suggest responses to agent
4. Provide relevant knowledge articles

Latency requirement: <500ms
```

**8. Automated Quality Scoring**
```
Score every call automatically:
- Agent politeness
- Issue resolution
- Call efficiency
- Customer satisfaction

Replace manual QA sampling
```

**9. Voice Analytics**
```
Analyze audio directly:
- Tone/emotion detection
- Speaker diarization
- Accent/language detection
- Background noise analysis

Models: Wav2Vec, Whisper
```

---

## Interview Tips

### How to Present This Design

**1. Start with Clarifying Questions (5 min)**
- "How many calls per month?"
- "What's the latency requirement?"
- "What's the cost budget?"
- "Who are the users?"

**2. High-Level Design (10 min)**
- Draw the system diagram
- Explain data flow
- Highlight key components
- Mention the importance scoring innovation

**3. Deep Dive (20 min)**
- Interviewer picks components to explore
- Be ready to discuss any service in detail
- Show code-level understanding
- Explain trade-offs

**4. Scalability Discussion (10 min)**
- How to scale to 10M calls?
- How to handle geographic distribution?
- How to reduce costs further?

**5. Q&A (5 min)**

### Key Points to Emphasize

1. **Cost Consciousness**: "The key insight is that not all calls need expensive processing"

2. **Tiered Architecture**: "Different queries need different resources - route intelligently"

3. **Evidence-Based**: "Every answer cites sources - no hallucinations"

4. **Production-Ready**: "This isn't a demo - it's designed for real scale"

### Common Follow-Up Questions

**Q: How do you handle calls in multiple languages?**
A: Phase 2 - detect language, translate to English, process, translate back

**Q: What if OpenAI API is down?**
A: Circuit breaker pattern, fallback to cached results, graceful degradation

**Q: How do you ensure PII isn't sent to OpenAI?**
A: PII detection + redaction before any LLM call, store original encrypted

**Q: How do you measure success?**
A: Track: Query satisfaction, cost per query, P99 latency, importance scoring accuracy

---

## Conclusion

This system demonstrates **staff-level thinking**:

1. ✅ **Problem Definition**: Clear understanding of business needs
2. ✅ **Cost Awareness**: 97% cost reduction through intelligent design
3. ✅ **Scalability**: Linear scaling to millions of calls
4. ✅ **Reliability**: Multiple layers of fault tolerance
5. ✅ **Observability**: Comprehensive monitoring and alerts
6. ✅ **Trade-offs**: Explicit reasoning about alternatives
7. ✅ **Future Vision**: Clear roadmap for evolution

**The Key Innovation**: Importance scoring as a gatekeeper for expensive operations.

This pattern applies beyond call transcripts:
- Document analysis
- Video understanding
- Code review
- Any scenario where LLM cost is prohibitive

**Remember**: System design is about making **good-enough decisions** under constraints. There's no perfect solution - only trade-offs.
