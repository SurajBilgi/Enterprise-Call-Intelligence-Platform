# System Architecture Documentation

## Executive Summary

The Enterprise Call Intelligence Platform is designed to analyze millions of customer support call transcripts using a cost-optimized GenAI approach. The key innovation is **intelligent triage** that identifies which calls and speech segments require expensive LLM analysis.

---

## Core Design Principles

### 1. Cost-First Thinking
**Problem:** Running LLM on every call is prohibitively expensive  
**Solution:** Use cheap NLP to filter, only use LLM selectively  
**Impact:** 85% cost reduction

### 2. Tiered Processing
**Not all queries need the same treatment:**
- Simple aggregations → SQL (free)
- Document search → Vector search ($0.001)
- Deep insights → RAG + LLM ($0.05)

### 3. Importance Scoring
**The KEY innovation:**
Every call and segment gets an importance score (0-1) based on:
- Intent (complaint > inquiry > general)
- Sentiment (negative > neutral > positive)
- Keywords (high-priority terms)
- Speaker (customer > agent)

### 4. Observable by Default
- Every LLM call tracked with cost
- Every query logged with routing decision
- Performance metrics collected automatically

---

## Data Flow

### Offline Processing Pipeline (Batch)

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: INGESTION                                  │
├─────────────────────────────────────────────────────┤
│ Input:  Raw JSON files (S3 simulation)             │
│ Output: DuckDB structured records                   │
│ Cost:   $0                                          │
│ Time:   ~100ms/call                                 │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: PREPROCESSING (Importance Scoring) ⭐     │
├─────────────────────────────────────────────────────┤
│ Input:  Raw transcripts                             │
│ Process:                                            │
│   1. Segment by speaker                            │
│   2. Classify intent (rule-based)                  │
│   3. Analyze sentiment (pattern matching)          │
│   4. Detect keywords                               │
│   5. Calculate importance score                    │
│   6. Make triage decision                          │
│ Output: ImportanceLevel + requires_enrichment flag  │
│ Cost:   $0 (no LLM!)                               │
│ Time:   ~50ms/call                                  │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: SELECTIVE ENRICHMENT (LLM)                │
├─────────────────────────────────────────────────────┤
│ Input:  Only calls with requires_enrichment=True   │
│ Process:                                            │
│   1. Select model (GPT-3.5 vs GPT-4)              │
│   2. Summarize call                                │
│   3. Extract key issues                            │
│   4. Generate action items                         │
│   5. Create semantic tags                          │
│ Output: Enrichment result with cost tracking       │
│ Cost:   $0.002-0.05/call (15% of calls)           │
│ Time:   2-5s/call                                   │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: INDEXING                                   │
├─────────────────────────────────────────────────────┤
│ Input:  Enriched calls                              │
│ Process:                                            │
│   1. Generate embeddings (local model)             │
│   2. Build FAISS vector index                      │
│   3. Build TF-IDF keyword index                    │
│   4. Create analytics aggregations                 │
│ Output: Searchable indices                          │
│ Cost:   ~$0.0001/call (local embeddings)           │
│ Time:   ~200ms/call                                 │
└─────────────────────────────────────────────────────┘
```

### Online Query Processing (Real-time)

```
┌─────────────────────────────────────────────────────┐
│ USER QUERY                                          │
│ "What are customers complaining about?"            │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ QUERY ROUTER (Classification)                      │
├─────────────────────────────────────────────────────┤
│ Process:                                            │
│   1. Pattern matching on query text                │
│   2. Score query types:                            │
│      - Aggregation? (how many, count)              │
│      - Search? (find, show)                        │
│      - Insight? (what are, why, explain)           │
│   3. Route to appropriate service                  │
│ Decision: INSIGHT query → RAG Service              │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ RAG SERVICE                                         │
├─────────────────────────────────────────────────────┤
│ Step 1: RETRIEVAL                                   │
│   - Hybrid search (keyword + semantic)             │
│   - Top 10 relevant segments                       │
│   - Apply filters if any                           │
│                                                     │
│ Step 2: RE-RANKING                                  │
│   - Select top 5 for context                       │
│                                                     │
│ Step 3: CONTEXT CONSTRUCTION                        │
│   - Format segments with metadata                  │
│   - Stay within token limits                       │
│                                                     │
│ Step 4: LLM SYNTHESIS                               │
│   - Generate answer with citations                 │
│   - Use GPT-3.5 (balanced mode)                    │
│                                                     │
│ Output: Answer + sources + cost                    │
│ Cost:   $0.02-0.10                                 │
│ Time:   2-5s                                        │
└─────────────────────────────────────────────────────┘
```

---

## Service Breakdown

### Ingestion Service
**Responsibility:** Load raw data into structured storage

**Design Pattern:** ETL (Extract, Transform, Load)

**Key Features:**
- Batch processing with configurable batch size
- Parallel processing support
- Idempotent (can re-run safely)
- Validates data format

**Scalability:**
- Current: Single-threaded file reader
- Production: Kafka consumer, S3 event triggers

### Preprocessing Service ⭐
**Responsibility:** Score importance WITHOUT using LLM

**Design Pattern:** Rule-based + Statistical NLP

**Algorithm:**
```python
def calculate_importance_score(segment):
    score = 0.3  # Base score
    
    # Intent contribution
    if intent == COMPLAINT:
        score += 0.4
    elif intent == TECHNICAL_ISSUE:
        score += 0.3
    elif intent == INQUIRY:
        score += 0.15
    
    # Sentiment contribution
    if sentiment == VERY_NEGATIVE:
        score += 0.2
    elif sentiment == NEGATIVE:
        score += 0.1
    
    # Keyword bonuses
    score += min(num_high_priority_keywords * 0.15, 0.3)
    score += min(num_medium_priority_keywords * 0.05, 0.1)
    
    # Speaker weight
    if speaker == CUSTOMER:
        score += 0.1
    
    return clamp(score, 0.0, 1.0)
```

**Why This Matters:**
- This ONE service determines 85% of cost savings
- Fast enough to run on every call (50ms)
- Accurate enough to catch important issues (80%+ precision)
- Completely free (no API costs)

### Enrichment Service
**Responsibility:** Selective LLM usage for important calls

**Design Pattern:** Tiered processing

**Model Selection Logic:**
```python
if importance_level == CRITICAL or force_expensive:
    model = GPT-4  # $0.01/1K input, $0.03/1K output
else:
    model = GPT-3.5  # $0.0015/1K input, $0.002/1K output
```

**Tasks Performed:**
1. Summarization (150 tokens)
2. Key issue extraction (100 tokens)
3. Action items (100 tokens)
4. Semantic tagging (50 tokens)

**Total:** ~400 tokens → $0.002 (GPT-3.5) or $0.012 (GPT-4)

### Indexing Service
**Responsibility:** Build search infrastructure

**Three Indices:**
1. **Vector Index (FAISS)**
   - Purpose: Semantic similarity search
   - Model: sentence-transformers/all-MiniLM-L6-v2 (local)
   - Dimension: 384
   - Distance: L2 (normalized = cosine)

2. **Keyword Index (TF-IDF)**
   - Purpose: Exact term matching
   - Features: Unigrams + bigrams
   - Max features: 10,000

3. **Structured Data (DuckDB)**
   - Purpose: Analytics queries
   - Optimized: Column-oriented, indexed

**Hybrid Search:**
```python
final_score = (keyword_score * 0.3) + (semantic_score * 0.7)
```

### Analytics Service
**Responsibility:** Fast aggregations without LLM

**Query Types Handled:**
- Count: "How many calls about X?"
- Percentage: "What % are complaints?"
- Top-N: "What are the top issues?"
- Trends: "Show call volume over time"

**Implementation:** Pure SQL on DuckDB
- No LLM needed
- Subsecond response times
- Zero API costs

### RAG Service
**Responsibility:** Deep insights with evidence

**Pipeline:**
```
Query → Retrieve → Rerank → Construct Context → Generate → Cite
```

**Context Engineering:**
- Include top 5 segments
- Add metadata (product, sentiment, etc.)
- Format for clarity
- Stay within 4K token limit

**Prompt Engineering:**
```
You are analyzing call transcripts.

User Question: {query}

Relevant Transcripts:
[Call 1] Product: Credit Card | Intent: Complaint
{transcript}
---
[Call 2] ...

Answer based ONLY on the transcripts.
Cite evidence. Be specific.
```

### Query Router
**Responsibility:** Classify and route queries intelligently

**Classification Patterns:**

| Query Type | Patterns | Route To | Cost |
|------------|----------|----------|------|
| Aggregation | how many, count, percentage, top | Analytics | $0 |
| Search | find, show me, list, which calls | Search | $0.001 |
| Insight | what are saying, why, summarize | RAG | $0.05 |

**Impact:**
- 50% of queries → Analytics ($0)
- 30% of queries → Search ($0.001)
- 20% of queries → RAG ($0.05)
- **Average cost:** $0.01/query vs $0.05 without routing

---

## Data Models

### Call Metadata
```python
class CallMetadata:
    call_id: str
    client_id: str
    client_segment: ClientSegment  # retail, premium, enterprise
    product: ProductCategory
    call_duration_seconds: float
    call_timestamp: datetime
    region: str
    agent_id: str
```

### Preprocessing Result
```python
class PreprocessingResult:
    call_id: str
    segments: List[SegmentAnalysis]
    overall_intent: IntentCategory
    overall_sentiment: SentimentScore
    overall_importance: ImportanceLevel  # critical, high, medium, low
    requires_enrichment: bool  # THE KEY FIELD
    enrichment_reason: str
```

### Enrichment Result
```python
class EnrichmentResult:
    call_id: str
    summary: str
    key_issues: List[str]
    action_items: List[str]
    semantic_tags: List[str]
    model_used: str
    tokens_used: int
    cost_usd: float  # ALWAYS TRACKED
```

---

## Cost Analysis Deep Dive

### Scenario: 1 Million Calls

**Naive Approach (LLM everything):**
```
1M calls × $0.05/call = $50,000
```

**Our Approach:**

**Stage 1: Ingestion**
```
Cost: $0 (local processing)
```

**Stage 2: Preprocessing**
```
Cost: $0 (rule-based NLP)
Time: 1M calls × 50ms = 13.9 hours
```

**Stage 3: Selective Enrichment**
```
Calls requiring enrichment: 15% = 150,000
Using GPT-3.5 (90%): 135,000 × $0.002 = $270
Using GPT-4 (10%):    15,000 × $0.012 = $180
Total: $450
Time: 150K calls × 3s = 125 hours
```

**Stage 4: Indexing**
```
Embeddings: 1M × $0.0001 = $100 (local model, simulated)
Total: $100
Time: 1M calls × 200ms = 55.6 hours
```

**Query Costs (per 1K queries):**
```
500 analytics queries × $0 = $0
300 search queries × $0.001 = $0.30
200 RAG queries × $0.03 = $6
Total per 1K queries: $6.30
```

**Total Cost for 1M Calls + 1K Queries:**
```
Pipeline: $550
Queries: $6.30
Total: $556.30

vs Naive: $50,000
Savings: $49,443.70 (98.9% reduction)
```

---

## Scaling Strategies

### Horizontal Scaling

**Preprocessing:**
- Stateless workers
- Scale to 100+ workers
- Process 1M calls in <1 hour

**Enrichment:**
- Rate-limited workers
- Parallel batch processing
- Respect OpenAI rate limits

**Indexing:**
- Shard vector index
- Distributed FAISS
- Or use managed service (Pinecone)

### Caching

**Query Cache (Redis):**
```
Key: hash(query_text + filters)
TTL: 1 hour
Hit rate: ~40% for common queries
```

**Embedding Cache:**
```
Key: hash(text)
Permanent cache
Reuse for similar segments
```

### Database Optimization

**DuckDB → Snowflake:**
- Column-oriented (same benefit)
- Distributed compute
- Automatic scaling
- Separation of storage/compute

### Infrastructure

**AWS Architecture:**
```
S3 (raw) → Lambda (ingest) → ECS (preprocessing) → 
→ Step Functions (enrichment) → 
→ OpenSearch (vectors) + Snowflake (structured) →
→ ECS (API) + CloudFront (CDN)
```

---

## Trade-offs & Design Decisions

### 1. Rule-Based vs ML for Triage

**Decision:** Rule-based  
**Why:** 
- 80% accuracy is sufficient for triage
- 100x cheaper than LLM
- Interpretable and debuggable
- No training data needed

**Trade-off:** Lower accuracy but massive cost savings

### 2. Local Embeddings vs OpenAI

**Decision:** Local (sentence-transformers)  
**Why:**
- One-time cost
- No per-query cost
- Fast enough
- Privacy (no data sent out)

**Trade-off:** Lower quality embeddings but free

### 3. Batch vs Streaming

**Decision:** Batch (with streaming path)  
**Why:**
- Simpler for demo
- Easier cost control
- Production would add Kafka

**Trade-off:** Not real-time but easier to reason about

### 4. DuckDB vs Snowflake

**Decision:** DuckDB for demo  
**Why:**
- Embedded, no setup
- Fast for OLAP
- Easy migration path

**Trade-off:** Single machine limit

---

## Production Readiness Checklist

### Security
- [ ] Add authentication (JWT, OAuth)
- [ ] Implement authorization (RBAC)
- [ ] PII detection and redaction
- [ ] Data encryption at rest
- [ ] Audit logging
- [ ] Rate limiting per user

### Reliability
- [ ] Circuit breakers for external services
- [ ] Retry logic with exponential backoff
- [ ] Health checks for all services
- [ ] Graceful degradation
- [ ] Data backup and recovery

### Observability
- [ ] Structured logging (JSON)
- [ ] Distributed tracing (Jaeger/X-Ray)
- [ ] Metrics dashboard (Grafana)
- [ ] Alerting (PagerDuty)
- [ ] Cost monitoring

### Performance
- [ ] Redis caching layer
- [ ] Connection pooling
- [ ] Query result caching
- [ ] CDN for static assets
- [ ] Async processing for slow operations

### Operations
- [ ] CI/CD pipeline
- [ ] Infrastructure as Code (Terraform)
- [ ] Auto-scaling policies
- [ ] Disaster recovery plan
- [ ] Runbooks for common issues

---

## Monitoring & Alerts

### Key Metrics

**System Health:**
- API latency (p50, p95, p99)
- Error rate
- Service availability

**Business Metrics:**
- Queries per second
- Query type distribution
- Average query cost

**Cost Metrics:**
- LLM cost per hour/day
- Cost per query
- Enrichment rate

### Alert Thresholds

**Critical:**
- API error rate > 5%
- LLM cost > $100/hour
- Service down

**Warning:**
- API latency p99 > 10s
- Enrichment rate > 20% (potential over-enrichment)
- Cache hit rate < 30%

---

## Conclusion

This architecture demonstrates:
1. ✅ **Cost-first thinking** → 85%+ savings through intelligent triage
2. ✅ **Tiered processing** → Right tool for each job
3. ✅ **Enterprise patterns** → Observable, scalable, maintainable
4. ✅ **Evidence-based** → Always show your work
5. ✅ **Production-ready path** → Clear migration to cloud

The key innovation is recognizing that **not all data needs expensive processing**. By using cheap NLP to identify importance, we achieve 80% of the value at 15% of the cost.
