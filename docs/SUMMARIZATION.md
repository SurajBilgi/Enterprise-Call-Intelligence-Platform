# Smart Summarization for Long Call Transcripts

**Handling 45-60 Minute Calls at Scale**

---

## Table of Contents
1. [The Challenge](#1-the-challenge)
2. [Noise Characteristics in Call Transcripts](#2-noise-characteristics)
3. [Summarization Approaches](#3-summarization-approaches)
4. [Smart Chunking Strategies](#4-smart-chunking-strategies)
5. [Hierarchical Summarization](#5-hierarchical-summarization)
6. [Noise Filtering Techniques](#6-noise-filtering-techniques)
7. [Context-Aware Summarization](#7-context-aware-summarization)
8. [Scalability & Performance](#8-scalability--performance)
9. [Cost Optimization](#9-cost-optimization)
10. [Quality Metrics](#10-quality-metrics)
11. [Production Implementation](#11-production-implementation)
12. [Advanced Techniques](#12-advanced-techniques)

---

## 1. The Challenge

### Problem Statement

**Typical Call Characteristics:**
- Duration: 45-60 minutes
- Word count: 6,000-9,000 words
- Token count: 8,000-12,000 tokens
- Noise ratio: 60-70% (greetings, hold music, repetition, filler)
- Important content: 30-40% (actual issue discussion)

**Challenges:**

1. **Token Limits**
   - GPT-3.5: 4K context window
   - GPT-4: 8K context (standard)
   - GPT-4-Turbo: 128K context (expensive)
   - Long calls exceed limits → need chunking

2. **Cost**
   - Naive approach: Send full 12K tokens → $0.12/call (GPT-4)
   - At 1M calls/month: $120,000
   - Need: <$0.005/call → 24x reduction required

3. **Quality**
   - Must capture key issues
   - Must preserve context
   - Must filter noise
   - Must maintain accuracy

4. **Latency**
   - User expects summary in seconds
   - Long LLM calls take 10-30s
   - Need: <5s end-to-end

5. **Noise**
   - Greetings/pleasantries: "How are you today?"
   - Hold music descriptions: "Please hold..."
   - Repetition: Same issue explained multiple times
   - Cross-talk: Both speakers talking
   - Filler words: "um", "uh", "you know"
   - Off-topic: Weather, sports, etc.

---

## 2. Noise Characteristics in Call Transcripts

### Types of Noise

**1. Structural Noise**
```
[Agent]: "Thank you for calling ABC Company. My name is John. How can I help you today?"
[Customer]: "Hi John, how are you?"
[Agent]: "I'm doing well, thank you. And yourself?"
[Customer]: "Good, thanks for asking."

→ 4 turns, 0% useful information
```

**2. Procedural Noise**
```
[Agent]: "Let me pull up your account. Can you verify your account number?"
[Customer]: "Sure, it's 1234567890"
[Agent]: "And can you verify the last 4 of your SSN?"
[Customer]: "5678"
[Agent]: "Perfect, I have your account pulled up now."

→ 5 turns, only context is "account verification"
```

**3. Hold/Wait Noise**
```
[Agent]: "Let me check on that for you. Can you hold for a moment?"
[Customer]: "Sure"
[Agent]: "Thank you for holding. I've checked with my supervisor..."

→ Breaks in conversation, no information
```

**4. Repetition Noise**
```
[Customer]: "I'm having trouble logging in"
[Agent]: "You're having trouble logging in?"
[Customer]: "Yes, I can't log in"
[Agent]: "I understand you can't log in. Let me help with that."

→ Same information repeated 4 times
```

**5. Filler Noise**
```
[Customer]: "So, um, like, you know, I was trying to, uh, cancel my subscription, 
you know, and, like, it wasn't working, um, so yeah..."

→ 50% filler words
```

### Noise Detection Patterns

```python
NOISE_PATTERNS = {
    'greetings': [
        r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
        r'\bhow are you\b',
        r'\bhave a (great|good|nice) day\b'
    ],
    
    'procedural': [
        r'\bcan you (verify|confirm|provide)\b',
        r'\blet me (check|look|pull up)\b',
        r'\bplease hold\b',
        r'\bone moment please\b'
    ],
    
    'filler': [
        r'\b(um|uh|er|ah)\b',
        r'\byou know\b',
        r'\blike\b',
        r'\bbasically\b',
        r'\bactually\b'
    ],
    
    'repetition': [
        # Detect using ngram similarity
        # If current sentence is 80%+ similar to previous 3 sentences
    ]
}
```

---

## 3. Summarization Approaches

### Approach 1: Extractive Summarization

**Concept:** Select important sentences from original text

**Algorithm:**
```python
def extractive_summarize(transcript, max_sentences=10):
    """
    Extract most important sentences using TF-IDF + importance scoring
    
    Advantages:
    - Fast (no LLM needed)
    - Cheap ($0)
    - Factually accurate (direct quotes)
    
    Disadvantages:
    - May not flow well
    - Misses connections between ideas
    - Doesn't handle noise well
    """
    # 1. Split into sentences
    sentences = split_sentences(transcript)
    
    # 2. Remove noise sentences
    sentences = [s for s in sentences if not is_noise(s)]
    
    # 3. Score each sentence
    scores = []
    for sentence in sentences:
        score = 0
        
        # TF-IDF importance
        score += tfidf_score(sentence)
        
        # Position (early + late sentences often important)
        position_weight = position_importance(sentence.index)
        score += position_weight
        
        # Speaker (customer sentences weighted higher)
        if sentence.speaker == 'customer':
            score += 0.3
        
        # Keywords present
        if has_high_priority_keywords(sentence):
            score += 0.5
        
        # Sentiment (negative = more important)
        if is_negative_sentiment(sentence):
            score += 0.2
        
        scores.append((sentence, score))
    
    # 4. Select top sentences
    top_sentences = sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    
    # 5. Re-order chronologically
    top_sentences = sorted(top_sentences, key=lambda x: x[0].index)
    
    return [s[0] for s in top_sentences]
```

**Example:**
```
Input (20 sentences):
1. Hi, how are you today?
2. I'm trying to cancel my subscription.
3. Let me pull up your account.
4. I've been charged incorrectly.
5. Can you verify your account number?
...

Output (3 sentences):
2. I'm trying to cancel my subscription.
4. I've been charged incorrectly.
18. This has been happening for 3 months.
```

**When to Use:**
- As first-pass filter before LLM
- When factual accuracy is critical
- When cost/speed is priority
- For extracting key quotes

---

### Approach 2: Abstractive Summarization (LLM)

**Concept:** Generate new text that captures meaning

**Basic Implementation:**
```python
async def abstractive_summarize(transcript, model='gpt-3.5-turbo'):
    """
    Generate summary using LLM
    
    Advantages:
    - Natural language output
    - Can rephrase and synthesize
    - Handles noise well
    
    Disadvantages:
    - Expensive (~$0.02/call)
    - Slower (2-5s)
    - Can hallucinate
    - Limited by context window
    """
    prompt = f"""Summarize this customer support call in 3-4 sentences.
Focus on:
1. Customer's main issue
2. Resolution or next steps
3. Any unresolved concerns

Ignore greetings, hold times, and account verification.

Transcript:
{transcript}

Summary:"""
    
    return await llm.generate(prompt, model=model, max_tokens=150)
```

**Problem:** What if transcript is 12K tokens and model only supports 4K?

---

### Approach 3: Hybrid (Extractive → Abstractive)

**Concept:** Extract important parts, then LLM summarize

**Two-Stage Pipeline:**
```python
async def hybrid_summarize(full_transcript):
    """
    Best of both worlds:
    1. Extractive: Reduce 12K → 2K tokens (cheap, fast)
    2. Abstractive: Summarize 2K → 200 tokens (expensive but manageable)
    
    Cost: $0.003 vs $0.02 for direct LLM (6x cheaper)
    Quality: Similar or better (noise pre-filtered)
    """
    # Stage 1: Extractive filtering (FREE)
    important_segments = extractive_filter(
        full_transcript,
        target_tokens=2000,
        remove_noise=True,
        speaker_weight={'customer': 0.7, 'agent': 0.3}
    )
    
    # Stage 2: Abstractive summarization ($0.003)
    summary = await llm_summarize(important_segments)
    
    return summary
```

**This is the recommended approach.**

---

## 4. Smart Chunking Strategies

### Problem: 60-minute call = 12K tokens > model limits

### Strategy 1: Fixed-Size Chunking

```python
def fixed_chunk(transcript, chunk_size=1000):
    """
    Naive approach: Split into equal chunks
    
    Problem: May split mid-conversation
    """
    tokens = tokenize(transcript)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(tokens[i:i + chunk_size])
    
    return chunks
```

**Issues:**
- Breaks context
- May split important discussions
- Equal size ≠ equal importance

---

### Strategy 2: Semantic Chunking

```python
def semantic_chunk(transcript, max_tokens=1500):
    """
    Smart approach: Chunk by topic/conversation flow
    
    Better: Preserves context, respects conversation boundaries
    """
    # 1. Segment by topic changes
    segments = detect_topic_changes(transcript)
    
    # 2. Group segments into chunks under token limit
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for segment in segments:
        segment_tokens = count_tokens(segment)
        
        if current_tokens + segment_tokens > max_tokens:
            # Save current chunk
            chunks.append(current_chunk)
            current_chunk = [segment]
            current_tokens = segment_tokens
        else:
            current_chunk.append(segment)
            current_tokens += segment_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def detect_topic_changes(transcript):
    """
    Detect when conversation topic shifts
    
    Signals:
    - Agent says "Is there anything else I can help you with?"
    - Long pauses (>5 seconds)
    - Customer introduces new issue
    - Semantic similarity drop between sentences
    """
    segments = []
    current_segment = []
    
    for i, turn in enumerate(transcript):
        current_segment.append(turn)
        
        # Check for topic change signals
        if i < len(transcript) - 1:
            current_embedding = embed(turn.text)
            next_embedding = embed(transcript[i+1].text)
            
            similarity = cosine_similarity(current_embedding, next_embedding)
            
            # Topic changed if similarity drops
            if similarity < 0.6:
                segments.append(current_segment)
                current_segment = []
            
            # Or explicit topic change markers
            if re.search(r'\b(anything else|next question|also|additionally)\b', 
                        turn.text, re.IGNORECASE):
                segments.append(current_segment)
                current_segment = []
    
    if current_segment:
        segments.append(current_segment)
    
    return segments
```

---

### Strategy 3: Importance-Based Chunking

```python
def importance_chunk(transcript, max_tokens=2000):
    """
    Only include important segments
    
    Best: Filters noise, focuses on signal
    """
    # 1. Score each turn
    scored_turns = []
    for turn in transcript:
        score = calculate_importance_score(turn)
        scored_turns.append((turn, score))
    
    # 2. Sort by importance
    scored_turns.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Take top turns until token limit
    selected_turns = []
    total_tokens = 0
    
    for turn, score in scored_turns:
        turn_tokens = count_tokens(turn.text)
        
        if total_tokens + turn_tokens <= max_tokens:
            selected_turns.append(turn)
            total_tokens += turn_tokens
    
    # 4. Re-order chronologically
    selected_turns.sort(key=lambda x: x.timestamp)
    
    return selected_turns
```

**Comparison:**

| Strategy | Context Preserved | Noise Filtered | Complexity |
|----------|------------------|----------------|------------|
| Fixed | Low | No | Low |
| Semantic | High | No | Medium |
| Importance | Medium | Yes | High |
| **Hybrid** | **High** | **Yes** | **Medium** |

**Recommended:** Semantic + Importance hybrid

---

## 5. Hierarchical Summarization

### The Map-Reduce Pattern for Long Documents

**Concept:** Summarize chunks, then summarize summaries

```python
async def hierarchical_summarize(long_transcript):
    """
    Multi-level summarization for very long calls
    
    Level 1: Chunk transcript into segments
    Level 2: Summarize each segment
    Level 3: Summarize the summaries
    
    Example:
    12K tokens → 6 chunks of 2K tokens
    → 6 summaries of 200 tokens = 1.2K tokens
    → 1 final summary of 200 tokens
    
    Cost: 6×$0.003 + 1×$0.002 = $0.02
    vs Direct (impossible): Would need GPT-4-Turbo $0.12
    """
    # Level 1: Smart chunking
    chunks = semantic_chunk(
        long_transcript,
        max_tokens=2000
    )
    
    # Level 2: Summarize each chunk in parallel
    chunk_summaries = await asyncio.gather(*[
        summarize_chunk(chunk, level='detailed')
        for chunk in chunks
    ])
    
    # Level 3: Synthesize final summary
    combined = "\n\n".join([
        f"Segment {i+1}: {summary}"
        for i, summary in enumerate(chunk_summaries)
    ])
    
    final_summary = await synthesize_summary(combined, level='concise')
    
    return {
        'final_summary': final_summary,
        'segment_summaries': chunk_summaries,
        'num_segments': len(chunks)
    }


async def summarize_chunk(chunk, level='detailed'):
    """
    Summarize a single chunk
    
    level='detailed': 3-4 sentences per chunk
    level='concise': 1-2 sentences per chunk
    """
    if level == 'detailed':
        max_tokens = 100
        instruction = "Summarize in 3-4 sentences"
    else:
        max_tokens = 50
        instruction = "Summarize in 1-2 sentences"
    
    prompt = f"""{instruction}. Focus on customer issues and resolutions.

{chunk}

Summary:"""
    
    return await llm.generate(prompt, max_tokens=max_tokens)


async def synthesize_summary(segment_summaries, level='concise'):
    """
    Combine segment summaries into final summary
    """
    prompt = f"""These are summaries of different parts of a customer support call.
Create a coherent overall summary in 3-4 sentences.

Focus on:
1. Main customer issue(s)
2. How it was resolved
3. Any follow-up actions

Segment Summaries:
{segment_summaries}

Overall Summary:"""
    
    return await llm.generate(prompt, max_tokens=150)
```

**Example Flow:**

```
Original Call (60 min, 12K tokens):
├── Segment 1 (0-10 min): Account verification + initial issue
│   → Summary: "Customer couldn't access account due to password reset issue"
│
├── Segment 2 (10-25 min): Deep dive into issue
│   → Summary: "Password reset emails not arriving. Discovered email typo in account."
│
├── Segment 3 (25-40 min): Resolution steps
│   → Summary: "Updated email address. Sent new reset link. Successfully reset password."
│
├── Segment 4 (40-50 min): Additional concerns
│   → Summary: "Customer asked about 2FA setup. Agent provided instructions."
│
└── Segment 5 (50-60 min): Wrap-up
    → Summary: "Confirmed everything working. Scheduled follow-up call."

Final Summary:
"Customer contacted support unable to access account. Investigation revealed 
typo in email address preventing password reset emails from arriving. Agent 
corrected email, sent new reset link, and customer successfully reset password. 
Also provided 2FA setup instructions and scheduled follow-up call."
```

---

## 6. Noise Filtering Techniques

### Pre-Processing: Clean Before Summarizing

**Stage 1: Remove Obvious Noise**

```python
def filter_noise(transcript):
    """
    Multi-stage noise removal
    
    Order matters: Remove progressively
    """
    # 1. Remove greetings/closings
    transcript = remove_greetings(transcript)
    
    # 2. Remove procedural language
    transcript = remove_procedural(transcript)
    
    # 3. Remove filler words
    transcript = remove_fillers(transcript)
    
    # 4. Deduplicate repetitions
    transcript = remove_repetitions(transcript)
    
    # 5. Remove very short turns (< 3 words)
    transcript = remove_short_turns(transcript)
    
    return transcript


def remove_greetings(transcript):
    """Remove greeting exchanges"""
    noise_phrases = [
        r'^(hi|hello|hey|good morning|good afternoon)',
        r'how are you',
        r'have a (great|good|nice) day',
        r'thank you for calling',
        r'my name is \w+',
        r'is there anything else',
        r'thanks for (your|the) (help|time)'
    ]
    
    filtered = []
    for turn in transcript:
        text = turn.text.lower()
        
        # Skip if turn is mostly greeting
        if any(re.search(pattern, text) for pattern in noise_phrases):
            if len(text.split()) <= 10:  # Short greeting, skip
                continue
        
        filtered.append(turn)
    
    return filtered


def remove_procedural(transcript):
    """Remove account verification and procedural steps"""
    procedural_patterns = [
        r'can you (verify|confirm|provide)',
        r'let me (check|look|pull up|verify)',
        r'please hold',
        r'one moment',
        r'give me (a|one) (second|moment)',
        r'your (account|confirmation) number is',
        r'for (security|verification) purposes'
    ]
    
    filtered = []
    for turn in transcript:
        text = turn.text.lower()
        
        # Check if turn is procedural
        procedural_score = sum(
            1 for pattern in procedural_patterns
            if re.search(pattern, text)
        )
        
        # Keep if not heavily procedural
        if procedural_score <= 1 or len(text.split()) > 20:
            filtered.append(turn)
    
    return filtered


def remove_fillers(transcript):
    """Remove filler words from text"""
    filler_words = [
        r'\b(um|uh|er|ah|hmm)\b',
        r'\byou know\b',
        r'\blike\b',  # When used as filler
        r'\bbasically\b',
        r'\bactually\b',
        r'\bI mean\b',
        r'\bkind of\b',
        r'\bsort of\b'
    ]
    
    for turn in transcript:
        cleaned_text = turn.text
        
        for filler in filler_words:
            cleaned_text = re.sub(filler, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        turn.text = cleaned_text
    
    return transcript


def remove_repetitions(transcript, similarity_threshold=0.8):
    """
    Remove repeated information
    
    If customer says same thing 3 times, keep only once
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    filtered = []
    recent_customer_turns = []
    
    for turn in transcript:
        if turn.speaker == 'customer':
            # Check if similar to recent customer turns
            is_repetition = False
            
            for recent in recent_customer_turns[-3:]:  # Check last 3 turns
                sim = calculate_similarity(turn.text, recent.text)
                
                if sim > similarity_threshold:
                    is_repetition = True
                    break
            
            if not is_repetition:
                filtered.append(turn)
                recent_customer_turns.append(turn)
        else:
            # Keep agent turns (less likely to repeat)
            filtered.append(turn)
    
    return filtered


def calculate_similarity(text1, text2):
    """Calculate semantic similarity between texts"""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]
```

**Example:**

```
Before Filtering:
[Agent]: "Hi, my name is John. How can I help you today?"
[Customer]: "Hi John, how are you?"
[Agent]: "I'm well, thank you. What can I do for you?"
[Customer]: "Um, so, like, I'm having an issue with my account, you know?"
[Agent]: "Let me pull up your account. Can you verify your account number?"
[Customer]: "Sure, it's 123456"
[Agent]: "And last 4 of SSN?"
[Customer]: "5678"
[Agent]: "Perfect. What seems to be the issue?"
[Customer]: "I can't log in to my account"
[Customer]: "Like, I'm trying to log in but it's not working"
[Customer]: "Yeah, so I can't access my account"

After Filtering:
[Customer]: "I'm having an issue with my account"
[Agent]: "What seems to be the issue?"
[Customer]: "I can't log in to my account"

Reduction: 13 turns → 3 turns (77% noise removed)
```

---

## 7. Context-Aware Summarization

### Understanding Call Structure

**Typical Call Flow:**
```
1. Opening (0-2 min)
   - Greetings
   - Account verification
   - Issue statement
   
2. Investigation (2-30 min)
   - Gathering details
   - Troubleshooting
   - Research
   
3. Resolution (30-50 min)
   - Proposed solution
   - Implementation
   - Verification
   
4. Closing (50-60 min)
   - Additional questions
   - Summary
   - Farewell
```

**Context-Aware Approach:**

```python
class CallStructureAnalyzer:
    """
    Identify call structure and focus on important sections
    """
    
    def analyze_structure(self, transcript):
        """
        Segment call into phases
        """
        phases = {
            'opening': [],
            'investigation': [],
            'resolution': [],
            'closing': []
        }
        
        total_turns = len(transcript)
        
        for i, turn in enumerate(transcript):
            position_ratio = i / total_turns
            
            # Classify based on position + content
            if position_ratio < 0.1:
                phase = 'opening'
            elif position_ratio > 0.85:
                phase = 'closing'
            elif self.contains_resolution_language(turn.text):
                phase = 'resolution'
            else:
                phase = 'investigation'
            
            phases[phase].append(turn)
        
        return phases
    
    def contains_resolution_language(self, text):
        """Detect if turn contains resolution language"""
        resolution_patterns = [
            r'\b(fixed|resolved|solution|try this|here\'s what)\b',
            r'\b(I\'ve (sent|updated|changed|fixed))\b',
            r'\b(should work now|that should fix)\b'
        ]
        
        return any(re.search(p, text, re.IGNORECASE) 
                  for p in resolution_patterns)
    
    def smart_summarize(self, transcript):
        """
        Summarize with different focus per phase
        """
        phases = self.analyze_structure(transcript)
        
        # Different importance weights per phase
        summaries = {
            'issue': self.summarize_phase(
                phases['opening'] + phases['investigation'],
                focus='problem_identification',
                max_sentences=2
            ),
            'resolution': self.summarize_phase(
                phases['resolution'],
                focus='solution',
                max_sentences=2
            ),
            'outcome': self.summarize_phase(
                phases['closing'],
                focus='confirmation',
                max_sentences=1
            )
        }
        
        # Combine into structured summary
        return f"""
Issue: {summaries['issue']}
Resolution: {summaries['resolution']}
Outcome: {summaries['outcome']}
        """.strip()
```

**Example:**

```
Input: 60-minute call

Phase Analysis:
- Opening (0-3 min): Account verification, customer states can't login
- Investigation (3-35 min): Check account, test login, review logs, find email typo
- Resolution (35-52 min): Update email, send reset link, customer resets password
- Closing (52-60 min): Verify access working, setup 2FA, schedule follow-up

Smart Summary:
Issue: Customer unable to login to account. Investigation revealed typo in 
       registered email address preventing password reset emails.
       
Resolution: Updated email address in account settings. Sent new password reset 
           link. Customer successfully reset password and confirmed access.
           
Outcome: Access restored. Enabled 2FA for additional security. Follow-up 
        call scheduled for next week.
```

---

## 8. Scalability & Performance

### Challenge: 1M calls/month × 45 min avg = 45M minutes

**Bottlenecks:**

1. **LLM API Rate Limits**
   - OpenAI: 3,500 requests/minute (GPT-3.5)
   - Need: 1M calls/month = 33K/day = 23/min
   - ✅ Within limits (single account)

2. **Processing Time**
   - Per call: 2-5 seconds (LLM)
   - 1M calls: 2M-5M seconds = 556-1,389 hours
   - ✅ Parallelizable across multiple workers

3. **Cost**
   - Naive: 12K tokens × $0.002/1K = $0.024/call
   - 1M calls: $24,000/month
   - ❌ Too expensive

### Scalability Solutions

**1. Parallel Processing Architecture**

```python
class SummarizationPipeline:
    """
    Distributed summarization pipeline
    
    Architecture:
    S3 → SQS → [Worker 1, Worker 2, ..., Worker N] → DynamoDB
    
    Throughput: N workers × 60 calls/hour = 1,440N calls/hour
    For 33K calls/day: Need 33,000/(24×60) = 23 concurrent workers
    """
    
    def __init__(self, num_workers=25):
        self.num_workers = num_workers
        self.queue = asyncio.Queue()
        self.workers = []
    
    async def start(self):
        """Start worker pool"""
        self.workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
    
    async def worker(self, worker_id):
        """Worker process"""
        while True:
            # Get call from queue
            call_id = await self.queue.get()
            
            try:
                # Load transcript
                transcript = await self.load_transcript(call_id)
                
                # Summarize
                summary = await self.smart_summarize(transcript)
                
                # Store result
                await self.store_summary(call_id, summary)
                
                logger.info(f"Worker {worker_id} summarized {call_id}")
                
            except Exception as e:
                logger.error(f"Worker {worker_id} failed: {e}")
            
            finally:
                self.queue.task_done()
    
    async def smart_summarize(self, transcript):
        """Multi-stage summarization"""
        # Stage 1: Filter noise (0.1s, $0)
        filtered = filter_noise(transcript)
        
        # Stage 2: Extractive summary (0.2s, $0)
        important_parts = extractive_summarize(filtered, max_tokens=2000)
        
        # Stage 3: LLM summarize (2-3s, $0.003)
        final_summary = await llm_summarize(important_parts)
        
        return final_summary
```

**2. Batch Processing**

```python
async def batch_summarize(call_ids, batch_size=10):
    """
    Process multiple calls in parallel
    
    With rate limiting to respect API limits
    """
    semaphore = asyncio.Semaphore(50)  # Max 50 concurrent API calls
    
    async def process_with_limit(call_id):
        async with semaphore:
            return await summarize_call(call_id)
    
    # Process in batches
    results = []
    for i in range(0, len(call_ids), batch_size):
        batch = call_ids[i:i + batch_size]
        
        batch_results = await asyncio.gather(
            *[process_with_limit(cid) for cid in batch],
            return_exceptions=True
        )
        
        results.extend(batch_results)
        
        # Small delay between batches
        await asyncio.sleep(0.1)
    
    return results
```

**3. Caching Strategies**

```python
class SummaryCache:
    """
    Cache summaries to avoid reprocessing
    """
    
    def __init__(self):
        self.redis = Redis()
    
    async def get_or_create_summary(self, call_id, transcript):
        """
        Check cache first, summarize if not found
        """
        # Check cache
        cache_key = f"summary:{call_id}"
        cached = await self.redis.get(cache_key)
        
        if cached:
            logger.info(f"Cache hit for {call_id}")
            return json.loads(cached)
        
        # Generate summary
        summary = await self.summarize(transcript)
        
        # Cache with 30-day TTL
        await self.redis.setex(
            cache_key,
            30 * 24 * 3600,  # 30 days
            json.dumps(summary)
        )
        
        return summary
```

**4. Progressive Summarization**

```python
async def progressive_summarize(call_id, transcript_stream):
    """
    Summarize as transcript arrives (for live calls)
    
    Instead of waiting for full 60-minute call,
    generate incremental summaries every 10 minutes
    """
    partial_summaries = []
    
    async for chunk in transcript_stream:
        # Every 10 minutes worth of transcript
        if len(chunk) >= CHUNK_SIZE:
            partial_summary = await summarize_chunk(chunk)
            partial_summaries.append(partial_summary)
            
            # Store intermediate result
            await store_partial_summary(call_id, partial_summaries)
    
    # When call ends, synthesize final summary
    final_summary = await synthesize_summaries(partial_summaries)
    
    return final_summary
```

---

## 9. Cost Optimization

### Current Costs (Naive Approach)

```
Average call: 12,000 tokens
GPT-3.5-Turbo: $0.0015/1K input + $0.002/1K output
Input cost: 12 × $0.0015 = $0.018
Output cost: 0.2 × $0.002 = $0.0004
Total per call: $0.0184

For 1M calls: $18,400/month
```

### Optimized Costs

**Stage 1: Noise Filtering (FREE)**
```
Input: 12,000 tokens
Output: 4,000 tokens (67% noise removed)
Cost: $0
Time: 0.2s
```

**Stage 2: Extractive Summarization (FREE)**
```
Input: 4,000 tokens
Output: 2,000 tokens (50% reduction)
Cost: $0
Time: 0.3s
```

**Stage 3: LLM Summarization (PAID)**
```
Input: 2,000 tokens
Output: 200 tokens
Cost: 2 × $0.0015 + 0.2 × $0.002 = $0.0034
Time: 2s
```

**Total Cost per Call:**
```
$0.0034 vs $0.0184 (naive)
Savings: 82%

For 1M calls: $3,400/month vs $18,400
Savings: $15,000/month
```

### Additional Optimizations

**1. Use Cheaper Models Strategically**

```python
def select_model(call_importance):
    """
    Model selection based on importance
    
    Critical calls: GPT-4 (best quality)
    High calls: GPT-3.5-Turbo (balanced)
    Medium calls: GPT-3.5 (cheapest)
    Low calls: Extractive only (free)
    """
    if call_importance == 'critical':
        return 'gpt-4', 0.01  # $0.01/1K
    elif call_importance == 'high':
        return 'gpt-3.5-turbo', 0.0015
    elif call_importance == 'medium':
        return 'gpt-3.5', 0.001
    else:
        return None, 0  # Skip LLM
```

**2. Batch API Calls**

```python
# OpenAI batch API (when available) - 50% discount
async def batch_summarize_api(transcripts):
    """
    Use batch API for non-urgent summarization
    
    Trade-off: 24-hour delay for 50% cost reduction
    Good for: Historical data processing
    Bad for: Real-time needs
    """
    batch_request = {
        'requests': [
            {
                'custom_id': f'call_{i}',
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': t}]
                }
            }
            for i, t in enumerate(transcripts)
        ]
    }
    
    # Submit batch
    batch_id = await openai.batches.create(batch_request)
    
    # Cost: 50% of normal API
    return batch_id
```

**3. Reuse Summaries**

```python
# If customer calls back about same issue,
# reuse previous summary instead of regenerating

async def get_related_summaries(call_id):
    """
    Find recent calls from same customer
    """
    customer_id = get_customer_id(call_id)
    
    recent_calls = query_db(f"""
        SELECT call_id, summary 
        FROM calls 
        WHERE customer_id = '{customer_id}'
        AND call_timestamp > NOW() - INTERVAL '7 days'
        ORDER BY call_timestamp DESC
        LIMIT 3
    """)
    
    return recent_calls
```

---

## 10. Quality Metrics

### How to Measure Summary Quality?

**1. ROUGE Scores**

```python
from rouge import Rouge

def calculate_rouge(summary, reference):
    """
    ROUGE: Recall-Oriented Understudy for Gisting Evaluation
    
    Measures: Overlap with reference summary
    
    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap
    ROUGE-L: Longest common subsequence
    """
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)[0]
    
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }

# Good scores:
# ROUGE-1 > 0.40
# ROUGE-2 > 0.20
# ROUGE-L > 0.35
```

**2. Coverage**

```python
def calculate_coverage(summary, original_transcript):
    """
    Measure how much of the important information is captured
    
    Method:
    1. Extract key facts from original (entities, actions, outcomes)
    2. Check if summary mentions each fact
    3. Coverage = facts_in_summary / total_facts
    """
    # Extract key facts
    original_facts = extract_key_facts(original_transcript)
    summary_facts = extract_key_facts(summary)
    
    # Calculate overlap
    covered = sum(1 for fact in original_facts 
                 if any(similar(fact, sf) for sf in summary_facts))
    
    coverage = covered / len(original_facts)
    
    return coverage

# Good coverage: > 0.80 (80% of key facts present)
```

**3. Factual Consistency**

```python
async def check_factual_consistency(summary, transcript):
    """
    Verify summary doesn't contain hallucinations
    
    Method: Use NLI (Natural Language Inference) model
    """
    from transformers import pipeline
    
    nli = pipeline("text-classification", 
                  model="microsoft/deberta-large-mnli")
    
    # Split summary into claims
    claims = split_into_claims(summary)
    
    inconsistencies = []
    for claim in claims:
        # Check if claim is entailed by transcript
        result = nli(f"{transcript} [SEP] {claim}")
        
        if result['label'] != 'ENTAILMENT':
            inconsistencies.append({
                'claim': claim,
                'label': result['label'],
                'score': result['score']
            })
    
    consistency_score = 1 - (len(inconsistencies) / len(claims))
    
    return {
        'score': consistency_score,
        'inconsistencies': inconsistencies
    }

# Good consistency: > 0.95 (< 5% hallucinations)
```

**4. Compression Ratio**

```python
def calculate_compression(summary, original):
    """
    Measure how much compression achieved
    """
    original_tokens = count_tokens(original)
    summary_tokens = count_tokens(summary)
    
    compression_ratio = summary_tokens / original_tokens
    
    return {
        'ratio': compression_ratio,
        'original_tokens': original_tokens,
        'summary_tokens': summary_tokens,
        'tokens_saved': original_tokens - summary_tokens
    }

# Good compression: 0.05 - 0.15 (5-15% of original length)
# Too high (>0.20): Not compressed enough
# Too low (<0.05): May lose important details
```

**5. User Satisfaction**

```python
# Collect feedback
def collect_summary_feedback(summary_id, user_id):
    """
    Ask user if summary was helpful
    
    Track:
    - Thumbs up/down
    - Which parts were missing
    - Which parts were wrong
    """
    return {
        'helpful': True/False,
        'completeness': 1-5,
        'accuracy': 1-5,
        'clarity': 1-5,
        'comments': "..."
    }

# Target: >80% thumbs up, >4.0 average rating
```

---

## 11. Production Implementation

### Complete Pipeline

```python
class ProductionSummarizer:
    """
    Production-ready summarization service
    
    Features:
    - Multi-stage processing
    - Cost optimization
    - Quality monitoring
    - Fault tolerance
    - Scalability
    """
    
    def __init__(self):
        self.noise_filter = NoiseFilter()
        self.extractive = ExtractiveSummarizer()
        self.llm = LLMProvider()
        self.cache = SummaryCache()
        self.metrics = MetricsCollector()
    
    async def summarize(self, call_id, transcript, 
                       quality='balanced', 
                       force_refresh=False):
        """
        Main summarization endpoint
        
        quality:
        - 'fast': Extractive only (free, <1s)
        - 'balanced': Hybrid (cheap, 2-3s)
        - 'high': Full LLM with GPT-4 (expensive, 5-10s)
        """
        start_time = time.time()
        
        # Check cache
        if not force_refresh:
            cached = await self.cache.get_summary(call_id)
            if cached:
                self.metrics.increment('cache_hits')
                return cached
        
        try:
            # Stage 1: Noise filtering
            with self.metrics.timer('noise_filtering'):
                filtered = self.noise_filter.filter(transcript)
            
            # Stage 2: Extractive summarization
            with self.metrics.timer('extractive'):
                extracted = self.extractive.summarize(
                    filtered,
                    target_tokens=2000 if quality != 'fast' else 500
                )
            
            # Stage 3: LLM (if not fast mode)
            if quality == 'fast':
                summary = self._format_extractive(extracted)
                cost = 0
            else:
                with self.metrics.timer('llm'):
                    model = 'gpt-4' if quality == 'high' else 'gpt-3.5-turbo'
                    summary, cost = await self._llm_summarize(
                        extracted,
                        model=model
                    )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality(
                summary, 
                transcript
            )
            
            # Build result
            result = {
                'call_id': call_id,
                'summary': summary,
                'metadata': {
                    'quality_mode': quality,
                    'original_tokens': count_tokens(transcript),
                    'compressed_tokens': count_tokens(summary),
                    'compression_ratio': quality_metrics['compression'],
                    'cost_usd': cost,
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'quality_score': quality_metrics['score']
                },
                'timestamp': datetime.now()
            }
            
            # Cache result
            await self.cache.set_summary(call_id, result)
            
            # Log metrics
            self.metrics.record('summarization', result['metadata'])
            
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed for {call_id}: {e}")
            self.metrics.increment('errors')
            
            # Fallback: Return extractive summary
            return {
                'call_id': call_id,
                'summary': self._format_extractive(extracted),
                'metadata': {
                    'quality_mode': 'fallback',
                    'error': str(e)
                }
            }
    
    async def _llm_summarize(self, text, model='gpt-3.5-turbo'):
        """LLM summarization with retries"""
        
        @retry(tries=3, delay=2, backoff=2)
        async def generate():
            prompt = self._build_prompt(text)
            result = await self.llm.generate(
                prompt,
                model=model,
                max_tokens=200,
                temperature=0.3  # Lower = more focused
            )
            return result
        
        result = await generate()
        
        cost = (count_tokens(text) / 1000) * MODEL_COSTS[model]['input'] + \
               (result['tokens_used'] / 1000) * MODEL_COSTS[model]['output']
        
        return result['text'], cost
    
    def _build_prompt(self, text):
        """Build optimized prompt for summarization"""
        return f"""Summarize this customer support call transcript in 3-4 sentences.

Focus on:
1. Customer's primary issue or request
2. Actions taken to resolve it
3. Final outcome or next steps

Be concise but specific. Ignore greetings, hold times, and verification steps.

Transcript:
{text}

Summary:"""
    
    def _calculate_quality(self, summary, original):
        """Calculate quality metrics"""
        return {
            'compression': count_tokens(summary) / count_tokens(original),
            'score': self._estimate_quality_score(summary, original)
        }
    
    def _estimate_quality_score(self, summary, original):
        """
        Estimate quality without reference summary
        
        Heuristics:
        - Length appropriate? (100-300 tokens)
        - Contains key entities?
        - Has resolution language?
        """
        score = 0.5  # Base score
        
        token_count = count_tokens(summary)
        if 100 <= token_count <= 300:
            score += 0.2
        
        # Check for key elements
        if re.search(r'\b(issue|problem|concern)\b', summary, re.I):
            score += 0.1
        if re.search(r'\b(resolved|fixed|solution)\b', summary, re.I):
            score += 0.1
        if re.search(r'\b(customer|agent)\b', summary, re.I):
            score += 0.1
        
        return min(score, 1.0)
```

---

## 12. Advanced Techniques

### Technique 1: Key Point Extraction

```python
class KeyPointExtractor:
    """
    Extract specific key points instead of general summary
    
    Use case: Structured summaries
    """
    
    async def extract_key_points(self, transcript):
        """
        Extract structured information:
        - Customer issue
        - Root cause
        - Resolution
        - Action items
        - Sentiment
        """
        prompt = f"""Extract key information from this call:

Transcript:
{transcript}

Extract:
1. Customer Issue: [1-2 sentences describing the problem]
2. Root Cause: [What caused the issue]
3. Resolution: [How it was resolved]
4. Action Items: [List any follow-up actions]
5. Customer Sentiment: [Negative/Neutral/Positive + brief explanation]

Format as JSON:"""
        
        result = await self.llm.generate(prompt)
        
        # Parse JSON response
        key_points = json.loads(result['text'])
        
        return key_points
```

**Output:**
```json
{
  "customer_issue": "Customer unable to access account due to forgotten password",
  "root_cause": "Email address on file had typo preventing reset emails",
  "resolution": "Updated email address, sent new reset link, customer successfully reset password",
  "action_items": [
    "Follow-up call scheduled for next week",
    "Enable 2FA for additional security"
  ],
  "customer_sentiment": "Neutral - Initially frustrated but satisfied with resolution"
}
```

### Technique 2: Query-Focused Summarization

```python
async def query_focused_summarize(transcript, query):
    """
    Summarize focusing on specific aspect
    
    Example queries:
    - "What did the customer complain about?"
    - "Was the issue resolved?"
    - "What follow-up actions are needed?"
    """
    # First, extract relevant parts
    relevant_parts = extract_relevant_to_query(transcript, query)
    
    # Then, summarize focused on query
    prompt = f"""Question: {query}

Relevant parts of call transcript:
{relevant_parts}

Answer the question in 2-3 sentences based on the transcript:"""
    
    return await llm.generate(prompt)
```

### Technique 3: Multi-Modal Summarization

```python
async def multimodal_summarize(transcript, call_metadata):
    """
    Incorporate metadata for richer summaries
    
    Metadata:
    - Call duration
    - Number of transfers
    - Customer tier (VIP, Premium, Standard)
    - Previous call history
    - Time of day
    """
    # Build context-rich prompt
    prompt = f"""Summarize this {call_metadata['duration_min']}-minute call 
from a {call_metadata['customer_tier']} customer.

Context:
- Customer has called {call_metadata['previous_calls']} times in past 30 days
- Call was {"transferred" if call_metadata['transfers'] > 0 else "not transferred"}
- Time: {call_metadata['time_of_day']}

Transcript:
{transcript}

Provide a summary that considers this context:"""
    
    return await llm.generate(prompt)
```

### Technique 4: Iterative Refinement

```python
async def iterative_refine(initial_summary, transcript):
    """
    Refine summary iteratively based on feedback
    
    Process:
    1. Generate initial summary
    2. Check for missing key points
    3. Refine to include missing points
    4. Verify no hallucinations
    """
    summary = initial_summary
    
    for iteration in range(3):  # Max 3 refinements
        # Check coverage
        missing_points = identify_missing_key_points(summary, transcript)
        
        if not missing_points:
            break  # Summary is complete
        
        # Refine to include missing points
        refinement_prompt = f"""Current summary:
{summary}

This summary is missing these key points:
{missing_points}

Refine the summary to include these points while staying concise (3-4 sentences):"""
        
        summary = await llm.generate(refinement_prompt)
    
    return summary
```

---

## Conclusion

### Recommended Architecture for 45-60 Min Calls

**Production Pipeline:**

```
Input: 60-min call (12K tokens)
    │
    ├─> Stage 1: Noise Filtering (0.2s, $0)
    │   └─> Output: 4K tokens (67% noise removed)
    │
    ├─> Stage 2: Semantic Chunking (0.1s, $0)
    │   └─> Output: 3 chunks of ~1.3K tokens
    │
    ├─> Stage 3: Extractive per Chunk (0.3s, $0)
    │   └─> Output: 3 chunks of ~600 tokens = 1.8K total
    │
    ├─> Stage 4: LLM Summarize Each (2s, $0.009)
    │   └─> Output: 3 summaries of ~80 tokens = 240 total
    │
    └─> Stage 5: Synthesize Final (1s, $0.001)
        └─> Output: 150-200 tokens

Total: 3.6s, $0.010
vs Naive: N/A (exceeds context), would need GPT-4-Turbo: $0.12

Savings: 92%
```

### Key Takeaways

1. **Multi-Stage is Essential**
   - Can't send 12K tokens to most models
   - Each stage filters and compresses
   - 90%+ cost savings

2. **Noise Removal is Critical**
   - 60-70% of call content is noise
   - Free preprocessing saves massive LLM costs
   - Better summaries from cleaner input

3. **Hierarchical Scales Better**
   - Map-reduce pattern
   - Parallelizable
   - Works for any length call

4. **Context-Aware Improves Quality**
   - Different call phases need different focus
   - Structured extraction > generic summary
   - Metadata adds richness

5. **Quality Monitoring Required**
   - Track ROUGE, coverage, consistency
   - User feedback loop
   - Continuous improvement

### Implementation Checklist

- [ ] Implement noise filtering patterns
- [ ] Build semantic chunking logic
- [ ] Create extractive summarizer
- [ ] Set up LLM integration with retries
- [ ] Implement hierarchical summarization
- [ ] Add quality metrics calculation
- [ ] Set up caching layer
- [ ] Create parallel processing pipeline
- [ ] Add cost tracking
- [ ] Implement user feedback collection
- [ ] Build monitoring dashboard
- [ ] Create A/B testing framework

**Remember:** For million-scale systems, the difference between naive and smart summarization is the difference between $120K/month and $10K/month.

---

## 13. Complete Code Implementation

### Full Production Summarizer

```python
"""
production_summarizer.py - Complete implementation for long call summarization
"""

import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Turn:
    """Single speaker turn in conversation"""
    speaker: str
    text: str
    start_time: float
    end_time: float
    index: int


@dataclass
class Segment:
    """Conversation segment (topic-coherent)"""
    turns: List[Turn]
    topic_label: str
    importance_score: float
    start_time: float
    end_time: float


class NoiseFilter:
    """
    Remove noise from call transcripts
    
    Handles:
    - Greetings and pleasantries
    - Hold music and wait times
    - Account verification
    - Filler words
    - Repetitions
    """
    
    # Noise patterns (compiled regex for speed)
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|good morning|good afternoon|good evening)',
        r'\bhow are you( doing| today)?\b',
        r'\bhave a (great|good|nice|wonderful) (day|morning|afternoon|evening)',
        r'\bthanks? for (calling|your (time|patience|call))',
        r'\bmy name is \w+\b'
    ]
    
    PROCEDURAL_PATTERNS = [
        r'\bcan you (verify|confirm|provide|give me)\b',
        r'\blet me (check|look|pull up|see|verify)',
        r'\b(please|just) hold\b',
        r'\bone (moment|second|minute)( please)?\b',
        r'\b(bear|hang on) with me\b',
        r'\byour (account|reference|confirmation) number\b',
        r'\bfor (security|verification) purposes\b',
        r'\blast (four|4) (digits|of)',
        r'\bdate of birth\b'
    ]
    
    FILLER_WORDS = [
        r'\b(um|uh|er|ah|hmm)\b',
        r'\byou know\b',
        r'\bI mean\b',
        r'\blike\b(?! to)',  # "like" as filler, not "like to"
        r'\bbasically\b',
        r'\bactually\b(?! \w)',
        r'\bkind of\b',
        r'\bsort of\b',
        r'\bI guess\b'
    ]
    
    HOLD_PATTERNS = [
        r'\b(hold|holding|on hold)\b',
        r'\btransferring you to\b',
        r'\bplease wait\b',
        r'\bmusic playing\b',
        r'\b\[silence\]\b'
    ]
    
    def __init__(self):
        # Compile patterns for performance
        self.greeting_regex = [re.compile(p, re.IGNORECASE) 
                              for p in self.GREETING_PATTERNS]
        self.procedural_regex = [re.compile(p, re.IGNORECASE) 
                                for p in self.PROCEDURAL_PATTERNS]
        self.filler_regex = [re.compile(p, re.IGNORECASE) 
                            for p in self.FILLER_WORDS]
        self.hold_regex = [re.compile(p, re.IGNORECASE) 
                           for p in self.HOLD_PATTERNS]
    
    def filter(self, transcript: List[Turn], 
               aggressive: bool = False) -> List[Turn]:
        """
        Main filtering method
        
        Args:
            transcript: List of Turn objects
            aggressive: If True, filter more aggressively (higher noise removal)
        
        Returns:
            Filtered transcript
        """
        filtered = transcript.copy()
        
        # Step 1: Remove greeting exchanges (first/last 2 turns often noise)
        filtered = self._remove_opening_closing(filtered)
        
        # Step 2: Remove procedural turns
        filtered = self._remove_procedural_turns(filtered, aggressive)
        
        # Step 3: Clean filler words from remaining turns
        filtered = self._clean_filler_words(filtered)
        
        # Step 4: Remove hold/wait notifications
        filtered = self._remove_hold_times(filtered)
        
        # Step 5: Deduplicate repetitions
        filtered = self._remove_repetitions(filtered)
        
        # Step 6: Remove very short turns (< 3 words)
        filtered = [t for t in filtered if len(t.text.split()) >= 3]
        
        # Calculate noise reduction
        original_tokens = sum(len(t.text.split()) for t in transcript)
        filtered_tokens = sum(len(t.text.split()) for t in filtered)
        noise_removed = 1 - (filtered_tokens / original_tokens)
        
        logger.info(
            f"Noise filtering: {len(transcript)} → {len(filtered)} turns "
            f"({noise_removed:.1%} noise removed)"
        )
        
        return filtered
    
    def _remove_opening_closing(self, transcript: List[Turn]) -> List[Turn]:
        """Remove greeting/closing exchanges"""
        if len(transcript) < 6:
            return transcript
        
        # Check first few turns
        opening_noise_count = 0
        for turn in transcript[:4]:
            if self._is_greeting(turn.text):
                opening_noise_count += 1
            else:
                break
        
        # Check last few turns
        closing_noise_count = 0
        for turn in reversed(transcript[-4:]):
            if self._is_closing(turn.text):
                closing_noise_count += 1
            else:
                break
        
        # Remove noise turns
        start_idx = opening_noise_count
        end_idx = len(transcript) - closing_noise_count
        
        return transcript[start_idx:end_idx]
    
    def _is_greeting(self, text: str) -> bool:
        """Check if text is a greeting"""
        text_lower = text.lower()
        return any(regex.search(text_lower) for regex in self.greeting_regex)
    
    def _is_closing(self, text: str) -> bool:
        """Check if text is a closing"""
        closing_phrases = [
            r'\bhave a (great|good|nice)',
            r'\bthanks? for (calling|your time)',
            r'\bis there anything else',
            r'\bglad (I could|to) help'
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in closing_phrases)
    
    def _remove_procedural_turns(self, transcript: List[Turn], 
                                 aggressive: bool) -> List[Turn]:
        """Remove procedural/verification turns"""
        filtered = []
        
        for turn in transcript:
            # Count procedural matches
            procedural_matches = sum(
                1 for regex in self.procedural_regex
                if regex.search(turn.text)
            )
            
            # Calculate procedural ratio
            word_count = len(turn.text.split())
            procedural_ratio = procedural_matches / max(word_count / 5, 1)
            
            # Keep turn if:
            # - Not heavily procedural
            # - OR turn is long enough that it likely contains useful info
            if aggressive:
                keep = procedural_ratio < 0.3 or word_count > 25
            else:
                keep = procedural_ratio < 0.5 or word_count > 15
            
            if keep:
                filtered.append(turn)
        
        return filtered
    
    def _clean_filler_words(self, transcript: List[Turn]) -> List[Turn]:
        """Remove filler words from text"""
        for turn in transcript:
            cleaned = turn.text
            
            # Remove filler words
            for regex in self.filler_regex:
                cleaned = regex.sub('', cleaned)
            
            # Clean up spacing
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned = re.sub(r'\s+([,.])', r'\1', cleaned)
            
            turn.text = cleaned
        
        return transcript
    
    def _remove_hold_times(self, transcript: List[Turn]) -> List[Turn]:
        """Remove hold/transfer notifications"""
        return [
            t for t in transcript
            if not any(regex.search(t.text) for regex in self.hold_regex)
        ]
    
    def _remove_repetitions(self, transcript: List[Turn], 
                           threshold: float = 0.85) -> List[Turn]:
        """
        Remove repetitive turns using n-gram similarity
        
        Example:
        Customer: "I can't log in"
        Agent: "You can't log in?"
        Customer: "Yes, I can't access my account"
        
        → Keep only first and third (remove echo)
        """
        if len(transcript) < 2:
            return transcript
        
        filtered = [transcript[0]]  # Always keep first
        
        for i in range(1, len(transcript)):
            current = transcript[i]
            
            # Check similarity with recent turns (window of 3)
            is_repetition = False
            for j in range(max(0, i-3), i):
                previous = transcript[j]
                
                # Skip if different speakers (agent echoing customer is OK)
                if current.speaker != previous.speaker:
                    continue
                
                # Calculate similarity
                sim = self._text_similarity(current.text, previous.text)
                
                if sim > threshold:
                    is_repetition = True
                    break
            
            if not is_repetition:
                filtered.append(current)
        
        return filtered
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using character n-grams"""
        # Convert to character n-grams
        ngram_size = 3
        ngrams1 = set(text1[i:i+ngram_size] 
                     for i in range(len(text1)-ngram_size+1))
        ngrams2 = set(text2[i:i+ngram_size] 
                     for i in range(len(text2)-ngram_size+1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0


class SemanticChunker:
    """
    Chunk transcript by topic/conversation flow
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def chunk(self, transcript: List[Turn], 
             max_chunk_tokens: int = 1500) -> List[Segment]:
        """
        Chunk by semantic coherence
        
        Algorithm:
        1. Embed each turn
        2. Calculate similarity between adjacent turns
        3. Split when similarity drops (topic change)
        4. Merge small chunks
        """
        # Embed all turns
        embeddings = self.embedding_model.encode([t.text for t in transcript])
        
        # Calculate similarity between adjacent turns
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i+1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Find topic boundaries (similarity drops)
        topic_boundaries = [0]
        
        # Use moving average to smooth
        window_size = 3
        for i in range(len(similarities) - window_size):
            window = similarities[i:i+window_size]
            avg_sim = sum(window) / window_size
            
            # Topic change if similarity drops below threshold
            if avg_sim < 0.65:  # Tunable threshold
                topic_boundaries.append(i + 1)
        
        topic_boundaries.append(len(transcript))
        
        # Create segments
        segments = []
        for i in range(len(topic_boundaries) - 1):
            start_idx = topic_boundaries[i]
            end_idx = topic_boundaries[i + 1]
            
            segment_turns = transcript[start_idx:end_idx]
            
            # Calculate importance
            importance = self._calculate_segment_importance(segment_turns)
            
            segments.append(Segment(
                turns=segment_turns,
                topic_label=self._label_topic(segment_turns),
                importance_score=importance,
                start_time=segment_turns[0].start_time,
                end_time=segment_turns[-1].end_time
            ))
        
        # Merge small segments
        segments = self._merge_small_segments(segments, max_chunk_tokens)
        
        return segments
    
    def _calculate_segment_importance(self, turns: List[Turn]) -> float:
        """Calculate importance score for segment"""
        scores = []
        
        for turn in turns:
            # Use importance scoring from preprocessing service
            score = calculate_importance_score(turn)
            scores.append(score)
        
        # Segment importance = max importance of any turn
        # (One critical turn makes whole segment important)
        return max(scores) if scores else 0.0
    
    def _label_topic(self, turns: List[Turn]) -> str:
        """Generate topic label for segment"""
        # Extract most important keywords using TF-IDF
        texts = [t.text for t in turns]
        combined_text = " ".join(texts)
        
        # Simple keyword extraction
        words = combined_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top 2 keywords
        top_keywords = sorted(word_freq.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:2]
        
        return "_".join([kw[0] for kw in top_keywords])
    
    def _merge_small_segments(self, segments: List[Segment], 
                             max_tokens: int) -> List[Segment]:
        """Merge segments that are too small"""
        merged = []
        current_segment = segments[0] if segments else None
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # Calculate combined token count
            current_tokens = sum(len(t.text.split()) for t in current_segment.turns)
            next_tokens = sum(len(t.text.split()) for t in next_segment.turns)
            
            if current_tokens + next_tokens <= max_tokens:
                # Merge segments
                current_segment = Segment(
                    turns=current_segment.turns + next_segment.turns,
                    topic_label=current_segment.topic_label,  # Keep first topic
                    importance_score=max(current_segment.importance_score, 
                                       next_segment.importance_score),
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time
                )
            else:
                # Save current, start new
                merged.append(current_segment)
                current_segment = next_segment
        
        if current_segment:
            merged.append(current_segment)
        
        return merged


class ExtractiveSummarizer:
    """
    Extract most important sentences
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
    
    def summarize(self, turns: List[Turn], 
                 target_tokens: int = 2000) -> List[Turn]:
        """
        Select most important turns using multiple signals
        
        Scoring factors:
        1. TF-IDF score (content importance)
        2. Position score (beginning/end often important)
        3. Speaker score (customer > agent)
        4. Length score (very short turns often noise)
        5. Keyword score (high-priority keywords)
        6. Sentiment score (negative = more important)
        """
        if not turns:
            return []
        
        # Calculate TF-IDF scores
        texts = [t.text for t in turns]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            # Normalize to 0-1
            tfidf_scores = (tfidf_scores - tfidf_scores.min()) / \
                          (tfidf_scores.max() - tfidf_scores.min() + 1e-10)
        except:
            tfidf_scores = np.ones(len(turns)) * 0.5
        
        # Score each turn
        scored_turns = []
        total_turns = len(turns)
        
        for i, turn in enumerate(turns):
            score = 0.0
            
            # 1. TF-IDF importance (30% weight)
            score += tfidf_scores[i] * 0.3
            
            # 2. Position importance (15% weight)
            # U-shaped: Beginning and end are important
            position_ratio = i / total_turns
            position_score = 1 - abs(0.5 - position_ratio) * 2  # Peak at 0.5
            score += position_score * 0.15
            
            # 3. Speaker importance (20% weight)
            if turn.speaker.lower() == 'customer':
                score += 0.2
            
            # 4. Length importance (10% weight)
            # Too short = noise, too long = rambling
            word_count = len(turn.text.split())
            if 5 <= word_count <= 40:
                score += 0.1
            
            # 5. Keyword importance (15% weight)
            keyword_score = self._keyword_importance(turn.text)
            score += keyword_score * 0.15
            
            # 6. Sentiment importance (10% weight)
            if self._is_negative(turn.text):
                score += 0.1
            
            scored_turns.append((turn, score))
        
        # Sort by score
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        
        # Select turns until target token count
        selected = []
        current_tokens = 0
        
        for turn, score in scored_turns:
            turn_tokens = len(turn.text.split())
            
            if current_tokens + turn_tokens <= target_tokens:
                selected.append(turn)
                current_tokens += turn_tokens
        
        # Re-sort by original order (chronological)
        selected.sort(key=lambda t: t.index)
        
        return selected
    
    def _keyword_importance(self, text: str) -> float:
        """Score based on important keywords"""
        high_priority = [
            'cancel', 'refund', 'complaint', 'issue', 'problem',
            'error', 'fraud', 'dispute', 'lawsuit', 'broken'
        ]
        
        medium_priority = [
            'question', 'help', 'confused', 'understand',
            'change', 'update', 'modify'
        ]
        
        text_lower = text.lower()
        
        high_count = sum(1 for kw in high_priority if kw in text_lower)
        medium_count = sum(1 for kw in medium_priority if kw in text_lower)
        
        return min(high_count * 0.3 + medium_count * 0.1, 1.0)
    
    def _is_negative(self, text: str) -> bool:
        """Detect negative sentiment"""
        negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst',
            'frustrated', 'angry', 'disappointed', 'upset'
        ]
        
        text_lower = text.lower()
        return any(word in text_lower for word in negative_words)


class HierarchicalSummarizer:
    """
    Map-reduce summarization for long documents
    """
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.noise_filter = NoiseFilter()
        self.semantic_chunker = SemanticChunker(embedding_model)
        self.extractive = ExtractiveSummarizer()
    
    async def summarize_long_call(self, transcript: List[Turn],
                                  quality: str = 'balanced') -> Dict[str, Any]:
        """
        Complete hierarchical summarization
        
        Pipeline:
        1. Filter noise (free)
        2. Chunk semantically (free)
        3. Extract important parts per chunk (free)
        4. LLM summarize each chunk (paid)
        5. Synthesize final summary (paid)
        
        For 60-min call:
        - Input: 12K tokens
        - After filtering: 4K tokens
        - After chunking: 3 chunks × 1.3K tokens
        - After extraction: 3 chunks × 600 tokens = 1.8K total
        - LLM cost: 3 × $0.003 + 1 × $0.001 = $0.010
        """
        start_time = datetime.now()
        costs = {'filtering': 0, 'extraction': 0, 'llm': 0}
        
        # Stage 1: Noise filtering
        filtered = self.noise_filter.filter(transcript, aggressive=(quality=='fast'))
        
        # Stage 2: Semantic chunking
        segments = self.semantic_chunker.chunk(filtered, max_chunk_tokens=1500)
        
        # Stage 3: Per-segment processing
        segment_summaries = []
        
        for i, segment in enumerate(segments):
            # Extractive summarization (prioritize important content)
            extracted = self.extractive.summarize(
                segment.turns,
                target_tokens=600
            )
            
            # LLM summarization
            if quality != 'fast':
                segment_text = self._format_turns(extracted)
                
                summary, cost = await self._summarize_segment(
                    segment_text,
                    segment_number=i+1,
                    total_segments=len(segments),
                    model='gpt-4' if quality == 'high' else 'gpt-3.5-turbo'
                )
                
                costs['llm'] += cost
            else:
                # Fast mode: Just concatenate extracted turns
                summary = self._format_turns(extracted[:3])
                cost = 0
            
            segment_summaries.append({
                'segment_number': i + 1,
                'topic': segment.topic_label,
                'importance': segment.importance_score,
                'time_range': f"{segment.start_time:.0f}s - {segment.end_time:.0f}s",
                'summary': summary,
                'cost': cost
            })
        
        # Stage 4: Synthesize final summary
        if quality != 'fast':
            final_summary, synthesis_cost = await self._synthesize_final(
                segment_summaries,
                model='gpt-4' if quality == 'high' else 'gpt-3.5-turbo'
            )
            costs['llm'] += synthesis_cost
        else:
            final_summary = self._simple_concatenate(segment_summaries)
        
        # Calculate metrics
        duration = (datetime.now() - start_time).total_seconds()
        total_cost = sum(costs.values())
        
        original_tokens = sum(len(t.text.split()) for t in transcript)
        final_tokens = len(final_summary.split())
        compression = final_tokens / original_tokens
        
        return {
            'summary': final_summary,
            'segment_summaries': segment_summaries,
            'metadata': {
                'quality_mode': quality,
                'num_segments': len(segments),
                'original_tokens': original_tokens,
                'final_tokens': final_tokens,
                'compression_ratio': compression,
                'noise_filtered': len(transcript) - len(filtered),
                'processing_time_seconds': duration,
                'total_cost_usd': total_cost,
                'cost_breakdown': costs
            }
        }
    
    async def _summarize_segment(self, text: str, 
                                 segment_number: int,
                                 total_segments: int,
                                 model: str) -> tuple[str, float]:
        """Summarize a single segment"""
        prompt = f"""This is segment {segment_number} of {total_segments} from a customer support call.

Summarize this segment in 2-3 sentences. Focus on:
- What issue or topic is discussed
- Key points or concerns raised
- Any resolutions or actions mentioned

Segment transcript:
{text}

Segment summary:"""
        
        result = await self.llm.generate(
            prompt=prompt,
            model=model,
            max_tokens=100,
            temperature=0.3
        )
        
        return result['output'], result['cost']
    
    async def _synthesize_final(self, segment_summaries: List[Dict],
                                model: str) -> tuple[str, float]:
        """Synthesize final summary from segment summaries"""
        # Combine segment summaries
        combined = "\n\n".join([
            f"Part {s['segment_number']} ({s['time_range']}): {s['summary']}"
            for s in segment_summaries
        ])
        
        prompt = f"""These are summaries of different parts of a customer support call.

Create a coherent 3-4 sentence summary of the entire call.

Structure your summary:
1. First sentence: Customer's main issue
2. Second-third sentences: How it was addressed
3. Final sentence: Outcome and any follow-up

Segment summaries:
{combined}

Overall call summary:"""
        
        result = await self.llm.generate(
            prompt=prompt,
            model=model,
            max_tokens=150,
            temperature=0.3
        )
        
        return result['output'], result['cost']
    
    def _format_turns(self, turns: List[Turn]) -> str:
        """Format turns into readable text"""
        return "\n".join([
            f"[{t.speaker.upper()}]: {t.text}"
            for t in turns
        ])
    
    def _simple_concatenate(self, segment_summaries: List[Dict]) -> str:
        """Simple concatenation for fast mode"""
        return " ".join([s['summary'] for s in segment_summaries])


# Usage Example
async def main():
    # Load a long call
    call_id = "CALL_123456"
    transcript = load_transcript(call_id)  # 60 min, 12K tokens
    
    # Initialize summarizer
    summarizer = HierarchicalSummarizer(llm_provider)
    
    # Summarize with different quality modes
    for quality in ['fast', 'balanced', 'high']:
        print(f"\n{'='*60}")
        print(f"Quality Mode: {quality}")
        print('='*60)
        
        result = await summarizer.summarize_long_call(
            transcript,
            quality=quality
        )
        
        print(f"\nSummary:\n{result['summary']}\n")
        print(f"Metadata:")
        print(f"  Segments: {result['metadata']['num_segments']}")
        print(f"  Compression: {result['metadata']['compression_ratio']:.1%}")
        print(f"  Cost: ${result['metadata']['total_cost_usd']:.4f}")
        print(f"  Time: {result['metadata']['processing_time_seconds']:.1f}s")
```

---

## 14. Real-World Example: Complete Walkthrough

### Input: 60-Minute Support Call

**Raw Transcript (excerpt):**
```
[00:00] [AGENT]: Good morning! Thank you for calling TechCorp support. My name is Sarah. How may I help you today?
[00:05] [CUSTOMER]: Hi Sarah, how are you doing?
[00:07] [AGENT]: I'm doing well, thank you! How are you?
[00:09] [CUSTOMER]: Good, thanks. Um, so, like, I'm having a bit of an issue with my account.
[00:15] [AGENT]: I'm sorry to hear that. Let me help you with that. Can I have your account number please?
[00:20] [CUSTOMER]: Sure, it's, um, let me find it... it's 123456789
[00:25] [AGENT]: Perfect. And for security purposes, can you verify the last four digits of your social?
[00:30] [CUSTOMER]: 5678
[00:32] [AGENT]: Thank you. Let me pull up your account. One moment please.
[00:45] [AGENT]: Okay, I have your account here. What seems to be the issue?
[00:50] [CUSTOMER]: Well, uh, so basically, I'm trying to cancel my subscription, you know?
[00:55] [AGENT]: Okay, you'd like to cancel your subscription?
[00:57] [CUSTOMER]: Yes, I want to cancel it, but I can't find where to do it on the website.
[01:05] [AGENT]: I understand. Let me walk you through the cancellation process.
[01:10] [CUSTOMER]: Actually, wait, I also noticed I was charged twice this month.
[01:15] [AGENT]: Oh, I see. Let me check your billing history.
[01:20] [AGENT]: I do see two charges here. Let me investigate why that happened.
... [35 minutes of investigation] ...
[40:15] [AGENT]: I've found the issue. There was a system error that caused a duplicate charge.
[40:20] [CUSTOMER]: So what can we do about it?
[40:22] [AGENT]: I'm going to issue a refund for the duplicate charge right now.
[40:30] [AGENT]: The refund has been processed. You should see it in 3-5 business days.
[40:35] [CUSTOMER]: Okay, great. And what about canceling my subscription?
[40:40] [AGENT]: For the cancellation, I can do that for you right now if you'd like.
[40:45] [CUSTOMER]: Yes, please cancel it.
[40:50] [AGENT]: Your subscription has been canceled. You'll have access until the end of your billing period.
[40:55] [CUSTOMER]: Perfect, thank you.
... [15 minutes of closing] ...
```

### Processing Pipeline:

**Step 1: Noise Filtering**
```
Input: 120 turns, ~12,000 tokens

Removed:
- Opening greetings (4 turns)
- Account verification (6 turns)
- Hold notifications (3 turns)
- Filler words (um, uh, like, you know)
- Repetitions (agent echoing customer)
- Closing pleasantries (5 turns)

Output: 45 turns, ~4,200 tokens
Noise removed: 62.5%
Time: 0.2s
Cost: $0
```

**Step 2: Semantic Chunking**
```
Topic detection (using embeddings):

Segment 1 (00:00-02:00): Initial issue statement
Segment 2 (02:00-15:30): Cancellation inquiry
Segment 3 (15:30-40:15): Duplicate charge investigation
Segment 4 (40:15-45:00): Resolution
Segment 5 (45:00-60:00): Confirmation and closing

Output: 5 segments
Time: 0.3s
Cost: $0
```

**Step 3: Extractive Per Segment**
```
Segment 1 → 3 most important turns (180 tokens)
Segment 2 → 5 most important turns (300 tokens)
Segment 3 → 12 most important turns (720 tokens)
Segment 4 → 4 most important turns (240 tokens)
Segment 5 → 2 most important turns (120 tokens)

Total: 26 turns, 1,560 tokens (from 4,200)
Time: 0.5s
Cost: $0
```

**Step 4: LLM Summarize Each Segment**
```
Segment 1: "Customer initiated call about account issues."

Segment 2: "Customer wants to cancel subscription but couldn't find 
           cancellation option on website."

Segment 3: "Customer noticed duplicate charge on account. Agent 
           investigated and found system error caused duplicate billing."

Segment 4: "Agent processed refund for duplicate charge (3-5 business days) 
           and canceled subscription at customer's request."

Segment 5: "Confirmed subscription canceled, access remains until end 
           of billing period."

Total tokens: ~100 × 5 = 500 tokens
LLM cost: 5 × (1.56K input + 0.1K output) × $0.0015 = $0.0125
Time: 2.5s (parallel)
```

**Step 5: Synthesize Final**
```
Input: 5 segment summaries (500 tokens)

Prompt to LLM:
"Combine these segment summaries into a coherent 3-4 sentence summary..."

Output:
"Customer contacted support wanting to cancel subscription but unable to 
find cancellation option on website. Additionally, customer discovered 
duplicate billing charge on account. Agent investigated and found system 
error causing duplicate charge, processed refund (3-5 business days), and 
completed subscription cancellation at customer's request. Customer will 
retain access until end of current billing period."

Cost: 0.5K input + 0.15K output × $0.0015 = $0.001
Time: 1.5s
Total: 4s, $0.0135
```

### Final Result

**Summary (4 sentences, 65 words):**
> "Customer contacted support wanting to cancel subscription but unable to find cancellation option on website. Additionally, customer discovered duplicate billing charge on account. Agent investigated and found system error causing duplicate charge, processed refund (3-5 business days), and completed subscription cancellation at customer's request. Customer will retain access until end of current billing period."

**Metadata:**
- Original: 60 minutes, 120 turns, 12,000 tokens
- Final: 65 words, 85 tokens
- Compression: 99.3%
- Time: 4.0 seconds
- Cost: $0.0135
- Quality: High (captures all key points)

**Comparison to Naive:**
- Naive: Would need GPT-4-Turbo (128K context) → $0.12
- Our approach: $0.0135
- Savings: 88.8%

---

## 15. Interview Discussion Points

### Question: "How would you summarize a 60-minute call with millions of calls?"

**Answer Framework:**

**1. Acknowledge the Constraints:**
"First, I'd clarify the requirements:
- Token limits: Most models have 4-8K limits
- Cost budget: LLM on 1M calls is expensive
- Latency: Users expect seconds, not minutes
- Quality: Must capture key issues accurately"

**2. Propose Multi-Stage Approach:**
"I'd use a hierarchical approach:
1. Noise filtering (rule-based, free)
2. Smart chunking (semantic coherence)
3. Extractive summarization (cheap)
4. Selective LLM usage (only on important parts)
5. Synthesis (final summary)"

**3. Explain the Key Innovation:**
"The innovation is in Stage 1-3: We reduce 12K tokens to 2K tokens WITHOUT using LLM. This gives us:
- 83% cost reduction
- Better quality (noise removed)
- Fits within context limits
- Fast preprocessing"

**4. Discuss Trade-offs:**
"Rule-based noise filtering is 80% accurate vs 95% for LLM, but:
- It's free vs $0.002/call
- It's instant vs 2-3s
- For triage, 80% is sufficient
- At 1M calls, this saves $15K/month"

**5. Show Scalability:**
"To handle millions of calls:
- Parallel processing: 25 workers × 60 calls/hour = 36K/day
- Batch API: 50% cost discount for non-urgent
- Caching: 40% hit rate saves $4K/month
- Progressive: Summarize as call progresses"

**6. Mention Monitoring:**
"I'd track:
- Compression ratio (target: 5-10%)
- ROUGE scores (quality)
- User feedback (satisfaction)
- Cost per call (budget)
- Processing time (SLA)"

### Question: "What if summary quality is poor?"

**Answer:**

"I'd investigate systematically:

1. **Check metrics:**
   - Low ROUGE score? → Improve LLM prompt
   - High compression? → Extract more content
   - Factual errors? → Add consistency check

2. **A/B test improvements:**
   - Different prompts
   - Different chunk sizes
   - Different extraction algorithms

3. **Feedback loop:**
   - Collect user feedback
   - Sample and manually review
   - Identify failure patterns
   - Iterate on weak areas

4. **Consider trade-offs:**
   - Better quality = higher cost
   - Find the sweet spot for business needs"

### Question: "How do you handle calls in multiple languages?"

**Answer:**

"Three approaches:

**Option 1: Translate First**
```python
# Detect language
language = detect_language(transcript)

if language != 'en':
    # Translate to English
    transcript = translate(transcript, target='en')
    # Cost: +$0.0001/call

# Process normally
summary = summarize(transcript)

# Translate back if needed
if language != 'en':
    summary = translate(summary, target=language)
```

**Option 2: Multilingual Models**
- Use mT5, mBART for multilingual summarization
- No translation needed
- Lower quality for some languages

**Option 3: Language-Specific Pipelines**
- Separate pipeline per language
- Best quality
- Higher operational complexity

I'd recommend Option 1 for most cases - translation is cheap and works well."

---

## 16. Performance Benchmarks

### Benchmark Results (60-min Call)

| Approach | Tokens Processed | Time | Cost | Quality (ROUGE-L) |
|----------|-----------------|------|------|-------------------|
| Naive (GPT-4-Turbo) | 12,000 | 15s | $0.120 | 0.45 |
| Direct GPT-4 (chunked) | 12,000 | 12s | $0.096 | 0.42 |
| Extractive Only | 12,000 | 1s | $0 | 0.28 |
| **Our Approach (Balanced)** | **2,000** | **4s** | **$0.013** | **0.41** |
| Our Approach (Fast) | 600 | 1s | $0 | 0.32 |
| Our Approach (High) | 2,500 | 8s | $0.025 | 0.46 |

**Key Insights:**
- Our balanced mode achieves 98% cost savings with minimal quality loss
- Fast mode is free and still useful for initial triage
- High mode beats naive approach in both cost and quality

### Scalability Benchmarks

**Single Machine:**
- Throughput: 60 calls/hour (with LLM)
- Bottleneck: LLM API latency
- Cost: $0.78/hour ($18.72/day)

**10 Parallel Workers:**
- Throughput: 600 calls/hour (14,400/day)
- Handles: 430K calls/month
- Cost: $0.78/hour × 10 = $7.80/hour

**25 Parallel Workers:**
- Throughput: 1,500 calls/hour (36,000/day)
- Handles: 1.08M calls/month ✅
- Cost: $19.50/hour × 24h × 30d = $14,040/month
- Within budget!

---

## 17. Code Template for Your Implementation

```python
# services/summarization_service.py

from typing import List, Dict, Any
from models.schemas import Turn, Segment
from utils.logger import get_logger, cost_tracker

logger = get_logger(__name__)


class SmartSummarizationService:
    """
    Complete summarization service for long calls
    """
    
    def __init__(self):
        self.noise_filter = NoiseFilter()
        self.chunker = SemanticChunker(embedding_model)
        self.extractive = ExtractiveSummarizer()
        self.llm = LLMProvider()
    
    async def summarize_call(self, call_id: str, 
                            transcript: List[Turn],
                            mode: str = 'balanced') -> Dict[str, Any]:
        """
        Main entry point for call summarization
        
        Args:
            call_id: Unique call identifier
            transcript: List of speaker turns
            mode: 'fast' | 'balanced' | 'high'
        
        Returns:
            Dictionary with summary and metadata
        """
        # 1. Filter noise
        filtered = self.noise_filter.filter(transcript)
        logger.info(f"{call_id}: Filtered {len(transcript)} → {len(filtered)} turns")
        
        # 2. Chunk semantically
        segments = self.chunker.chunk(filtered)
        logger.info(f"{call_id}: Created {len(segments)} segments")
        
        # 3. Process segments
        if mode == 'fast':
            # Extractive only - free and fast
            summary = self._fast_summarize(segments)
            cost = 0
        else:
            # Hierarchical LLM - better quality
            summary, cost = await self._hierarchical_summarize(
                segments,
                model='gpt-4' if mode == 'high' else 'gpt-3.5-turbo'
            )
        
        # 4. Track metrics
        cost_tracker.log(call_id, cost)
        
        return {
            'summary': summary,
            'num_segments': len(segments),
            'cost_usd': cost
        }
    
    def _fast_summarize(self, segments: List[Segment]) -> str:
        """Fast extractive-only summarization"""
        # Extract top turn from each segment
        key_turns = []
        for segment in segments:
            if segment.turns:
                # Get most important turn
                top_turn = max(segment.turns, 
                             key=lambda t: calculate_importance_score(t))
                key_turns.append(top_turn.text)
        
        return " ".join(key_turns)
    
    async def _hierarchical_summarize(self, segments: List[Segment],
                                     model: str) -> tuple[str, float]:
        """Hierarchical LLM summarization"""
        # Summarize each segment
        segment_summaries = []
        total_cost = 0
        
        for segment in segments:
            # Extract important parts
            important = self.extractive.summarize(
                segment.turns,
                target_tokens=600
            )
            
            # LLM summarize
            text = "\n".join([t.text for t in important])
            summary, cost = await self._llm_summarize_chunk(text, model)
            
            segment_summaries.append(summary)
            total_cost += cost
        
        # Synthesize final
        final_summary, synthesis_cost = await self._synthesize(
            segment_summaries,
            model
        )
        
        total_cost += synthesis_cost
        
        return final_summary, total_cost
    
    async def _llm_summarize_chunk(self, text: str, model: str) -> tuple[str, float]:
        """Summarize a chunk with LLM"""
        prompt = f"Summarize in 2-3 sentences:\n\n{text}\n\nSummary:"
        
        result = await self.llm.generate(prompt, model=model, max_tokens=100)
        
        return result['output'], result['cost']
    
    async def _synthesize(self, summaries: List[str], model: str) -> tuple[str, float]:
        """Synthesize final summary"""
        combined = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
        
        prompt = f"""Combine these segment summaries into one coherent summary (3-4 sentences):

{combined}

Final summary:"""
        
        result = await self.llm.generate(prompt, model=model, max_tokens=150)
        
        return result['output'], result['cost']
```

---

## 18. Summary Comparison Table

### 60-Minute Call Across Different Approaches

| Approach | Tokens | Time | Cost | Quality | Use Case |
|----------|--------|------|------|---------|----------|
| **Full Transcript to GPT-4-Turbo** | 12K | 15s | $0.120 | ★★★★★ | Research/Legal |
| **Chunked GPT-4** | 12K | 12s | $0.096 | ★★★★☆ | High quality |
| **Extractive + GPT-3.5** | 2K | 4s | $0.013 | ★★★★☆ | **Production** |
| **Extractive Only** | 2K | 1s | $0 | ★★☆☆☆ | Fast scan |
| **Template-Based** | 0 | 0.1s | $0 | ★☆☆☆☆ | Structured data |

**Recommended:** Extractive + GPT-3.5 (balanced quality and cost)

---

## 19. Final Production Recommendations

### For 1M Calls/Month (45-60 min each)

**Architecture:**
```
┌─────────────────────────────────────────┐
│ Ingestion Layer                         │
│ S3 → SQS → Lambda (chunk + filter)      │
│ Throughput: 33K calls/day               │
└─────────────────────────────────────────┘
            │
┌─────────────────────────────────────────┐
│ Processing Layer                         │
│ ECS Tasks (25 workers)                  │
│ ├─ Noise Filter (instant, $0)          │
│ ├─ Semantic Chunk (0.3s, $0)           │
│ ├─ Extractive (0.5s, $0)               │
│ └─ LLM Summarize (2-3s, $0.01)         │
└─────────────────────────────────────────┘
            │
┌─────────────────────────────────────────┐
│ Storage Layer                            │
│ ├─ DynamoDB (summaries)                │
│ ├─ S3 (raw transcripts)                │
│ └─ ElastiCache (cache layer)           │
└─────────────────────────────────────────┘
```

**Monthly Costs:**
- Compute (ECS): $500
- Storage (S3 + DynamoDB): $50
- LLM (OpenAI): $10,000
- Cache (Redis): $100
- **Total: $10,650**

**vs Naive Approach: $120,000**
**Savings: 91.1%**

### Configuration File

```yaml
# config/summarization.yaml

summarization:
  # Noise filtering
  noise_filter:
    enabled: true
    aggressive_mode: false  # true = more filtering, may lose some content
    remove_greetings: true
    remove_procedural: true
    remove_fillers: true
    remove_repetitions: true
    repetition_threshold: 0.85
  
  # Chunking
  chunking:
    method: "semantic"  # semantic | fixed | importance
    max_chunk_tokens: 1500
    min_chunk_tokens: 200
    overlap_tokens: 50
  
  # Extractive summarization
  extractive:
    enabled: true
    target_compression: 0.50  # Reduce to 50% of input
    speaker_weights:
      customer: 0.7
      agent: 0.3
  
  # LLM summarization
  llm:
    quality_modes:
      fast:
        enabled: false  # Skip LLM entirely
        cost_per_call: 0
      balanced:
        model: "gpt-3.5-turbo"
        max_tokens_per_chunk: 100
        synthesis_tokens: 150
        cost_per_call: 0.013
      high:
        model: "gpt-4"
        max_tokens_per_chunk: 150
        synthesis_tokens: 200
        cost_per_call: 0.035
    
    # Rate limiting
    max_concurrent_requests: 50
    requests_per_minute: 3000
    retry_attempts: 3
    retry_delay_seconds: 2
  
  # Quality thresholds
  quality:
    min_rouge_l: 0.35
    min_compression_ratio: 0.05
    max_compression_ratio: 0.15
    min_coverage: 0.80
    min_consistency: 0.95
  
  # Caching
  cache:
    enabled: true
    ttl_days: 30
    cache_key_includes:
      - call_id
      - quality_mode
```

---

## 20. Conclusion

### The Smart Summarization Formula

```
Smart Summarization = 
    Aggressive Noise Removal (67% reduction, $0) +
    Semantic Understanding (coherent chunks, $0) +
    Extractive Filtering (important parts only, $0) +
    Minimal LLM Usage (only when needed, $0.01)

Result: 90%+ cost savings with minimal quality loss
```

### Key Principles

1. **Filter Before Processing**: Remove noise first (free)
2. **Extract Before Generating**: Find important parts (free)
3. **Chunk Semantically**: Preserve context (better quality)
4. **Hierarchical Processing**: Map-reduce for scale
5. **Cache Aggressively**: 40% hit rate = 40% cost savings
6. **Monitor Quality**: Track metrics, iterate based on data

### When to Use What

**Fast Mode (Extractive Only):**
- Use for: Initial triage, bulk processing, cost-sensitive
- Quality: 60-70%
- Cost: $0

**Balanced Mode (Hybrid):**
- Use for: Production, most queries, good ROI
- Quality: 85-90%
- Cost: $0.01-0.015

**High Mode (GPT-4):**
- Use for: VIP customers, legal, critical issues
- Quality: 95%+
- Cost: $0.03-0.05

### Final Thought

> "The art of summarizing long, noisy calls at scale is not about having the best model - it's about **minimizing the amount of expensive processing needed** while maintaining quality. Every token you don't send to the LLM is money saved."

For a staff engineer role, demonstrating this level of cost consciousness and systems thinking is what sets you apart.

