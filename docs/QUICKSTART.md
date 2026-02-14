# Quick Start Guide

Get the Enterprise Call Intelligence Platform running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- 2GB+ RAM
- OpenAI API key (optional - for LLM features)

## Installation

### 1. Clone and Setup

```bash
# Navigate to the project directory
cd Enterprise-Call-Intelligence-Platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key (optional)
# nano .env
# OPENAI_API_KEY=sk-...
```

## Run the System

### Option 1: Full Demo with LLM (requires API key)

```bash
# Make scripts executable (first time only)
chmod +x run_pipeline.sh

# Run complete pipeline (generates 1000 calls, processes with LLM)
./run_pipeline.sh --num-calls 1000

# OR using make:
make run-pipeline

# OR direct command:
PYTHONPATH=. python pipelines/orchestrator.py --num-calls 1000
```

This will:
1. Generate 1000 dummy call transcripts
2. Ingest and preprocess them
3. Enrich important calls with LLM
4. Build search indices

**Expected time:** 10-15 minutes  
**Expected cost:** ~$1.50

### Option 2: Fast Demo without LLM (free!)

```bash
# Run pipeline without expensive LLM enrichment
./run_pipeline.sh --num-calls 100 --skip-enrichment

# OR using make:
make run-pipeline-fast

# OR direct command:
PYTHONPATH=. python pipelines/orchestrator.py --num-calls 100 --skip-enrichment
```

This will:
1. Generate 100 dummy calls
2. Preprocess and score importance
3. Skip LLM enrichment
4. Build indices

**Expected time:** 1-2 minutes  
**Expected cost:** $0

## Start Services

### Terminal 1: API Server

```bash
# Make script executable (first time only)
chmod +x run_api.sh

# Run API
./run_api.sh

# OR using make:
make run-api

# OR direct command:
PYTHONPATH=. python -m api.main
```

API will be available at: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Terminal 2: UI Dashboard

```bash
# Make script executable (first time only)
chmod +x run_ui.sh

# Run UI
./run_ui.sh

# OR using make:
make run-ui

# OR direct command:
PYTHONPATH=. streamlit run ui/streamlit_app.py
```

UI will open automatically at: `http://localhost:8501`

## Try It Out!

### In the UI (Streamlit)

1. Go to `http://localhost:8501`
2. Navigate to "💬 Chat" tab
3. Try these queries:

**Analytics (Fast, Free):**
- "How many calls about credit cards?"
- "What are the top complaints?"
- "Show me call trends"

**Search (Fast, Cheap):**
- "Find calls mentioning refunds"
- "Show me calls about account access"

**Insights (Slower, More Expensive - needs LLM):**
- "What are customers saying about fees?"
- "Summarize common problems from premium customers"

### Via API

```bash
# Health check
curl http://localhost:8000/health

# Submit a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many calls about credit cards?",
    "mode": "cheap"
  }'

# Get analytics summary
curl http://localhost:8000/analytics/summary

# Get system statistics
curl http://localhost:8000/statistics
```

## Using Make Commands

```bash
# Setup environment
make setup

# Run full pipeline
make run-pipeline

# Run fast pipeline (no LLM)
make run-pipeline-fast

# Start API
make run-api

# Start UI
make run-ui

# Run with Docker
make docker-up
```

## What to Expect

### After Running the Pipeline

You'll see:
```
===========================================================
PIPELINE COMPLETE
===========================================================
Total duration: 120.5s

System is ready for queries!
  - Start API: python -m api.main
  - Start UI: streamlit run ui/streamlit_app.py
===========================================================
```

### In the UI Dashboard

**Chat Tab:**
- Ask natural language questions
- See answers with sources
- View cost and performance metrics

**Analytics Tab:**
- Call volume metrics
- Top complaints
- Product distribution
- Sentiment trends

**Metrics Tab:**
- Processing pipeline status
- Cost breakdown by model
- Importance score distribution
- System statistics

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
pwd
# Should show: .../Enterprise-Call-Intelligence-Platform

# Make sure virtual environment is activated
which python
# Should show: .../venv/bin/python
```

### Database Errors

```bash
# Remove and recreate database
rm -rf storage/structured/*.db
python pipelines/orchestrator.py --num-calls 100 --skip-enrichment
```

### Memory Issues

```bash
# Use smaller dataset
python pipelines/orchestrator.py --num-calls 100
```

### API Connection Issues

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use different port
uvicorn api.main:app --port 8001
```

## Next Steps

1. **Explore the code:**
   - Start with `services/preprocessing_service.py` (importance scoring)
   - Then `services/query_router.py` (intelligent routing)
   - Finally `services/rag_service.py` (RAG pipeline)

2. **Experiment with queries:**
   - Try different query types
   - Compare costs between modes (cheap/balanced/deep)
   - Check the sources and citations

3. **Understand the architecture:**
   - Read `docs/ARCHITECTURE.md` for deep dive
   - Review `README.md` for overview
   - Check code comments for design reasoning

4. **Customize:**
   - Adjust importance thresholds in `config/default.yaml`
   - Add custom keywords for your domain
   - Modify the data generator for your use case

## Common Use Cases

### Generate More Data

```bash
python pipelines/data_generator.py --num-calls 5000
```

### Reprocess Existing Data

```bash
# Just preprocessing (fast)
python -c "
import asyncio
from services.ingestion_service import IngestionService
from services.preprocessing_service import PreprocessingService

async def main():
    ing = IngestionService()
    prep = PreprocessingService()
    calls = ing.get_unprocessed_calls()
    for call in calls:
        await prep.preprocess_call(call['call_id'])

asyncio.run(main())
"
```

### Test Individual Services

```bash
# Test preprocessing
python services/preprocessing_service.py

# Test enrichment
python services/enrichment_service.py

# Test query router
python services/query_router.py
```

## Performance Expectations

### With 1000 Calls:

**Pipeline:**
- Ingestion: ~10 seconds
- Preprocessing: ~1 minute
- Enrichment: ~8 minutes
- Indexing: ~30 seconds
- **Total: ~10 minutes**

**Queries:**
- Analytics: 50-100ms
- Search: 200-500ms
- RAG (with LLM): 2-5 seconds

### Cost for 1000 Calls:

- Preprocessing: $0
- Enrichment (15% = 150 calls): ~$0.30
- Indexing: ~$0.10
- **Total: ~$0.40**

## Need Help?

- Check `README.md` for full documentation
- Read `docs/ARCHITECTURE.md` for design details
- Open an issue on GitHub
- Review code comments for inline explanations

---

**Ready?** Start with:
```bash
make run-pipeline-fast && make run-ui
```

Then open `http://localhost:8501` and ask your first question!
