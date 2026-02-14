"""
API Gateway - FastAPI Application
==================================

RESTful API for the Enterprise Call Intelligence Platform.

Endpoints:
---------
- POST /query - Submit natural language query
- GET /analytics - Get analytics summary
- GET /calls/{call_id} - Get call details
- GET /health - Health check
- GET /metrics - Get system metrics

System Design:
-------------
- Async endpoints for high concurrency
- Request validation with Pydantic
- Error handling and logging
- Rate limiting (TODO)
- Authentication (TODO)
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.schemas import UserQuery, ProcessingMode, QueryResponse
from services.query_router import QueryRouter
from services.analytics_service import AnalyticsService
from utils.logger import get_logger, cost_tracker, metrics_collector
from utils.config_loader import config

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Call Intelligence Platform",
    description="AI-powered call transcript analysis with cost-optimized processing",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
query_router = QueryRouter()
analytics_service = AnalyticsService()


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model"""

    query: str
    mode: Optional[str] = "balanced"  # cheap, balanced, deep
    filters: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime
    version: str


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Enterprise Call Intelligence Platform API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    In production, would check:
    - Database connectivity
    - Vector store availability
    - LLM API status
    """
    return HealthResponse(status="healthy", timestamp=datetime.now(), version="1.0.0")


@app.post("/query", response_model=Dict[str, Any])
async def submit_query(request: QueryRequest):
    """
    Submit a natural language query.

    This is the MAIN endpoint users interact with.

    Process:
    -------
    1. Validate request
    2. Create UserQuery object
    3. Route to appropriate service
    4. Return response with cost tracking

    Example requests:
    ----------------
    {
        "query": "How many calls about credit cards?",
        "mode": "cheap"
    }

    {
        "query": "What are customers saying about fees?",
        "mode": "deep"
    }
    """
    try:
        # Validate mode
        mode_mapping = {
            "cheap": ProcessingMode.CHEAP,
            "balanced": ProcessingMode.BALANCED,
            "deep": ProcessingMode.DEEP,
        }

        processing_mode = mode_mapping.get(
            request.mode.lower(), ProcessingMode.BALANCED
        )

        # Create UserQuery
        user_query = UserQuery(
            query_text=request.query,
            query_id=str(uuid.uuid4()),
            processing_mode=processing_mode,
            filters=request.filters,
        )

        logger.info(
            "query_received",
            query_id=user_query.query_id,
            query=request.query[:100],
            mode=processing_mode.value,
        )

        # Route query
        response = await query_router.route_query(user_query)

        # Convert to dict
        response_dict = response.model_dump(mode="json")

        # Add cost summary
        response_dict["cost_breakdown"] = {
            "query_cost_usd": response.total_cost_usd,
            "estimated_cost_for_1m_similar_queries": response.total_cost_usd
            * 1_000_000,
        }

        logger.info(
            "query_completed",
            query_id=user_query.query_id,
            cost=response.total_cost_usd,
            duration_ms=response.processing_time_ms,
        )

        metrics_collector.increment("api_queries")

        return response_dict

    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/summary", response_model=Dict[str, Any])
async def get_analytics_summary(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
):
    """
    Get analytics summary.

    Returns precomputed metrics and aggregations.
    """
    try:
        # Parse dates if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        summary = analytics_service.get_summary(start_date=start_dt, end_date=end_dt)

        return summary.model_dump(mode="json")

    except Exception as e:
        logger.error("analytics_summary_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/trends/{metric}")
async def get_trend(
    metric: str,
    period: str = Query("daily", description="Period: daily, weekly, monthly"),
    num_periods: int = Query(30, description="Number of periods"),
):
    """
    Get trend data for a specific metric.

    Supported metrics:
    - call_volume
    - sentiment
    """
    try:
        if metric == "call_volume":
            trend = analytics_service.get_call_volume_trend(
                period=period, num_periods=num_periods
            )
        elif metric == "sentiment":
            trend = analytics_service.get_sentiment_trend(
                period=period, num_periods=num_periods
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")

        return trend.model_dump(mode="json")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("trend_failed", metric=metric, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calls/{call_id}")
async def get_call_details(call_id: str):
    """
    Get details for a specific call.

    Includes:
    - Metadata
    - Preprocessing results
    - Enrichment results
    - Segments
    """
    try:
        import duckdb

        db_path = config.storage_paths["structured_db"]
        conn = duckdb.connect(str(db_path))

        # Get call metadata
        call = conn.execute(
            "SELECT * FROM calls WHERE call_id = ?", [call_id]
        ).fetchone()

        if not call:
            raise HTTPException(status_code=404, detail="Call not found")

        # Get preprocessing
        preprocessing = conn.execute(
            "SELECT * FROM preprocessed_calls WHERE call_id = ?", [call_id]
        ).fetchone()

        # Get enrichment
        enrichment = conn.execute(
            "SELECT * FROM enrichments WHERE call_id = ?", [call_id]
        ).fetchone()

        # Get segments
        segments = conn.execute(
            "SELECT * FROM segments WHERE call_id = ? ORDER BY segment_id", [call_id]
        ).fetchall()

        conn.close()

        return {
            "call_id": call_id,
            "metadata": (
                dict(
                    zip(
                        [
                            "call_id",
                            "client_id",
                            "client_segment",
                            "product",
                            "call_duration_seconds",
                            "call_timestamp",
                            "region",
                            "agent_id",
                            "call_outcome",
                            "ingestion_timestamp",
                            "processed",
                            "preprocessed",
                            "enriched",
                        ],
                        call,
                    )
                )
                if call
                else None
            ),
            "preprocessing": (
                dict(
                    zip(
                        [
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
                        ],
                        preprocessing,
                    )
                )
                if preprocessing
                else None
            ),
            "enrichment": (
                dict(
                    zip(
                        [
                            "enrichment_id",
                            "call_id",
                            "segment_id",
                            "summary",
                            "key_issues",
                            "action_items",
                            "semantic_tags",
                            "model_used",
                            "tokens_used",
                            "cost_usd",
                            "enrichment_timestamp",
                            "confidence_score",
                        ],
                        enrichment,
                    )
                )
                if enrichment
                else None
            ),
            "num_segments": len(segments),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_call_failed", call_id=call_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get system metrics.

    Returns:
    - Query counts
    - Cost metrics
    - Performance metrics
    """
    try:
        # Get metrics
        metrics = metrics_collector.get_metrics()

        # Get cost summary
        cost_summary = cost_tracker.get_summary()

        return {
            "metrics": metrics,
            "costs": cost_summary,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("get_metrics_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get overall system statistics"""
    try:
        import duckdb

        db_path = config.storage_paths["structured_db"]
        conn = duckdb.connect(str(db_path))

        stats = {}

        # Total calls
        stats["total_calls"] = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

        # Processing status
        stats["processed"] = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE processed = TRUE"
        ).fetchone()[0]

        stats["preprocessed"] = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE preprocessed = TRUE"
        ).fetchone()[0]

        stats["enriched"] = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE enriched = TRUE"
        ).fetchone()[0]

        # Total cost
        total_enrichment_cost = (
            conn.execute("SELECT SUM(cost_usd) FROM enrichments").fetchone()[0] or 0.0
        )

        stats["total_enrichment_cost_usd"] = float(total_enrichment_cost)

        # Average cost per call
        if stats["enriched"] > 0:
            stats["avg_cost_per_enriched_call"] = (
                total_enrichment_cost / stats["enriched"]
            )
        else:
            stats["avg_cost_per_enriched_call"] = 0.0

        conn.close()

        return stats

    except Exception as e:
        logger.error("get_statistics_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)

    uvicorn.run("api.main:app", host=host, port=port, reload=True)
