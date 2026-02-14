"""
Streamlit UI - Interactive Dashboard
=====================================

User interface with:
1. Chat interface for natural language queries
2. Analytics dashboard with metrics and trends
3. Cost tracking and system stats
4. Query mode selection (cheap/balanced/deep)

Run with: streamlit run ui/streamlit_app.py
"""

import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

# Page config
st.set_page_config(
    page_title="Enterprise Call Intelligence Platform",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import services
try:
    from models.schemas import UserQuery, ProcessingMode
    from services.query_router import QueryRouter
    from services.analytics_service import AnalyticsService
    from utils.config_loader import config
    from utils.logger import cost_tracker, metrics_collector
except ImportError:
    st.error(
        "Failed to import services. Make sure you're running from the project root."
    )
    st.stop()


# Initialize services (with caching)
@st.cache_resource
def get_services():
    """Initialize and cache services"""
    return {"query_router": QueryRouter(), "analytics": AnalyticsService()}


def main():
    """Main app"""

    # Title
    st.title("📞 Enterprise Call Intelligence Platform")
    st.markdown("*AI-powered call transcript analysis with cost-optimized processing*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Query mode selection
        mode = st.selectbox(
            "Processing Mode",
            options=["Cheap", "Balanced", "Deep"],
            index=1,
            help="""
            - **Cheap**: Analytics only, no LLM ($0)
            - **Balanced**: Selective LLM usage ($0.01-0.05)
            - **Deep**: Full RAG with GPT-4 ($0.10-0.50)
            """,
        )

        # Navigation
        st.markdown("---")
        page = st.radio(
            "Navigation",
            options=["💬 Chat", "📊 Analytics", "📈 Metrics", "ℹ️ About"],
            index=0,
        )

        st.markdown("---")

        # System stats
        st.subheader("📊 System Stats")
        services = get_services()

        try:
            import duckdb

            db_path = config.storage_paths["structured_db"]
            conn = duckdb.connect(str(db_path))

            total_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            enriched = conn.execute(
                "SELECT COUNT(*) FROM calls WHERE enriched = TRUE"
            ).fetchone()[0]
            total_cost = (
                conn.execute("SELECT SUM(cost_usd) FROM enrichments").fetchone()[0]
                or 0.0
            )

            conn.close()

            st.metric("Total Calls", f"{total_calls:,}")
            st.metric("Enriched", f"{enriched:,}")
            st.metric("Total Cost", f"${total_cost:.4f}")

            if enriched > 0:
                avg_cost = total_cost / enriched
                st.metric("Avg Cost/Call", f"${avg_cost:.4f}")

                # Extrapolate to 1M calls
                extrapolated = avg_cost * 1_000_000
                st.info(f"**Estimated cost for 1M calls:** ${extrapolated:,.2f}")

        except Exception as e:
            st.warning(f"Could not load stats: {e}")

    # Main content
    if page == "💬 Chat":
        chat_page(mode)
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "📈 Metrics":
        metrics_page()
    else:
        about_page()


def chat_page(mode: str):
    """Chat interface"""
    st.header("💬 Ask Questions About Call Transcripts")

    # Example queries
    with st.expander("📝 Example Queries"):
        st.markdown(
            """
        **Analytics Queries (Fast, Free):**
        - How many calls about credit cards?
        - What are the top complaints?
        - Show me call trends for the last month
        
        **Search Queries (Fast, Cheap):**
        - Find calls mentioning refunds
        - Show me calls about account access issues
        
        **Insight Queries (Slower, More Expensive):**
        - What are customers saying about retirement fund performance?
        - Summarize common problems from premium customers
        - What concerns do customers have about fees?
        """
        )

    # Query input
    query_text = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are customers complaining about?",
        key="query_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if mode == "Deep":
            st.warning("⚠️ Deep mode uses expensive models (~$0.10-0.50/query)")
        elif mode == "Balanced":
            st.info("ℹ️ Balanced mode uses selective LLM (~$0.01-0.05/query)")
        else:
            st.success("✅ Cheap mode uses no LLM ($0/query)")

    # Process query
    if submit and query_text:
        with st.spinner("Processing query..."):
            try:
                # Create query
                mode_mapping = {
                    "Cheap": ProcessingMode.CHEAP,
                    "Balanced": ProcessingMode.BALANCED,
                    "Deep": ProcessingMode.DEEP,
                }

                user_query = UserQuery(
                    query_text=query_text,
                    query_id=f"ui_{datetime.now().timestamp()}",
                    processing_mode=mode_mapping[mode],
                )

                # Route query
                services = get_services()

                # Run async query
                response = asyncio.run(services["query_router"].route_query(user_query))

                # Display response
                st.markdown("---")
                st.subheader("📝 Answer")
                st.write(response.answer)

                # Metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Query Type", response.query_type.value.title())
                with col2:
                    st.metric("Cost", f"${response.total_cost_usd:.4f}")
                with col3:
                    st.metric("Time", f"{response.processing_time_ms:.0f}ms")
                with col4:
                    st.metric("Confidence", f"{response.confidence:.1%}")

                # Sources
                if response.sources and len(response.sources) > 0:
                    with st.expander(f"📚 Sources ({len(response.sources)})"):
                        for i, source in enumerate(response.sources[:5]):
                            st.markdown(f"**[{i+1}]** Score: {source.score:.3f}")
                            st.text(source.text[:300] + "...")
                            st.markdown(
                                f"*Product: {source.metadata.get('product', 'N/A')} | "
                                f"Intent: {source.metadata.get('intent', 'N/A')}*"
                            )
                            st.markdown("---")

                # Analytics data
                if response.analytics:
                    with st.expander("📊 Analytics Data"):
                        st.json(response.analytics)

            except Exception as e:
                st.error(f"Error processing query: {e}")
                import traceback

                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


def analytics_page():
    """Analytics dashboard"""
    st.header("📊 Analytics Dashboard")

    services = get_services()

    try:
        # Get summary
        summary = services["analytics"].get_summary()

        # KPIs
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Calls", f"{summary.total_calls:,}")

        with col2:
            if summary.sentiment_distribution:
                negative_count = sum(
                    count
                    for sentiment, count in summary.sentiment_distribution.items()
                    if "negative" in sentiment.lower()
                )
                negative_pct = (
                    (negative_count / summary.total_calls * 100)
                    if summary.total_calls > 0
                    else 0
                )
                st.metric("Negative Sentiment", f"{negative_pct:.1f}%")

        with col3:
            if summary.top_complaints:
                total_complaints = sum(c.count for c in summary.top_complaints)
                st.metric("Total Complaints", f"{total_complaints:,}")

        with col4:
            days = (summary.date_range["end"] - summary.date_range["start"]).days
            st.metric("Date Range", f"{days} days")

        # Top Complaints
        st.subheader("🔥 Top Complaint Categories")
        if summary.top_complaints:
            df = pd.DataFrame([c.model_dump() for c in summary.top_complaints])

            fig = px.bar(
                df,
                x="count",
                y="category",
                orientation="h",
                title="Complaint Volume by Category",
                labels={"count": "Number of Calls", "category": "Category"},
                color="count",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Top Products
        st.subheader("💳 Call Volume by Product")
        if summary.top_products:
            df = pd.DataFrame([c.model_dump() for c in summary.top_products])

            fig = px.pie(
                df,
                values="count",
                names="category",
                title="Call Distribution by Product",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Sentiment Distribution
        st.subheader("😊 Sentiment Distribution")
        if summary.sentiment_distribution:
            df = pd.DataFrame(
                [
                    {"sentiment": k, "count": v}
                    for k, v in summary.sentiment_distribution.items()
                ]
            )

            fig = px.bar(
                df,
                x="sentiment",
                y="count",
                title="Call Count by Sentiment",
                color="sentiment",
                color_discrete_map={
                    "very_positive": "green",
                    "positive": "lightgreen",
                    "neutral": "gray",
                    "negative": "orange",
                    "very_negative": "red",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        # Trends
        st.subheader("📈 Call Volume Trend")
        trend = services["analytics"].get_call_volume_trend(
            period="daily", num_periods=30
        )

        if trend.data_points:
            df = pd.DataFrame(trend.data_points)
            df["period"] = pd.to_datetime(df["period"])

            fig = px.line(
                df,
                x="period",
                y="value",
                title="Daily Call Volume (Last 30 Days)",
                labels={"period": "Date", "value": "Number of Calls"},
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        import traceback

        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def metrics_page():
    """System metrics and cost tracking"""
    st.header("📈 System Metrics & Cost Tracking")

    try:
        import duckdb

        db_path = config.storage_paths["structured_db"]
        conn = duckdb.connect(str(db_path))

        # Processing pipeline status
        st.subheader("🔄 Processing Pipeline Status")

        total = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        preprocessed = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE preprocessed = TRUE"
        ).fetchone()[0]
        enriched = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE enriched = TRUE"
        ).fetchone()[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ingested", f"{total:,}", help="Total calls loaded")
        with col2:
            pct = (preprocessed / total * 100) if total > 0 else 0
            st.metric("Preprocessed", f"{preprocessed:,}", f"{pct:.1f}%")
        with col3:
            pct = (enriched / total * 100) if total > 0 else 0
            st.metric("Enriched", f"{enriched:,}", f"{pct:.1f}%")

        # Progress bar
        if total > 0:
            progress = enriched / total
            st.progress(progress, text=f"Pipeline Progress: {progress:.1%}")

        # Cost breakdown
        st.subheader("💰 Cost Breakdown")

        cost_by_model = conn.execute(
            """
            SELECT 
                model_used,
                COUNT(*) as num_calls,
                SUM(tokens_used) as total_tokens,
                SUM(cost_usd) as total_cost
            FROM enrichments
            GROUP BY model_used
            ORDER BY total_cost DESC
        """
        ).fetchdf()

        if not cost_by_model.empty:
            st.dataframe(cost_by_model, use_container_width=True)

            # Cost visualization
            fig = px.bar(
                cost_by_model,
                x="model_used",
                y="total_cost",
                title="Cost by Model",
                labels={"model_used": "Model", "total_cost": "Total Cost (USD)"},
                color="total_cost",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Importance distribution
        st.subheader("🎯 Importance Score Distribution")

        importance_dist = conn.execute(
            """
            SELECT 
                overall_importance,
                COUNT(*) as count
            FROM preprocessed_calls
            GROUP BY overall_importance
            ORDER BY 
                CASE overall_importance
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END
        """
        ).fetchdf()

        if not importance_dist.empty:
            fig = px.pie(
                importance_dist,
                values="count",
                names="overall_importance",
                title="Call Distribution by Importance Level",
                color="overall_importance",
                color_discrete_map={
                    "critical": "red",
                    "high": "orange",
                    "medium": "yellow",
                    "low": "green",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show triage effectiveness
            st.info(
                f"""
            **Triage Effectiveness:**
            - Only **{importance_dist[importance_dist['overall_importance'].isin(['critical', 'high'])]['count'].sum():,}** calls 
            flagged for expensive LLM enrichment
            - That's **{importance_dist[importance_dist['overall_importance'].isin(['critical', 'high'])]['count'].sum() / importance_dist['count'].sum() * 100:.1f}%** of total calls
            - Potential cost savings: **85%+** compared to enriching all calls
            """
            )

        conn.close()

    except Exception as e:
        st.error(f"Error loading metrics: {e}")


def about_page():
    """About page with system information"""
    st.header("ℹ️ About the Platform")

    st.markdown(
        """
    ## Enterprise Call Intelligence Platform
    
    A production-grade AI system for analyzing millions of customer support call transcripts.
    
    ### 🏗️ Architecture Highlights
    
    #### Service-Oriented Design
    - **Ingestion Service**: Batch loading of transcripts
    - **Preprocessing Service**: Cheap NLP triage & importance scoring
    - **Enrichment Service**: Selective LLM usage for important calls
    - **Indexing Service**: Multi-modal search (vector + keyword)
    - **Analytics Service**: Fast aggregations without LLM
    - **RAG Service**: Deep semantic insights
    - **Query Router**: Intelligent query classification & routing
    
    #### Cost Optimization Strategy
    
    1. **Triage First**: Use cheap rule-based NLP to score importance (cost: $0)
    2. **Selective Enrichment**: Only 15% of calls get LLM processing
    3. **Tiered Processing**: 
       - Analytics queries: No LLM ($0)
       - Search queries: Embeddings only ($0.001)
       - Insight queries: Full RAG ($0.01-0.10)
    4. **Model Selection**: Use GPT-3.5 by default, GPT-4 only for critical cases
    
    **Result**: ~85% cost reduction compared to naive "LLM everything" approach
    
    ### 🎯 Key Features
    
    #### Importance Scoring
    - Identifies which calls and segments need attention
    - Combines multiple signals: keywords, intent, sentiment
    - Enables prioritization at scale
    
    #### Hybrid Search
    - Keyword search for exact matches
    - Semantic search for similar concepts
    - Combined ranking for best results
    
    #### Multi-Modal Queries
    - Fast analytics: "How many calls?" → SQL
    - Document search: "Find calls about X" → Vector search
    - Deep insights: "What are customers saying?" → RAG + LLM
    
    ### 📊 Scalability
    
    **Current Setup (Demo):**
    - Local embedding model
    - DuckDB for structured data
    - FAISS for vectors
    - Single machine
    
    **Production Path:**
    - Replace DuckDB → Snowflake/BigQuery
    - Replace FAISS → Pinecone/Weaviate/OpenSearch
    - Add Kafka/SQS for streaming
    - Deploy on K8s/ECS
    - Add Redis for caching
    - Implement proper observability (DataDog, Prometheus)
    
    ### 💡 Design Principles
    
    1. **Not all queries need LLM**: Route intelligently
    2. **Not all calls need enrichment**: Triage first
    3. **Cheap operations first**: Filter before expensive operations
    4. **Observability matters**: Track every cost
    5. **Evidence-based**: Always show sources
    
    ### 🚀 Next Steps for Production
    
    - Add authentication & authorization
    - Implement rate limiting
    - Add caching layer (Redis)
    - Set up monitoring & alerting
    - Add A/B testing framework
    - Implement feedback loop
    - Add PII detection & redaction
    - Set up CI/CD pipeline
    """
    )


if __name__ == "__main__":
    main()
