"""
Pipeline Orchestrator
=====================

End-to-end pipeline for processing call transcripts.

Stages:
1. Generate dummy data (optional)
2. Ingest transcripts
3. Preprocess & score importance
4. Selective enrichment (LLM)
5. Index for search
6. Ready for queries

Usage:
    python -m pipelines.orchestrator --num-calls 1000
    OR
    PYTHONPATH=. python pipelines/orchestrator.py --num-calls 1000
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import argparse
from datetime import datetime

from pipelines.data_generator import CallTranscriptGenerator
from services.ingestion_service import IngestionService
from services.preprocessing_service import PreprocessingService
from services.enrichment_service import EnrichmentService
from services.indexing_service import IndexingService
from utils.logger import get_logger, cost_tracker
from utils.config_loader import config

logger = get_logger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the entire processing pipeline.

    System Design Note:
    ------------------
    This represents an OFFLINE batch processing pipeline.

    In production:
    - Stage 1-2: Triggered by new data arrival (S3, Kafka)
    - Stage 3-5: Run continuously or on schedule
    - Each stage is idempotent and can be rerun
    """

    def __init__(self):
        self.ingestion = IngestionService()
        self.preprocessing = PreprocessingService()
        self.enrichment = EnrichmentService()
        self.indexing = IndexingService()

        logger.info("pipeline_orchestrator_initialized")

    async def run_full_pipeline(
        self,
        generate_data: bool = True,
        num_calls: int = 1000,
        skip_enrichment: bool = False,
    ):
        """
        Run the complete pipeline.

        Args:
            generate_data: Whether to generate dummy data
            num_calls: Number of calls to generate
            skip_enrichment: Skip expensive LLM enrichment (for testing)
        """
        start_time = datetime.now()

        logger.info("pipeline_started", num_calls=num_calls)

        print("\n" + "=" * 60)
        print("ENTERPRISE CALL INTELLIGENCE PLATFORM")
        print("Pipeline Execution")
        print("=" * 60 + "\n")

        # Stage 1: Generate Data (optional)
        if generate_data:
            print("Stage 1/5: Generating Dummy Data")
            print("-" * 60)

            output_dir = config.storage_paths["raw_transcripts"]
            generator = CallTranscriptGenerator(output_dir=output_dir)

            calls = generator.generate_dataset(num_calls=num_calls)
            print(f"✓ Generated {len(calls)} call transcripts\n")

        # Stage 2: Ingestion
        print("Stage 2/5: Ingesting Transcripts")
        print("-" * 60)

        num_ingested = await self.ingestion.ingest_all()
        print(f"✓ Ingested {num_ingested} transcripts\n")

        # Stage 3: Preprocessing
        print("Stage 3/5: Preprocessing & Importance Scoring")
        print("-" * 60)

        unprocessed = self.ingestion.get_unprocessed_calls()
        print(f"Processing {len(unprocessed)} calls...")

        preprocessing_results = []
        for call in unprocessed:
            result = await self.preprocessing.preprocess_call(call["call_id"])
            preprocessing_results.append(result)
            self.ingestion.mark_processed(call["call_id"], "preprocessed")

        # Show importance distribution
        importance_dist = {}
        for result in preprocessing_results:
            level = result.overall_importance.value
            importance_dist[level] = importance_dist.get(level, 0) + 1

        print(f"✓ Preprocessed {len(preprocessing_results)} calls")
        print("\nImportance Distribution:")
        for level, count in sorted(importance_dist.items()):
            pct = count / len(preprocessing_results) * 100
            print(f"  {level.upper()}: {count} ({pct:.1f}%)")

        requiring_enrichment = sum(
            1 for r in preprocessing_results if r.requires_enrichment
        )
        print(
            f"\nCalls requiring enrichment: {requiring_enrichment} "
            f"({requiring_enrichment / len(preprocessing_results) * 100:.1f}%)"
        )
        print()

        # Stage 4: Selective Enrichment
        if not skip_enrichment:
            print("Stage 4/5: Selective LLM Enrichment")
            print("-" * 60)

            calls_to_enrich = self.preprocessing.get_calls_requiring_enrichment()
            print(f"Enriching {len(calls_to_enrich)} calls...")

            enrichment_results = []
            for i, call_id in enumerate(calls_to_enrich):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(calls_to_enrich)}")

                result = await self.enrichment.enrich_call(call_id)
                enrichment_results.append(result)
                self.ingestion.mark_processed(call_id, "enriched")

            # Show enrichment stats
            total_cost = sum(r.cost_usd for r in enrichment_results)
            total_tokens = sum(r.tokens_used for r in enrichment_results)
            avg_cost = total_cost / len(enrichment_results) if enrichment_results else 0

            print(f"\n✓ Enriched {len(enrichment_results)} calls")
            print(f"  Total cost: ${total_cost:.4f}")
            print(f"  Avg cost per call: ${avg_cost:.4f}")
            print(f"  Total tokens: {total_tokens:,}")

            # Extrapolate to 1M calls
            cost_1m = avg_cost * 1_000_000
            cost_1m_all = avg_cost * num_calls  # If we enriched ALL calls
            savings = cost_1m_all - total_cost

            print(f"\n📊 Cost Analysis:")
            print(f"  Cost for 1M calls (with triage): ${cost_1m:,.2f}")
            print(f"  Cost if enriching ALL calls: ${cost_1m_all:,.2f}")
            print(
                f"  Savings from triage: ${savings:,.2f} ({savings/cost_1m_all*100:.1f}%)"
            )
            print()
        else:
            print("Stage 4/5: Skipping enrichment (--skip-enrichment)\n")

        # Stage 5: Indexing
        print("Stage 5/5: Building Search Indices")
        print("-" * 60)

        await self.indexing.index_all_calls()

        print(f"✓ Built vector and keyword indices\n")

        # Pipeline complete
        duration = (datetime.now() - start_time).total_seconds()

        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total duration: {duration:.1f}s")
        print(f"\nSystem is ready for queries!")
        print(f"  - Start API: python -m api.main")
        print(f"  - Start UI: streamlit run ui/streamlit_app.py")
        print("=" * 60 + "\n")

        logger.info("pipeline_completed", duration_seconds=duration)


async def main():
    """Run pipeline"""
    parser = argparse.ArgumentParser(description="Run processing pipeline")
    parser.add_argument(
        "--num-calls", type=int, default=1000, help="Number of calls to generate"
    )
    parser.add_argument(
        "--no-generate", action="store_true", help="Skip data generation"
    )
    parser.add_argument(
        "--skip-enrichment", action="store_true", help="Skip LLM enrichment"
    )

    args = parser.parse_args()

    # Ensure directories exist
    config.ensure_directories()

    # Run pipeline
    orchestrator = PipelineOrchestrator()
    await orchestrator.run_full_pipeline(
        generate_data=not args.no_generate,
        num_calls=args.num_calls,
        skip_enrichment=args.skip_enrichment,
    )


if __name__ == "__main__":
    asyncio.run(main())
