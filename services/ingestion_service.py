"""
Ingestion Service
=================

Responsible for:
- Loading raw call transcripts from storage
- Validating data format
- Storing in structured database
- Triggering downstream processing

System Design Notes:
-------------------
- This service runs in BATCH mode (offline processing)
- In production, this would consume from Kafka/SQS
- Handles millions of calls efficiently through batching
- Separates raw storage (S3) from structured storage (DB)
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import duckdb

from models.schemas import CallMetadata, ClientSegment, ProductCategory
from utils.logger import get_logger, track_performance, metrics_collector
from utils.config_loader import config

logger = get_logger(__name__)


class IngestionService:
    """
    Ingestion service for call transcripts.

    Architecture Pattern: ETL (Extract, Transform, Load)
    - Extract: Read from raw storage
    - Transform: Validate and normalize
    - Load: Store in structured DB

    Scalability Considerations:
    - Batch processing for efficiency
    - Parallel processing support
    - Idempotent (can re-run safely)
    - Checkpointing for large datasets
    """

    def __init__(self):
        self.raw_transcripts_path = config.storage_paths["raw_transcripts"]
        self.db_path = config.storage_paths["structured_db"]
        self.batch_size = config.get("services.ingestion.batch_size", 100)
        self.enable_parallel = config.get("services.ingestion.enable_parallel", True)

        # Ensure directories exist
        self.raw_transcripts_path.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """
        Initialize DuckDB database with schema.

        Why DuckDB?
        - Column-oriented (fast analytics)
        - Embedded (no server needed)
        - Excellent for OLAP queries
        - Easy migration to MotherDuck/Snowflake in production
        """
        conn = duckdb.connect(str(self.db_path))

        # Calls metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id VARCHAR PRIMARY KEY,
                client_id VARCHAR,
                client_segment VARCHAR,
                product VARCHAR,
                call_duration_seconds DOUBLE,
                call_timestamp TIMESTAMP,
                region VARCHAR,
                agent_id VARCHAR,
                call_outcome VARCHAR,
                ingestion_timestamp TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                preprocessed BOOLEAN DEFAULT FALSE,
                enriched BOOLEAN DEFAULT FALSE
            )
        """
        )

        # Create indexes for common queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_timestamp ON calls(call_timestamp)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_product ON calls(product)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_client_segment ON calls(client_segment)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_processed ON calls(processed)")

        conn.close()

        logger.info("database_initialized", db_path=str(self.db_path))

    async def ingest_transcript(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Ingest a single transcript file.

        Args:
            file_path: Path to transcript JSON file

        Returns:
            Ingested call data or None if failed
        """
        try:
            # Read transcript file
            with open(file_path, "r") as f:
                data = json.load(f)

            # Validate
            if not self._validate_transcript(data):
                logger.warning("invalid_transcript", file=str(file_path))
                return None

            # Extract metadata
            call_id = data["call_id"]
            metadata = data["metadata"]

            # Store in database
            self._store_metadata(call_id, metadata)

            logger.info("transcript_ingested", call_id=call_id)
            metrics_collector.increment("transcripts_ingested")

            return {
                "call_id": call_id,
                "metadata": metadata,
                "transcript": data["transcript"],
                "file_path": str(file_path),
            }

        except Exception as e:
            logger.error("ingestion_failed", file=str(file_path), error=str(e))
            metrics_collector.increment("ingestion_errors")
            return None

    def _validate_transcript(self, data: Dict[str, Any]) -> bool:
        """Validate transcript data structure"""
        required_fields = ["call_id", "transcript", "metadata"]

        if not all(field in data for field in required_fields):
            return False

        if not isinstance(data["transcript"], list):
            return False

        if len(data["transcript"]) == 0:
            return False

        return True

    def _store_metadata(self, call_id: str, metadata: Dict[str, Any]):
        """Store call metadata in database"""
        conn = duckdb.connect(str(self.db_path))

        try:
            # Handle datetime conversion
            call_timestamp = metadata.get("call_timestamp")
            if isinstance(call_timestamp, str):
                call_timestamp = datetime.fromisoformat(call_timestamp)

            conn.execute(
                """
                INSERT OR REPLACE INTO calls (
                    call_id, client_id, client_segment, product,
                    call_duration_seconds, call_timestamp, region,
                    agent_id, call_outcome, ingestion_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    call_id,
                    metadata.get("client_id"),
                    metadata.get("client_segment"),
                    metadata.get("product"),
                    metadata.get("call_duration_seconds"),
                    call_timestamp,
                    metadata.get("region"),
                    metadata.get("agent_id"),
                    metadata.get("call_outcome"),
                    datetime.now(),
                ],
            )

        finally:
            conn.close()

    async def ingest_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Ingest a batch of transcripts.

        System Design Note:
        ------------------
        Batch processing is CRITICAL for handling millions of calls:
        - Reduces I/O overhead
        - Enables parallel processing
        - Better resource utilization
        """
        results = []

        if self.enable_parallel:
            # Process in parallel
            tasks = [self.ingest_transcript(fp) for fp in file_paths]
            results = await asyncio.gather(*tasks)
        else:
            # Process sequentially
            for fp in file_paths:
                result = await self.ingest_transcript(fp)
                results.append(result)

        # Filter out None values
        results = [r for r in results if r is not None]

        return results

    async def ingest_all(self, limit: Optional[int] = None) -> int:
        """
        Ingest all transcripts from raw storage.

        Args:
            limit: Optional limit on number of files to process

        Returns:
            Number of transcripts ingested
        """
        with track_performance("ingestion_all"):
            # Find all transcript files
            transcript_files = list(self.raw_transcripts_path.glob("*.json"))

            if limit:
                transcript_files = transcript_files[:limit]

            total_files = len(transcript_files)
            logger.info("starting_ingestion", total_files=total_files)

            # Process in batches
            total_ingested = 0

            for i in range(0, total_files, self.batch_size):
                batch = transcript_files[i : i + self.batch_size]

                results = await self.ingest_batch(batch)
                total_ingested += len(results)

                logger.info(
                    "batch_ingested",
                    batch_num=i // self.batch_size + 1,
                    batch_size=len(results),
                    total_ingested=total_ingested,
                )

            logger.info("ingestion_complete", total_ingested=total_ingested)
            return total_ingested

    def get_unprocessed_calls(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get calls that haven't been processed yet.

        This enables incremental processing - important for continuous ingestion.
        """
        conn = duckdb.connect(str(self.db_path))

        query = "SELECT * FROM calls WHERE processed = FALSE"
        if limit:
            query += f" LIMIT {limit}"

        result = conn.execute(query).fetchdf()
        conn.close()

        return result.to_dict("records")

    def mark_processed(self, call_id: str, stage: str = "processed"):
        """Mark a call as processed at a specific stage"""
        conn = duckdb.connect(str(self.db_path))

        conn.execute(f"UPDATE calls SET {stage} = TRUE WHERE call_id = ?", [call_id])

        conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        conn = duckdb.connect(str(self.db_path))

        stats = {}

        # Total calls
        stats["total_calls"] = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

        # By segment
        stats["by_segment"] = (
            conn.execute(
                """
            SELECT client_segment, COUNT(*) as count
            FROM calls
            GROUP BY client_segment
        """
            )
            .fetchdf()
            .to_dict("records")
        )

        # By product
        stats["by_product"] = (
            conn.execute(
                """
            SELECT product, COUNT(*) as count
            FROM calls
            GROUP BY product
            ORDER BY count DESC
        """
            )
            .fetchdf()
            .to_dict("records")
        )

        # Processing status
        stats["processing_status"] = {
            "processed": conn.execute(
                "SELECT COUNT(*) FROM calls WHERE processed = TRUE"
            ).fetchone()[0],
            "unprocessed": conn.execute(
                "SELECT COUNT(*) FROM calls WHERE processed = FALSE"
            ).fetchone()[0],
            "preprocessed": conn.execute(
                "SELECT COUNT(*) FROM calls WHERE preprocessed = TRUE"
            ).fetchone()[0],
            "enriched": conn.execute(
                "SELECT COUNT(*) FROM calls WHERE enriched = TRUE"
            ).fetchone()[0],
        }

        conn.close()

        return stats


async def main():
    """Run ingestion service"""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest call transcripts")
    parser.add_argument("--limit", type=int, help="Limit number of files to ingest")
    args = parser.parse_args()

    service = IngestionService()

    num_ingested = await service.ingest_all(limit=args.limit)

    print(f"✓ Ingested {num_ingested} transcripts")

    # Show statistics
    stats = service.get_statistics()
    print("\nIngestion Statistics:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Processed: {stats['processing_status']['processed']}")
    print(f"  Unprocessed: {stats['processing_status']['unprocessed']}")


if __name__ == "__main__":
    asyncio.run(main())
