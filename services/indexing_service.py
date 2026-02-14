"""
Indexing Service - Multi-Modal Search Infrastructure
====================================================

Creates multiple indices for different query types:
1. Vector embeddings for semantic search
2. Keyword index for fast filtering
3. Structured data for analytics

System Design: Hybrid Search Architecture
-----------------------------------------
- Vector search: "What are customers saying about X?"
- Keyword search: "Find calls mentioning 'refund'"
- Structured queries: "Count calls by product"

Why Multiple Indices?
--------------------
- No single index is optimal for all query types
- Vector: Best for semantic similarity
- Keyword: Best for exact term matching
- Structured: Best for aggregations

Scalability:
-----------
- For millions of calls, use:
  - FAISS for vectors (can scale to billions)
  - Elasticsearch for keywords (distributed)
  - DuckDB/Snowflake for structured data
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import duckdb

from models.schemas import SearchResult
from utils.logger import get_logger, track_performance, metrics_collector, cost_tracker
from utils.config_loader import config

logger = get_logger(__name__)


class VectorStore:
    """
    Vector store using FAISS.

    Why FAISS?
    ----------
    - Extremely fast similarity search
    - Handles billions of vectors
    - Memory efficient
    - Supports GPU acceleration
    - Industry standard (used by Meta, OpenAI, etc.)

    For production:
    - Could use Pinecone, Weaviate, or Qdrant for managed solution
    - Could use OpenSearch with k-NN plugin
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.id_to_metadata = {}
        self.metadata_to_id = {}
        self.next_id = 0

        logger.info("vector_store_initialized", dimension=dimension)

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add vectors to index.

        Args:
            vectors: numpy array of shape (n, dimension)
            metadata: List of metadata dicts for each vector
        """
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have same length")

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        # Add to index
        self.index.add(vectors)

        # Store metadata
        for i, meta in enumerate(metadata):
            idx = self.next_id + i
            self.id_to_metadata[idx] = meta

            # Create reverse mapping
            call_id = meta.get("call_id")
            segment_id = meta.get("segment_id")
            if call_id and segment_id:
                self.metadata_to_id[f"{call_id}_{segment_id}"] = idx

        self.next_id += len(vectors)

        logger.info("vectors_added", count=len(vectors), total=self.next_id)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Returns:
            List of results with metadata and scores
        """
        # Normalize query
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Search
        distances, indices = self.index.search(query_vector, top_k)

        # Convert distances to similarity scores (0-1)
        # L2 distance ranges from 0 (identical) to 2 (opposite)
        # Convert to similarity: 1 - (distance / 2)
        similarities = 1 - (distances[0] / 2)

        # Prepare results
        results = []
        for idx, score in zip(indices[0], similarities):
            if idx == -1:  # No result found
                continue

            metadata = self.id_to_metadata.get(idx, {})
            results.append({**metadata, "score": float(score)})

        return results

    def save(self, path: Path):
        """Save index to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "vectors.index"))

        # Save metadata
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "id_to_metadata": self.id_to_metadata,
                    "metadata_to_id": self.metadata_to_id,
                    "next_id": self.next_id,
                    "dimension": self.dimension,
                },
                f,
            )

        logger.info("vector_store_saved", path=str(path))

    def load(self, path: Path):
        """Load index from disk"""
        # Load FAISS index
        self.index = faiss.read_index(str(path / "vectors.index"))

        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.id_to_metadata = data["id_to_metadata"]
            self.metadata_to_id = data["metadata_to_id"]
            self.next_id = data["next_id"]
            self.dimension = data["dimension"]

        logger.info("vector_store_loaded", path=str(path), count=self.next_id)


class KeywordIndex:
    """
    Keyword-based search using TF-IDF.

    Why TF-IDF?
    -----------
    - Fast exact keyword matching
    - Works well for filtering
    - Low memory footprint
    - No API costs

    For production:
    - Replace with Elasticsearch for distributed search
    - Add fuzzy matching, synonyms, etc.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams and bigrams
        )
        self.tfidf_matrix = None
        self.documents = []
        self.metadata = []

        logger.info("keyword_index_initialized")

    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Build TF-IDF index"""
        if len(documents) != len(metadata):
            raise ValueError("Documents and metadata must have same length")

        self.documents = documents
        self.metadata = metadata
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        logger.info("keyword_index_built", num_documents=len(documents))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using keywords"""
        if self.tfidf_matrix is None:
            return []

        # Transform query
        query_vec = self.vectorizer.transform([query])

        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if there's a match
                results.append(
                    {**self.metadata[idx], "score": float(similarities[idx])}
                )

        return results

    def save(self, path: Path):
        """Save index to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "documents": self.documents,
                    "metadata": self.metadata,
                },
                f,
            )

        logger.info("keyword_index_saved", path=str(path))

    def load(self, path: Path):
        """Load index from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.documents = data["documents"]
            self.metadata = data["metadata"]

        logger.info("keyword_index_loaded", path=str(path), count=len(self.documents))


class IndexingService:
    """
    Indexing service that creates and manages all indices.

    Architecture:
    ------------
    1. Load enriched calls from database
    2. Generate embeddings (using local model - no API cost!)
    3. Build vector index
    4. Build keyword index
    5. Persist to disk

    Cost Optimization:
    -----------------
    - Use local embedding model (sentence-transformers)
    - No OpenAI API calls needed
    - One-time cost for embeddings
    - Reuse for all queries
    """

    def __init__(self):
        self.db_path = config.storage_paths["structured_db"]
        self.vector_store_path = config.storage_paths["vector_store"]
        self.keyword_index_path = config.storage_paths["keyword_index"]

        # Embedding model
        embedding_model_name = config.get(
            "services.indexing.embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        logger.info("loading_embedding_model", model=embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dimension = config.get(
            "services.indexing.embedding_dimension", 384
        )

        # Initialize stores
        self.vector_store = VectorStore(dimension=self.embedding_dimension)
        self.keyword_index = KeywordIndex()

        # Try to load existing indices
        self._try_load_indices()

        logger.info("indexing_service_initialized")

    def _try_load_indices(self):
        """Try to load existing indices"""
        try:
            if (self.vector_store_path / "vectors.index").exists():
                self.vector_store.load(self.vector_store_path)
                logger.info("loaded_existing_vector_index")
        except Exception as e:
            logger.warning("failed_to_load_vector_index", error=str(e))

        try:
            if self.keyword_index_path.exists():
                self.keyword_index.load(self.keyword_index_path)
                logger.info("loaded_existing_keyword_index")
        except Exception as e:
            logger.warning("failed_to_load_keyword_index", error=str(e))

    async def index_all_calls(self):
        """
        Index all calls in the database.

        Process:
        -------
        1. Load enriched calls from DB
        2. Generate embeddings for each segment
        3. Build vector index
        4. Build keyword index
        5. Save indices
        """
        with track_performance("index_all_calls"):
            # Load calls from database
            conn = duckdb.connect(str(self.db_path))

            # Get all segments with enrichment
            segments_df = conn.execute(
                """
                SELECT 
                    s.segment_id,
                    s.call_id,
                    s.speaker,
                    s.text,
                    s.intent,
                    s.sentiment,
                    s.importance_score,
                    e.summary,
                    e.key_issues,
                    e.semantic_tags,
                    c.product,
                    c.client_segment,
                    c.call_timestamp
                FROM segments s
                LEFT JOIN enrichments e ON s.call_id = e.call_id
                JOIN calls c ON s.call_id = c.call_id
            """
            ).fetchdf()

            conn.close()

            if len(segments_df) == 0:
                logger.warning("no_segments_to_index")
                return

            logger.info("indexing_segments", count=len(segments_df))

            # Prepare documents for indexing
            documents = []
            metadata_list = []

            for _, row in segments_df.iterrows():
                # Create rich document for embedding
                # Combine segment text with enrichment for better semantic representation
                doc_parts = [row["text"]]

                if row["summary"] and str(row["summary"]) != "None":
                    doc_parts.append(f"Summary: {row['summary']}")

                if row["semantic_tags"] and str(row["semantic_tags"]) != "None":
                    try:
                        tags = json.loads(row["semantic_tags"])
                        doc_parts.append(f"Tags: {', '.join(tags)}")
                    except:
                        pass

                document = " | ".join(doc_parts)
                documents.append(document)

                # Metadata
                metadata_list.append(
                    {
                        "segment_id": row["segment_id"],
                        "call_id": row["call_id"],
                        "speaker": row["speaker"],
                        "text": row["text"],
                        "intent": row["intent"],
                        "sentiment": row["sentiment"],
                        "importance_score": float(row["importance_score"]),
                        "product": row["product"],
                        "client_segment": row["client_segment"],
                        "call_timestamp": str(row["call_timestamp"]),
                    }
                )

            # Generate embeddings
            logger.info("generating_embeddings", count=len(documents))
            embeddings = self.embedding_model.encode(
                documents, show_progress_bar=True, convert_to_numpy=True
            )

            # Track embedding cost (minimal for local model)
            # Estimate tokens: ~100 tokens per document on average
            estimated_tokens = len(documents) * 100
            embedding_cost = (estimated_tokens / 1000) * 0.00002  # Simulated cost
            cost_tracker.track_embedding_call(
                model="sentence-transformers",
                num_texts=len(documents),
                total_tokens=estimated_tokens,
                cost_per_1k=0.00002,
            )

            # Build vector index
            logger.info("building_vector_index")
            self.vector_store.add_vectors(embeddings, metadata_list)

            # Build keyword index
            logger.info("building_keyword_index")
            self.keyword_index.build_index(documents, metadata_list)

            # Save indices
            logger.info("saving_indices")
            self.vector_store.save(self.vector_store_path)
            self.keyword_index.save(self.keyword_index_path)

            logger.info(
                "indexing_complete",
                total_segments=len(documents),
                embedding_cost=embedding_cost,
            )

            metrics_collector.increment("segments_indexed", len(documents))

    def semantic_search(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search using vector embeddings.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional metadata filters
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]

        # Search
        results = self.vector_store.search(
            query_embedding, top_k=top_k * 2
        )  # Get more for filtering

        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)

        # Convert to SearchResult objects
        search_results = []
        for r in results[:top_k]:
            search_results.append(
                SearchResult(
                    call_id=r["call_id"],
                    segment_id=r.get("segment_id"),
                    text=r["text"],
                    score=r["score"],
                    metadata=r,
                    semantic_score=r["score"],
                )
            )

        return search_results

    def keyword_search(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Keyword-based search using TF-IDF.

        Best for:
        - Exact term matching
        - Product names, technical terms
        - Filtering by specific keywords
        """
        # Search
        results = self.keyword_index.search(query, top_k=top_k * 2)

        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)

        # Convert to SearchResult objects
        search_results = []
        for r in results[:top_k]:
            search_results.append(
                SearchResult(
                    call_id=r["call_id"],
                    segment_id=r.get("segment_id"),
                    text=r["text"],
                    score=r["score"],
                    metadata=r,
                    keyword_score=r["score"],
                )
            )

        return search_results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining keyword and semantic search.

        This is BEST for most queries:
        - Keyword search catches exact matches
        - Semantic search catches similar concepts
        - Combined score gives best of both worlds
        """
        # Get results from both methods
        keyword_results = self.keyword_search(query, top_k=top_k * 2, filters=filters)
        semantic_results = self.semantic_search(query, top_k=top_k * 2, filters=filters)

        # Combine and rerank
        combined = {}

        for result in keyword_results:
            key = f"{result.call_id}_{result.segment_id}"
            combined[key] = {
                "result": result,
                "keyword_score": result.score,
                "semantic_score": 0.0,
            }

        for result in semantic_results:
            key = f"{result.call_id}_{result.segment_id}"
            if key in combined:
                combined[key]["semantic_score"] = result.score
            else:
                combined[key] = {
                    "result": result,
                    "keyword_score": 0.0,
                    "semantic_score": result.score,
                }

        # Calculate combined scores
        for key, data in combined.items():
            combined_score = (
                data["keyword_score"] * keyword_weight
                + data["semantic_score"] * semantic_weight
            )

            # Update result
            result = data["result"]
            result.keyword_score = data["keyword_score"]
            result.semantic_score = data["semantic_score"]
            result.combined_score = combined_score
            result.score = combined_score

        # Sort by combined score
        ranked_results = sorted(
            [data["result"] for data in combined.values()],
            key=lambda x: x.score,
            reverse=True,
        )

        return ranked_results[:top_k]

    def _apply_filters(
        self, results: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters to results"""
        filtered = []

        for result in results:
            match = True

            for key, value in filters.items():
                if key not in result:
                    match = False
                    break

                if isinstance(value, list):
                    if result[key] not in value:
                        match = False
                        break
                else:
                    if result[key] != value:
                        match = False
                        break

            if match:
                filtered.append(result)

        return filtered


async def main():
    """Run indexing service"""
    service = IndexingService()

    print("Building indices...")
    await service.index_all_calls()

    print("\nTesting search...")

    # Test semantic search
    results = service.semantic_search("customer complaints about fees", top_k=5)
    print(f"\nSemantic search results: {len(results)}")
    for r in results[:3]:
        print(f"  - {r.text[:100]}... (score: {r.score:.3f})")

    # Test hybrid search
    results = service.hybrid_search("refund request", top_k=5)
    print(f"\nHybrid search results: {len(results)}")
    for r in results[:3]:
        print(f"  - {r.text[:100]}... (score: {r.score:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
