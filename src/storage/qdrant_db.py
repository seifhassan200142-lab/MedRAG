"""
MedRAG — src/storage/qdrant_db.py
====================================
Phase 1 — Qdrant Vector Store Management

Responsibilities:
  1. Collection creation with BOTH Dense (1024-dim cosine) and
     Sparse (BM25 keyword) named vectors — enables Hybrid Search.
  2. Payload indexing on 'source' and 'chunk_type' for fast metadata filtering.
  3. Batch upsert of documents with their dense + sparse vectors.
  4. Collection health checks and stats.

Qdrant Named Vector Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Collection: medrag_v1                                          │
  │                                                                 │
  │  Named vectors:                                                 │
  │    "dense"  → VectorParams(size=1024, distance=Cosine)          │
  │    "sparse" → SparseVectorParams(index=SparseIndexParams)       │
  │                                                                 │
  │  Payload (metadata):                                            │
  │    source, filename, page, chunk_type, chunk_index, bbox, …    │
  │                                                                 │
  │  Payload indexes:                                               │
  │    source     → keyword index (fast filter)                     │
  │    chunk_type → keyword index (fast filter)                     │
  └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_qdrant_client, get_settings

# Constants
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1024  # BGE-M3 output dimension
UPSERT_BATCH_SIZE = 64   # Points per upsert batch


# ---------------------------------------------------------------------------
# Collection Manager
# ---------------------------------------------------------------------------

class QdrantManager:
    """
    Manages the MedRAG Qdrant collection lifecycle.

    All interactions are synchronous (used during ingestion).
    Query-time interactions use AsyncQdrantClient via QdrantSearcher.
    """

    def __init__(self, client: QdrantClient | None = None) -> None:
        self.client = client or get_qdrant_client()
        self.settings = get_settings()
        self.collection_name = self.settings.qdrant_collection

    # ------------------------------------------------------------------
    # Collection Lifecycle
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with dense + sparse named vectors.

        Args:
            recreate: If True, drops existing collection before creating.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            if not recreate:
                logger.info(
                    f"Collection '{self.collection_name}' already exists. "
                    f"Skipping creation. Use recreate=True to reset."
                )
                return
            logger.warning(f"Recreating collection '{self.collection_name}'…")
            self.client.delete_collection(self.collection_name)

        logger.info(f"Creating collection '{self.collection_name}' with hybrid vectors…")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=False,          # Keep in memory for fast retrieval
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,      # BM25 index in memory
                    )
                )
            },
            # Optimizers for medical search workloads
            optimizers_config={
                "indexing_threshold": 20_000,  # Build HNSW after 20k points
                "memmap_threshold": 100_000,
            },
        )

        # Payload indexes for fast metadata filtering
        self._create_payload_indexes()

        logger.success(
            f"Collection '{self.collection_name}' created with "
            f"dense ({DENSE_VECTOR_SIZE}-dim cosine) + sparse (BM25) vectors."
        )

    def _create_payload_indexes(self) -> None:
        """Create keyword payload indexes for fast filter queries."""
        indexed_fields = ["source", "chunk_type", "filename", "page"]
        for field in indexed_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=(
                    PayloadSchemaType.KEYWORD
                    if field != "page"
                    else PayloadSchemaType.INTEGER
                ),
            )
            logger.debug(f"Payload index created: {field}")

    def collection_info(self) -> dict[str, Any]:
        """Return collection status and point count."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
            "vectors_count": info.vectors_count,
        }

    def collection_exists(self) -> bool:
        """Return True if the collection exists."""
        return self.client.collection_exists(self.collection_name)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def upsert_documents(
        self,
        documents: list[Document],
        dense_vectors: list[list[float]],
        sparse_vectors: list[Any],
    ) -> int:
        """
        Upsert documents with their dense + sparse vectors into Qdrant.

        Args:
            documents: LangChain Documents with .page_content and .metadata.
            dense_vectors: Parallel list of 1024-dim float vectors.
            sparse_vectors: Parallel list of Qdrant SparseVector objects.

        Returns:
            Number of points successfully upserted.
        """
        if not (len(documents) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError(
                f"Mismatched lengths: docs={len(documents)}, "
                f"dense={len(dense_vectors)}, sparse={len(sparse_vectors)}"
            )

        points: list[PointStruct] = []
        for doc, dense, sparse in zip(documents, dense_vectors, sparse_vectors):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: sparse,
                    },
                    payload={
                        "page_content": doc.page_content,
                        **doc.metadata,
                    },
                )
            )

        # Batch upsert
        total_upserted = 0
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
            total_upserted += len(batch)
            logger.debug(
                f"Upserted batch {i // UPSERT_BATCH_SIZE + 1}: "
                f"{len(batch)} points ({total_upserted}/{len(points)} total)"
            )

        logger.success(f"Upserted {total_upserted} points to '{self.collection_name}'.")
        return total_upserted

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_filter(
        source: str | None = None,
        chunk_type: str | None = None,
    ) -> Filter | None:
        """
        Build a Qdrant payload filter for source and/or chunk_type.

        Args:
            source: Filter by specific source file path.
            chunk_type: "text" | "table" | "figure"

        Returns:
            Qdrant Filter object, or None if no filters requested.
        """
        conditions = []
        if source:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source))
            )
        if chunk_type:
            conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))
            )
        if not conditions:
            return None
        from qdrant_client.models import Filter as QFilter, Must
        return QFilter(must=conditions)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_by_source(self, source_path: str) -> None:
        """Delete all points from a specific source document."""
        from qdrant_client.models import Filter as QFilter, FieldCondition, MatchValue, Must

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=QFilter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_path))]
            ),
        )
        logger.info(f"Deleted points for source: {source_path}")


# ---------------------------------------------------------------------------
# High-level Ingestion Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
      ParsedDocuments → Chunked Documents → Embedded → Qdrant

    Usage:
        pipeline = IngestionPipeline()
        pipeline.run_from_directory("/data/pdfs/")
    """

    def __init__(self) -> None:
        from src.embeddings.embedder import EmbedderService
        from src.ingestion.chunker import HybridChunker
        from src.ingestion.parser import MedicalDocumentParser

        self.parser = MedicalDocumentParser(caption_figures=True)
        self.embedder = EmbedderService()
        self.chunker = HybridChunker(embeddings=self.embedder._dense_model)
        self.db = QdrantManager()

    def run(self, pdf_paths: list[str]) -> dict[str, Any]:
        """
        Run end-to-end ingestion for a list of PDF paths.

        Returns:
            Summary dict with counts and status.
        """
        # Ensure collection exists
        if not self.db.collection_exists():
            self.db.create_collection()

        all_chunks: list[Document] = []
        for path in pdf_paths:
            try:
                parsed = self.parser.parse(path)
                chunks = self.chunker.chunk(parsed)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error(f"Ingestion failed for {path}: {exc}")

        if not all_chunks:
            return {"status": "no_chunks", "upserted": 0}

        # Embed
        dense_vecs, sparse_vecs = self.embedder.embed_documents(all_chunks)

        # Store
        upserted = self.db.upsert_documents(all_chunks, dense_vecs, sparse_vecs)

        stats = self.db.collection_info()
        logger.success(f"Ingestion complete. Collection stats: {stats}")
        return {
            "status": "success",
            "pdfs_processed": len(pdf_paths),
            "chunks_generated": len(all_chunks),
            "upserted": upserted,
            "collection": stats,
        }

    def run_from_directory(self, directory: str) -> dict[str, Any]:
        """Ingest all PDFs from a directory."""
        from pathlib import Path
        pdfs = [str(p) for p in Path(directory).rglob("*.pdf")]
        logger.info(f"Starting ingestion of {len(pdfs)} PDFs from {directory}")
        return self.run(pdfs)
