"""
MedRAG — src/embeddings/embedder.py
=====================================
Embedding layer — BAAI/bge-m3

BGE-M3 characteristics relevant to MedRAG:
  • 1024-dimensional dense embeddings (excellent semantic capture)
  • Multilingual — covers Modern Standard Arabic, Egyptian Arabic dialect,
    and clinical English in a shared embedding space
  • Supports both dense retrieval (semantic similarity) and sparse retrieval
    (BGE-M3's ColBERT / lexical score) natively via FlagEmbedding

This module provides:
  1. get_hf_embeddings() → HuggingFaceEmbeddings (LangChain compatible)
     Used by: SemanticChunker, LangChain vectorstore wrappers
  2. get_fastembed_sparse() → FastEmbed BM25 sparse encoder
     Used by: Qdrant sparse vector uploads
  3. EmbedderService — high-level service that batches and embeds lists
     of Documents, returning (dense_vectors, sparse_vectors) pairs
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import get_settings


# ---------------------------------------------------------------------------
# LangChain-compatible HuggingFace Embeddings (Dense)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_hf_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a singleton HuggingFaceEmbeddings object for BAAI/bge-m3.

    Cached: model loads once per process (model is ~2.2 GB).

    Returns:
        HuggingFaceEmbeddings configured for 1024-dim dense vectors.
    """
    s = get_settings()
    logger.info(f"Loading embedding model: {s.embedding_model} on {s.embedding_device}")

    # BGE-M3 encode_kwargs: normalize embeddings for cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={
            "device": s.embedding_device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,  # Required for cosine similarity
            "batch_size": 32,
        },
        cache_folder=".cache/embeddings",
        show_progress=True,
    )
    logger.success(f"Embedding model loaded: {s.embedding_model}")
    return embeddings


# ---------------------------------------------------------------------------
# FastEmbed Sparse Encoder (BM25)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_sparse_encoder() -> Any:
    """
    Return a FastEmbed sparse text encoder for BM25 keyword retrieval.

    FastEmbed's SparseTextEmbedding uses SPLADE++ or BM25 under the hood
    and returns Qdrant-compatible SparseVector objects.
    """
    try:
        from fastembed import SparseTextEmbedding

        encoder = SparseTextEmbedding(
            model_name="Qdrant/bm25",
            cache_dir=".cache/sparse",
        )
        logger.info("Sparse encoder (BM25 via FastEmbed) loaded.")
        return encoder
    except ImportError:
        raise ImportError(
            "fastembed is not installed. Run: pip install fastembed"
        )


# ---------------------------------------------------------------------------
# High-level EmbedderService
# ---------------------------------------------------------------------------

class EmbedderService:
    """
    Orchestrates dense + sparse embedding for a list of Documents.

    Usage:
        service = EmbedderService()
        dense, sparse = service.embed_documents(documents)
        # dense: list[list[float]]  shape: (N, 1024)
        # sparse: list[SparseVector]
    """

    def __init__(self) -> None:
        self._dense_model = get_hf_embeddings()
        self._sparse_encoder = get_sparse_encoder()
        logger.info("EmbedderService ready (dense + sparse).")

    # ------------------------------------------------------------------
    # Dense embedding
    # ------------------------------------------------------------------

    def embed_texts_dense(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of strings into dense 1024-dim vectors.

        Args:
            texts: Raw strings (not Documents).

        Returns:
            List of 1024-dimensional float vectors.
        """
        if not texts:
            return []

        logger.debug(f"Dense embedding: {len(texts)} texts")
        vectors = self._dense_model.embed_documents(texts)
        return vectors

    def embed_query_dense(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        return self._dense_model.embed_query(query)

    # ------------------------------------------------------------------
    # Sparse embedding (BM25)
    # ------------------------------------------------------------------

    def embed_texts_sparse(self, texts: list[str]) -> list[Any]:
        """
        Encode texts into sparse BM25 vectors (Qdrant SparseVector format).

        Returns:
            List of SparseVector objects (indices + values arrays).
        """
        from qdrant_client.models import SparseVector

        if not texts:
            return []

        logger.debug(f"Sparse embedding: {len(texts)} texts")
        sparse_embeddings = list(self._sparse_encoder.embed(texts, batch_size=32))

        # Convert FastEmbed output to Qdrant SparseVector
        qdrant_sparse = []
        for emb in sparse_embeddings:
            indices = emb.indices.tolist()
            values = emb.values.tolist()
            qdrant_sparse.append(SparseVector(indices=indices, values=values))

        return qdrant_sparse

    def embed_query_sparse(self, query: str) -> Any:
        """Embed a single query into a sparse BM25 vector."""
        from qdrant_client.models import SparseVector

        results = list(self._sparse_encoder.embed([query], batch_size=1))
        emb = results[0]
        return SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )

    # ------------------------------------------------------------------
    # Combined embedding (Documents)
    # ------------------------------------------------------------------

    def embed_documents(
        self, documents: list[Document]
    ) -> tuple[list[list[float]], list[Any]]:
        """
        Embed a list of LangChain Documents.

        Returns:
            (dense_vectors, sparse_vectors) tuple.
            dense_vectors: list of 1024-dim float lists
            sparse_vectors: list of SparseVector objects
        """
        texts = [doc.page_content for doc in documents]
        dense = self.embed_texts_dense(texts)
        sparse = self.embed_texts_sparse(texts)
        logger.info(f"Embedded {len(documents)} documents (dense + sparse).")
        return dense, sparse

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two dense vectors."""
        va = np.array(a)
        vb = np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
