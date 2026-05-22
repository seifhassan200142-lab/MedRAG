"""
MedRAG — src/retrieval/searcher.py
=====================================
Phase 3 — Advanced Retrieval

Full retrieval pipeline:
  1. Hybrid Search
     Dense semantic search (BGE-M3 1024-dim cosine) +
     Sparse keyword search (BM25 via FastEmbed)
     → Run against all `search_queries` from AugmentedQuery

  2. RRF — Reciprocal Rank Fusion
     Merge and normalize ranked lists from dense + sparse results.
     RRF formula: score(d, R) = Σ 1 / (k + rank(d, r))  ∀r∈R
     Default k=60 (empirically optimal for retrieval fusion).

  3. Cross-Encoder Reranking
     BAAI/bge-reranker-large: computes a relevance score for each
     (query, passage) pair directly — far more accurate than
     bi-encoder similarity but requires all candidates to be loaded
     into memory. Applied as the final gate to select Top-K.

Architecture note: async-first for non-blocking FastAPI/Chainlit integration.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from src.config import get_qdrant_client, get_settings
from src.embeddings.embedder import EmbedderService
from src.retrieval.expansion import AugmentedQuery

# Named vector names must match qdrant_db.py
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single retrieved context chunk with scores and metadata."""
    content: str
    metadata: dict[str, Any]
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def source(self) -> str:
        return self.metadata.get("filename", self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int:
        return self.metadata.get("page", 0)

    @property
    def chunk_type(self) -> str:
        return self.metadata.get("chunk_type", "text")

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={**self.metadata, "rrf_score": self.rrf_score, "rerank_score": self.rerank_score},
        )


@dataclass
class SearchResult:
    """Complete result from the retrieval pipeline."""
    query: str
    augmented_query: AugmentedQuery | None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    total_candidates: int = 0

    @property
    def documents(self) -> list[Document]:
        return [c.to_document() for c in self.chunks]

    @property
    def context_text(self) -> str:
        """Concatenated context for LLM consumption."""
        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            parts.append(
                f"[Source {i}: {chunk.source}, Page {chunk.page}, "
                f"Type: {chunk.chunk_type}]\n{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    BAAI/bge-reranker-large cross-encoder for precise relevance scoring.

    The cross-encoder reads (query, passage) together, enabling far
    richer interaction than dot-product bi-encoder similarity.
    """

    def __init__(self) -> None:
        from FlagEmbedding import FlagReranker

        s = get_settings()
        logger.info(f"Loading cross-encoder: {s.reranker_model}")
        try:
            self._reranker = FlagReranker(
                s.reranker_model,
                use_fp16=True,   # ~2× faster, negligible accuracy loss
            )
            logger.success(f"Cross-encoder loaded: {s.reranker_model}")
        except Exception as exc:
            logger.warning(
                "Cross-encoder load failed (%s); falling back to RRF-only reranking.",
                exc,
            )
            self._reranker = None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Score all (query, chunk) pairs and return top_k by score.

        Args:
            query: The clinical query string (expanded form preferred).
            chunks: Candidate chunks from hybrid search + RRF.
            top_k: Number of chunks to return.

        Returns:
            Top-k RetrievedChunk objects sorted by rerank_score descending.
        """
        if not chunks:
            return []

        if self._reranker is None:
            logger.warning(
                "Cross-encoder unavailable; using RRF ranks only for final selection."
            )
            for chunk in chunks:
                chunk.rerank_score = chunk.rrf_score
            ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
            return ranked[:top_k]

        pairs = [[query, chunk.content] for chunk in chunks]

        logger.debug(f"Reranking {len(pairs)} pairs with cross-encoder…")
        try:
            scores = self._reranker.compute_score(pairs, normalize=True)
        except Exception as exc:
            logger.warning(
                "Cross-encoder scoring failed (%s); falling back to RRF-only reranking.",
                exc,
            )
            for chunk in chunks:
                chunk.rerank_score = chunk.rrf_score
            ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
            return ranked[:top_k]

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        selected = ranked[:top_k]

        logger.info(
            f"Reranking: {len(chunks)} → {len(selected)} chunks | "
            f"Top score: {selected[0].rerank_score:.4f}"
        )
        return selected


# ---------------------------------------------------------------------------
# Hybrid Searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """
    Executes the full Phase 3 retrieval pipeline.

    Usage:
        searcher = HybridSearcher()
        result = await searcher.search(augmented_query)
        # result.documents → final top-k LangChain Documents
        # result.chunks    → RetrievedChunk with scores + metadata
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_qdrant_client()
        self.embedder = EmbedderService()
        self.reranker = CrossEncoderReranker()
        self.collection_name = self.settings.qdrant_collection

        logger.info("HybridSearcher initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        augmented_query: AugmentedQuery,
        metadata_filter: dict[str, str] | None = None,
    ) -> SearchResult:
        """
        Run full hybrid search pipeline for an AugmentedQuery.

        Args:
            augmented_query: From Phase 2 expansion.
            metadata_filter: Optional {"source": "...", "chunk_type": "..."}

        Returns:
            SearchResult with reranked top-k chunks.
        """
        result = SearchResult(
            query=augmented_query.original,
            augmented_query=augmented_query,
        )

        # Collect candidates from all search queries (expanded + HyDE)
        all_candidates: list[RetrievedChunk] = []
        for q in augmented_query.search_queries:
            candidates = await self._hybrid_search_single(q, metadata_filter)
            all_candidates.extend(candidates)

        result.total_candidates = len(all_candidates)

        # Deduplicate by content hash
        all_candidates = self._deduplicate(all_candidates)

        # RRF: fuse dense + sparse rankings
        fused = self._rrf_fuse(all_candidates)

        # Cross-encoder reranking
        top_k = self.settings.final_top_k
        reranked = self.reranker.rerank(
            query=augmented_query.expanded,
            chunks=fused,
            top_k=top_k,
        )

        result.chunks = reranked
        logger.info(
            f"Search complete: {result.total_candidates} candidates → "
            f"{len(fused)} after RRF → {len(reranked)} final chunks."
        )
        return result

    async def search_raw(
        self,
        query: str,
        top_k: int | None = None,
    ) -> SearchResult:
        """
        Search with a plain string query (no augmentation).
        Useful for testing and direct API calls.
        """
        from src.retrieval.expansion import AugmentedQuery

        aq = AugmentedQuery(
            original=query,
            expanded=query,
            hyde_passage=query,
        )
        result = await self.search(aq)
        if top_k and top_k != self.settings.final_top_k:
            result.chunks = result.chunks[:top_k]
        return result

    # ------------------------------------------------------------------
    # Hybrid search for a single query string
    # ------------------------------------------------------------------

    async def _hybrid_search_single(
        self,
        query: str,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Execute dense + sparse searches concurrently and collect results.

        Dense and sparse results carry separate rank lists; RRF fuses them.
        """
        filter_obj = self._build_filter(metadata_filter)

        # Embed concurrently
        dense_vec = await asyncio.get_event_loop().run_in_executor(
            None, self.embedder.embed_query_dense, query
        )
        sparse_vec = await asyncio.get_event_loop().run_in_executor(
            None, self.embedder.embed_query_sparse, query
        )

        # Run both searches concurrently
        dense_task = asyncio.create_task(
            self._dense_search(dense_vec, filter_obj)
        )
        sparse_task = asyncio.create_task(
            self._sparse_search(sparse_vec, filter_obj)
        )
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        logger.debug(
            f"Query '{query[:50]}': dense={len(dense_results)}, sparse={len(sparse_results)}"
        )
        return dense_results + sparse_results

    async def _dense_search(
        self, dense_vector: list[float], filter_obj: Any
    ) -> list[RetrievedChunk]:
        """Run dense cosine similarity search."""
        from qdrant_client.http.models import models

        top_k = self.settings.hybrid_dense_top_k
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=models.NearestQuery(nearest=dense_vector),
                using=DENSE_VECTOR_NAME,
                query_filter=filter_obj,
                limit=top_k,
                with_payload=True,
                score_threshold=0.1,
            ),
        )
        points = self._extract_points(response)
        chunks = []
        for rank, hit in enumerate(points):
            chunk = self._hit_to_chunk(hit)
            chunk.rrf_score = self._rrf_score(rank, label="dense")
            chunks.append(chunk)
        return chunks

    async def _sparse_search(
        self, sparse_vector: Any, filter_obj: Any
    ) -> list[RetrievedChunk]:
        """Run sparse BM25 keyword search."""
        from qdrant_client.http.models import models

        top_k = self.settings.hybrid_sparse_top_k
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=models.NearestQuery(nearest=sparse_vector),
                using=SPARSE_VECTOR_NAME,
                query_filter=filter_obj,
                limit=top_k,
                with_payload=True,
            ),
        )
        points = self._extract_points(response)
        chunks = []
        for rank, hit in enumerate(points):
            chunk = self._hit_to_chunk(hit)
            chunk.rrf_score = self._rrf_score(rank, label="sparse")
            chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # Qdrant response helpers

    @staticmethod
    def _extract_points(response: Any) -> list[Any]:
        """Normalize Qdrant query_points response to a list of scored points."""
        if isinstance(response, tuple) and response:
            response = response[0]

        if hasattr(response, "points"):
            return response.points

        if isinstance(response, list):
            return response

        raise ValueError(
            f"Unsupported Qdrant query_points response type: {type(response)}"
        )

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_score(rank: int, label: str = "", k: int = 60) -> float:
        """Compute RRF score: 1 / (k + rank)."""
        return 1.0 / (k + rank + 1)

    def _rrf_fuse(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Aggregate RRF scores across all candidate lists.

        Multiple hits for the same chunk (from dense + sparse + HyDE queries)
        accumulate higher RRF scores, naturally surfacing the best matches.
        """
        score_map: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for chunk in candidates:
            key = hash(chunk.content[:200])  # content fingerprint
            score_map[key] += chunk.rrf_score
            if key not in chunk_map:
                chunk_map[key] = chunk

        # Assign fused RRF scores and sort
        fused = []
        for key, chunk in chunk_map.items():
            chunk.rrf_score = score_map[key]
            fused.append(chunk)

        fused.sort(key=lambda c: c.rrf_score, reverse=True)

        # Keep top candidates for reranking (reranker scales as O(n))
        pre_rerank_k = min(len(fused), self.settings.final_top_k * 5)
        logger.debug(f"RRF fused {len(candidates)} → {len(fused)} unique → top {pre_rerank_k}")
        return fused[:pre_rerank_k]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_to_chunk(hit: Any) -> RetrievedChunk:
        """Convert a Qdrant ScoredPoint to a RetrievedChunk."""
        payload = hit.payload or {}
        content = payload.pop("page_content", "")
        return RetrievedChunk(
            content=content,
            metadata=payload,
        )

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove duplicate chunks by content fingerprint."""
        seen: set[int] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            fp = hash(chunk.content[:200])
            if fp not in seen:
                seen.add(fp)
                unique.append(chunk)
        return unique

    @staticmethod
    def _build_filter(metadata_filter: dict[str, str] | None) -> Any | None:
        """Build a Qdrant Filter from a metadata dict."""
        if not metadata_filter:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in metadata_filter.items()
        ]
        return Filter(must=conditions)
