"""
MedRAG — src/ingestion/chunker.py
===================================
Phase 1 — Hybrid Document Chunker

Two-pass chunking strategy:
  Pass 1 — SemanticChunker (bge-m3):
      Groups sentences by embedding similarity into thematically coherent
      chunks. Preserves clinical context across sentence boundaries.

  Pass 2 — RecursiveCharacterTextSplitter:
      Hard safety boundary. Any semantic chunk exceeding `max_chunk_size`
      tokens is split recursively to prevent LLM token overflow.
      Overlap ensures continuity of medical reasoning across splits.

Figure/Table chunks are passed through without re-splitting to preserve
their structural integrity.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.ingestion.parser import ParsedDocument


class HybridChunker:
    """
    Two-pass hybrid chunker combining semantic + recursive splitting.

    Args:
        embeddings: A LangChain Embeddings instance (bge-m3).
        max_chunk_size: Character ceiling for the recursive safety pass.
        chunk_overlap: Overlap characters for the recursive pass.
        semantic_breakpoint_type: SemanticChunker strategy.
            "percentile" | "standard_deviation" | "interquartile"
        semantic_breakpoint_threshold: Threshold for breakpoint detection.
    """

    def __init__(
        self,
        embeddings: Any,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 150,
        semantic_breakpoint_type: str = "percentile",
        semantic_breakpoint_threshold: float = 95.0,
    ) -> None:
        self.embeddings = embeddings
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Pass 1: Semantic chunker using bge-m3 similarity
        self._semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=semantic_breakpoint_type,
            breakpoint_threshold_amount=semantic_breakpoint_threshold,
            sentence_split_regex=r"(?<=[.?!。])\s+",  # handles Arabic + English
        )

        # Pass 2: Recursive hard-safety splitter
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",       # paragraph breaks
                "\n",         # line breaks
                ". ",         # sentence boundary (English)
                ".\n",
                "؟ ",         # Arabic question mark
                "! ",
                " ",
                "",
            ],
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
        )

        logger.info(
            f"HybridChunker initialized — "
            f"semantic_threshold={semantic_breakpoint_threshold} ({semantic_breakpoint_type}), "
            f"max_chunk_size={max_chunk_size}, overlap={chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, parsed_doc: ParsedDocument) -> list[Document]:
        """
        Chunk a ParsedDocument into final Document list.

        Figures/Tables bypass semantic chunking (already discrete units).
        Text blocks go through both passes.

        Returns:
            Flat list[Document] with all metadata preserved + chunk_index added.
        """
        text_docs = [
            d for d in parsed_doc.documents
            if d.metadata.get("chunk_type") == "text"
        ]
        non_text_docs = [
            d for d in parsed_doc.documents
            if d.metadata.get("chunk_type") != "text"
        ]

        logger.debug(
            f"Chunking {len(text_docs)} text docs + "
            f"{len(non_text_docs)} non-text docs from {parsed_doc.source_path}"
        )

        # Pass 1: Semantic chunking on text blocks
        semantic_chunks = self._semantic_pass(text_docs)

        # Pass 2: Recursive safety split
        final_text_chunks = self._recursive_pass(semantic_chunks)

        # Combine + index
        all_chunks = final_text_chunks + non_text_docs
        all_chunks = self._add_chunk_indices(all_chunks)

        logger.success(
            f"Chunking complete: {len(final_text_chunks)} text chunks + "
            f"{len(non_text_docs)} non-text → {len(all_chunks)} total chunks."
        )
        return all_chunks

    def chunk_many(self, parsed_docs: list[ParsedDocument]) -> list[Document]:
        """Chunk a list of ParsedDocuments and return combined flat list."""
        all_chunks: list[Document] = []
        for parsed_doc in parsed_docs:
            try:
                chunks = self.chunk(parsed_doc)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error(f"Chunking failed for {parsed_doc.source_path}: {exc}")
        logger.info(f"Total chunks across all documents: {len(all_chunks)}")
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _semantic_pass(self, text_docs: list[Document]) -> list[Document]:
        """
        Run SemanticChunker on text documents.

        Groups documents by source to ensure semantic boundaries respect
        document provenance (no cross-document merging).
        """
        if not text_docs:
            return []

        # Group by source to avoid cross-document contamination
        source_groups: dict[str, list[Document]] = {}
        for doc in text_docs:
            key = doc.metadata.get("source", "unknown")
            source_groups.setdefault(key, []).append(doc)

        semantic_chunks: list[Document] = []
        for source, docs in source_groups.items():
            combined_text = "\n\n".join(d.page_content for d in docs)
            # Inherit metadata from first doc as base
            base_meta = docs[0].metadata.copy()

            try:
                splits = self._semantic_splitter.create_documents(
                    texts=[combined_text],
                    metadatas=[base_meta],
                )
                semantic_chunks.extend(splits)
                logger.debug(
                    f"Semantic pass: {len(docs)} docs → {len(splits)} chunks "
                    f"[{source}]"
                )
            except Exception as exc:
                logger.warning(
                    f"SemanticChunker failed for {source}: {exc}. "
                    f"Falling back to raw docs."
                )
                semantic_chunks.extend(docs)

        return semantic_chunks

    def _recursive_pass(self, chunks: list[Document]) -> list[Document]:
        """
        Apply RecursiveCharacterTextSplitter as a hard safety ceiling.

        Chunks already within max_chunk_size are passed through unchanged.
        """
        final: list[Document] = []
        split_count = 0

        for chunk in chunks:
            if len(chunk.page_content) <= self.max_chunk_size:
                final.append(chunk)
            else:
                sub_chunks = self._recursive_splitter.split_documents([chunk])
                final.extend(sub_chunks)
                split_count += 1
                logger.debug(
                    f"Recursive split: 1 chunk ({len(chunk.page_content)} chars) "
                    f"→ {len(sub_chunks)} sub-chunks."
                )

        if split_count:
            logger.debug(f"Recursive safety pass: {split_count} oversized chunks split.")

        return final

    @staticmethod
    def _add_chunk_indices(chunks: list[Document]) -> list[Document]:
        """Add a sequential `chunk_index` to every chunk's metadata."""
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        return chunks
