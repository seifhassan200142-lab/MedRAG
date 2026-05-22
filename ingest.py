#!/usr/bin/env python3
"""
MedRAG — ingest.py
===================
CLI script for batch PDF ingestion into Qdrant.

Usage:
    python ingest.py --dir /path/to/pdfs
    python ingest.py --files paper1.pdf paper2.pdf
    python ingest.py --recreate --dir /data/medical_docs
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Ensure src is importable from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import configure_logging
from src.storage.qdrant_db import IngestionPipeline, QdrantManager


def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="MedRAG PDF Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=str, help="Directory containing PDFs to ingest")
    parser.add_argument("--files", nargs="+", type=str, help="Specific PDF files to ingest")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection (WARNING: deletes all existing data)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip VLM figure captioning (faster, less rich)",
    )
    args = parser.parse_args()

    if not args.dir and not args.files:
        parser.print_help()
        sys.exit(1)

    # Optionally recreate collection
    if args.recreate:
        logger.warning("Recreating Qdrant collection — all existing data will be lost!")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            logger.info("Aborted.")
            sys.exit(0)
        db = QdrantManager()
        db.create_collection(recreate=True)

    # Build pipeline
    from src.ingestion.parser import MedicalDocumentParser
    from src.ingestion.chunker import HybridChunker
    from src.embeddings.embedder import EmbedderService

    embedder = EmbedderService()
    pipeline = IngestionPipeline()

    # Override figure captioning flag
    if args.no_figures:
        pipeline.parser = MedicalDocumentParser(caption_figures=False)

    # Collect PDF paths
    pdf_paths: list[str] = []
    if args.dir:
        pdf_paths.extend(str(p) for p in Path(args.dir).rglob("*.pdf"))
        logger.info(f"Found {len(pdf_paths)} PDFs in {args.dir}")

    if args.files:
        for f in args.files:
            p = Path(f)
            if p.exists() and p.suffix.lower() == ".pdf":
                pdf_paths.append(str(p))
            else:
                logger.warning(f"Skipping (not found or not PDF): {f}")

    if not pdf_paths:
        logger.error("No valid PDF files found. Exiting.")
        sys.exit(1)

    # Run ingestion
    logger.info(f"Starting ingestion of {len(pdf_paths)} PDF(s)…")
    result = pipeline.run(pdf_paths)

    # Summary
    print("\n" + "═" * 50)
    print("  MEDRAG INGESTION COMPLETE")
    print("═" * 50)
    print(f"  PDFs processed : {result.get('pdfs_processed', 0)}")
    print(f"  Chunks created : {result.get('chunks_generated', 0)}")
    print(f"  Vectors stored : {result.get('upserted', 0)}")
    if result.get("collection"):
        col = result["collection"]
        print(f"  Collection     : {col['name']} ({col['points_count']} total points)")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
