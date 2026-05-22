"""
MedRAG — src/ingestion/parser.py
==================================
Phase 1 — Document Parsing & Visual Extraction

Pipeline:
  PDF file
    └─► Docling  →  text blocks  +  figure bounding boxes
          └─► PIL crop image region
                └─► VLM (Gemini-1.5-Flash | Llama-3.2-Vision)
                      └─► medical image caption
    └─► Combined list[Document] with rich metadata

Metadata schema per chunk:
  {
    "source":       str,   # absolute file path
    "filename":     str,   # basename
    "page":         int,   # 1-indexed
    "chunk_type":   str,   # "text" | "table" | "figure"
    "figure_index": int,   # only for figures
    "bbox":         list,  # [x0, y0, x1, y1] normalized 0-1
  }
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import VLMProvider, get_settings


# ---------------------------------------------------------------------------
# Data class for raw parse results before chunking
# ---------------------------------------------------------------------------

@dataclass
class ParsedDocument:
    """Raw output from the parser — list of LangChain Documents."""
    source_path: str
    documents: list[Document] = field(default_factory=list)
    figure_count: int = 0
    page_count: int = 0


# ---------------------------------------------------------------------------
# VLM Image Captioning
# ---------------------------------------------------------------------------

class VLMCaptioner:
    """
    Generates medical descriptions for cropped figure images.

    Provider-agnostic: uses Gemini (cloud) or Llama-3.2-Vision (local Ollama).
    """

    SYSTEM_PROMPT = (
        "You are a medical imaging assistant. "
        "Describe the following medical figure, chart, or diagram extracted from a clinical document. "
        "Be concise, accurate, and use standard medical terminology. "
        "Focus on: anatomical structures visible, any measurements or scales shown, "
        "pathological findings, and clinical significance. "
        "Limit your response to 3-5 sentences."
    )

    def __init__(self) -> None:
        self.settings = get_settings()
        self._provider = self.settings.vlm_provider
        logger.info(f"VLMCaptioner initialized — provider={self._provider.value}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def caption(self, image: Image.Image) -> str:
        """Convert a PIL Image to a medical caption string."""
        try:
            if self._provider == VLMProvider.GEMINI:
                return self._caption_gemini(image)
            elif self._provider == VLMProvider.OLLAMA:
                return self._caption_ollama(image)
            raise ValueError(f"Unknown VLM provider: {self._provider}")
        except Exception as exc:
            logger.warning(f"VLM captioning failed: {exc}")
            return "[Image description unavailable]"

    def _pil_to_base64(self, image: Image.Image, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _caption_gemini(self, image: Image.Image) -> str:
        import google.generativeai as genai

        api_key = self.settings.google_api_key
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set for Gemini VLM.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Gemini accepts PIL images directly
        response = model.generate_content(
            [self.SYSTEM_PROMPT, image],
            generation_config=genai.GenerationConfig(max_output_tokens=256),
        )
        return response.text.strip()

    def _caption_ollama(self, image: Image.Image) -> str:
        import httpx

        b64 = self._pil_to_base64(image)
        payload = {
            "model": self.settings.vlm_ollama_model,
            "prompt": self.SYSTEM_PROMPT,
            "images": [b64],
            "stream": False,
        }
        resp = httpx.post(
            f"{self.settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------

class MedicalDocumentParser:
    """
    Orchestrates Docling-based PDF parsing + VLM figure captioning.

    Usage:
        parser = MedicalDocumentParser()
        parsed = parser.parse("/path/to/clinical_paper.pdf")
        # parsed.documents → list[Document] ready for chunking
    """

    def __init__(self, caption_figures: bool = True) -> None:
        self.caption_figures = caption_figures
        self.captioner = VLMCaptioner() if caption_figures else None
        self._init_docling()

    def _init_docling(self) -> None:
        """Lazy-import docling to avoid slow startup when not ingesting."""
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True          # OCR scanned pages
            pipeline_options.do_table_structure = True  # Detect tables

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Docling DocumentConverter initialized with OCR + table structure.")
        except ImportError as exc:
            raise ImportError(
                "Docling is not installed. Run: pip install docling"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a single PDF file into a ParsedDocument.

        Args:
            pdf_path: Absolute or relative path to the PDF.

        Returns:
            ParsedDocument with .documents list ready for chunking.
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")
        result = self._converter.convert(str(pdf_path))
        doc = result.document

        parsed = ParsedDocument(source_path=str(pdf_path))
        parsed.page_count = len(doc.pages) if hasattr(doc, "pages") else 0

        # ── Text & Table blocks ──────────────────────────────────────
        text_docs = self._extract_text_blocks(doc, pdf_path)
        parsed.documents.extend(text_docs)

        # ── Figure blocks ────────────────────────────────────────────
        if self.caption_figures:
            figure_docs = self._extract_figures(doc, pdf_path)
            parsed.documents.extend(figure_docs)
            parsed.figure_count = len(figure_docs)

        logger.success(
            f"Parsed {pdf_path.name}: {len(text_docs)} text chunks, "
            f"{parsed.figure_count} figures, {parsed.page_count} pages."
        )
        return parsed

    def parse_directory(self, directory: str | Path) -> list[ParsedDocument]:
        """Parse all PDFs in a directory recursively."""
        directory = Path(directory)
        pdfs = list(directory.rglob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {directory}")
        results = []
        for pdf in pdfs:
            try:
                results.append(self.parse(pdf))
            except Exception as exc:
                logger.error(f"Failed to parse {pdf.name}: {exc}")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text_blocks(
        self, doc: Any, source_path: Path
    ) -> list[Document]:
        """Extract text and table content from a Docling document."""
        documents: list[Document] = []

        for element, _level in doc.iterate_items():
            # Docling element types: TextItem, TableItem, etc.
            element_type = type(element).__name__

            if "Table" in element_type:
                chunk_type = "table"
                text = element.export_to_markdown() if hasattr(element, "export_to_markdown") else str(element)
            elif "Text" in element_type or "Section" in element_type or "Para" in element_type:
                chunk_type = "text"
                text = element.text if hasattr(element, "text") else str(element)
            else:
                continue  # skip non-content elements (headers handled inside text)

            text = text.strip()
            if len(text) < 20:
                continue  # Skip trivially short fragments

            page_num = self._get_page_number(element)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(source_path),
                        "filename": source_path.name,
                        "page": page_num,
                        "chunk_type": chunk_type,
                    },
                )
            )

        return documents

    def _extract_figures(
        self, doc: Any, source_path: Path
    ) -> list[Document]:
        """
        Extract figure/image elements, crop them from the page render,
        and generate VLM captions.
        """
        from docling.datamodel.base_models import FigureElement

        documents: list[Document] = []
        figure_idx = 0

        # Render PDF pages as images for cropping
        page_images = self._render_pages(str(source_path))

        for element, _level in doc.iterate_items():
            if not isinstance(element, FigureElement):
                continue

            page_num = self._get_page_number(element)
            page_img = page_images.get(page_num)

            if page_img is None:
                logger.warning(f"No rendered page for figure on page {page_num}")
                continue

            try:
                cropped = self._crop_figure(page_img, element)
                caption = self.captioner.caption(cropped)

                # Normalized bounding box for metadata
                w, h = page_img.size
                bbox = self._get_bbox(element, w, h)

                figure_idx += 1
                documents.append(
                    Document(
                        page_content=f"[FIGURE {figure_idx}] {caption}",
                        metadata={
                            "source": str(source_path),
                            "filename": source_path.name,
                            "page": page_num,
                            "chunk_type": "figure",
                            "figure_index": figure_idx,
                            "bbox": bbox,
                        },
                    )
                )
                logger.debug(f"Captioned figure {figure_idx} on page {page_num}")

            except Exception as exc:
                logger.warning(f"Figure extraction failed on page {page_num}: {exc}")

        return documents

    def _render_pages(self, pdf_path: str) -> dict[int, Image.Image]:
        """Render each PDF page to a PIL Image using pypdfium2."""
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(pdf_path)
            pages: dict[int, Image.Image] = {}
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=2.0)  # 2× for better crop quality
                pages[i + 1] = bitmap.to_pil()
            return pages
        except Exception as exc:
            logger.error(f"Page rendering failed: {exc}")
            return {}

    def _crop_figure(self, page_img: Image.Image, element: Any) -> Image.Image:
        """Crop a figure bounding box from a rendered page image."""
        w, h = page_img.size
        try:
            bbox = element.prov[0].bbox  # Docling bbox object
            x0 = int(bbox.l * w)
            y0 = int((1 - bbox.t) * h)  # Docling uses bottom-left origin
            x1 = int(bbox.r * w)
            y1 = int((1 - bbox.b) * h)
            x0, y0, x1, y1 = (
                max(0, x0), max(0, y0),
                min(w, x1), min(h, y1),
            )
            return page_img.crop((x0, y0, x1, y1))
        except (AttributeError, IndexError):
            # Fallback: return full page
            return page_img

    def _get_page_number(self, element: Any) -> int:
        """Safely extract 1-indexed page number from a Docling element."""
        try:
            return element.prov[0].page_no
        except (AttributeError, IndexError):
            return 0

    def _get_bbox(self, element: Any, page_w: int, page_h: int) -> list[float]:
        """Return normalized [x0, y0, x1, y1] bounding box."""
        try:
            bbox = element.prov[0].bbox
            return [
                round(bbox.l, 4),
                round(1 - bbox.t, 4),
                round(bbox.r, 4),
                round(1 - bbox.b, 4),
            ]
        except (AttributeError, IndexError):
            return [0.0, 0.0, 1.0, 1.0]
