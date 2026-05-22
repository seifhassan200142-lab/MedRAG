"""
MedRAG — src/app.py
=====================
Premium Clinical AI Interface — Chainlit

Features:
  • Token-by-token streaming with animated thinking indicator
  • Expandable source cards showing exact PDF chunks, page numbers, and types
  • Patient session sidebar with conversation summary
  • Arabic ↔ English auto-detection badge
  • Ingestion endpoint (drag-and-drop PDF upload)
  • Full session reset for new patient consultations

Launch:
    chainlit run src/app.py -w --port 8000
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import chainlit as cl
from chainlit.element import Text, Pdf, Image as ClImage
from loguru import logger

from src.config import configure_logging, get_settings
from src.generation.llm_chain import MedRAGChain
from src.retrieval.expansion import QueryAugmentor
from src.retrieval.searcher import HybridSearcher, SearchResult

configure_logging()
settings = get_settings()


# ---------------------------------------------------------------------------
# Chainlit session-scoped singletons
# ---------------------------------------------------------------------------

def _get_chain() -> MedRAGChain:
    """Get or create per-session MedRAGChain."""
    if not cl.user_session.get("chain"):
        cl.user_session.set("chain", MedRAGChain())
    return cl.user_session.get("chain")


def _get_searcher() -> HybridSearcher:
    """Get or create per-session HybridSearcher."""
    if not cl.user_session.get("searcher"):
        cl.user_session.set("searcher", HybridSearcher())
    return cl.user_session.get("searcher")


def _get_augmentor() -> QueryAugmentor:
    """Get or create per-session QueryAugmentor."""
    if not cl.user_session.get("augmentor"):
        cl.user_session.set("augmentor", QueryAugmentor())
    return cl.user_session.get("augmentor")


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Initialize session and display welcome screen."""
    # Eagerly load services
    _get_chain()
    _get_searcher()
    _get_augmentor()

    cl.user_session.set("turn_count", 0)
    cl.user_session.set("sources_history", [])

    # Welcome message with rich markdown
    welcome = """
## 🏥 MedRAG — Clinical AI Assistant

Welcome! I'm **MedRAG**, your intelligent clinical knowledge assistant.

I can answer medical questions in **English or Arabic** (including Egyptian dialect), 
drawing from your indexed clinical documents.

---

### 🚀 Getting Started

| Action | How |
|--------|-----|
| Ask a question | Type in English or Arabic below |
| Upload a PDF | Click the 📎 attachment icon |
| New patient session | Type `/reset` |
| View knowledge base status | Type `/status` |

---

> ⚠️ **Disclaimer**: MedRAG provides information from clinical documents for educational purposes only. Always consult a licensed healthcare professional for medical decisions.

---

*Type your first question to begin. I'm ready.*
"""
    await cl.Message(content=welcome, author="MedRAG").send()


@cl.on_chat_end
async def on_chat_end():
    """Cleanup on session end."""
    chain = cl.user_session.get("chain")
    if chain:
        chain.reset_session()
    logger.info("Session ended and memory cleared.")


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""

    # ── Command routing ──────────────────────────────────────────────
    text = message.content.strip()

    if text.lower() == "/reset":
        await _handle_reset()
        return

    if text.lower() == "/status":
        await _handle_status()
        return

    # ── File upload (PDF ingestion) ──────────────────────────────────
    if message.elements:
        await _handle_file_upload(message)
        return

    # ── Main Q&A Pipeline ────────────────────────────────────────────
    await _handle_qa(message)


# ---------------------------------------------------------------------------
# Q&A pipeline
# ---------------------------------------------------------------------------

async def _handle_qa(message: cl.Message):
    """Full RAG pipeline: augment → retrieve → stream response."""
    query = message.content.strip()

    if _is_greeting(query):
        await _send_greeting_response()
        return

    augmentor = _get_augmentor()
    searcher = _get_searcher()
    chain = _get_chain()

    turn = cl.user_session.get("turn_count", 0) + 1
    cl.user_session.set("turn_count", turn)

    # ── Step 1: Query augmentation ───────────────────────────────────
    async with cl.Step(name="Analyzing Query", show_input=False) as step:
        augmented = await augmentor.augment(query)

        # Detect Arabic
        is_arabic = QueryAugmentor.is_arabic(query)
        lang_badge = "🇪🇬 Arabic → Clinical EN" if is_arabic else "🇬🇧 English"

        step.output = (
            f"**Language detected**: {lang_badge}\n\n"
            f"**Clinical terms**: `{augmented.expanded}`\n\n"
            f"**HyDE passage preview**: _{augmented.hyde_passage[:150]}…_"
        )

    # ── Step 2: Hybrid retrieval ─────────────────────────────────────
    async with cl.Step(name="📚 Searching Knowledge Base", show_input=False) as step:
        search_result = await searcher.search(augmented)

        step.output = (
            f"**Candidates retrieved**: {search_result.total_candidates}\n\n"
            f"**After RRF + Reranking**: {len(search_result.chunks)} chunks selected\n\n"
            + _format_sources_summary(search_result)
        )

    # ── Step 3: Streaming generation ─────────────────────────────────
    response_msg = cl.Message(content="", author="MedRAG")
    await response_msg.send()

    full_response = ""
    try:
        async for token in chain.astream(query, search_result):
            await response_msg.stream_token(token)
            full_response += token

        await response_msg.update()
    except Exception as exc:
        logger.warning(
            "Streaming generation failed, falling back to full response: %s",
            exc,
        )
        full_response = await chain.ainvoke(query, search_result)
        await response_msg.update(content=full_response)

    # ── Step 4: Source cards ─────────────────────────────────────────
    if search_result.chunks:
        await _send_source_cards(search_result, response_msg)

    # ── Step 5: Update sidebar summary ──────────────────────────────
    await _update_sidebar(chain, turn)


# ---------------------------------------------------------------------------
# Source cards display
# ---------------------------------------------------------------------------

async def _send_source_cards(search_result: SearchResult, parent_msg: cl.Message):
    """Send expandable source reference cards."""
    elements = []
    for i, chunk in enumerate(search_result.chunks, 1):
        icon = {"text": "📄", "table": "📊", "figure": "🖼️"}.get(chunk.chunk_type, "📄")
        label = (
            f"{icon} Source {i}: {chunk.source} | "
            f"Page {chunk.page} | "
            f"Score: {chunk.rerank_score:.3f}"
        )
        content = (
            f"**File**: `{chunk.source}`\n"
            f"**Page**: {chunk.page}\n"
            f"**Type**: {chunk.chunk_type}\n"
            f"**Rerank Score**: {chunk.rerank_score:.4f}\n"
            f"**RRF Score**: {chunk.rrf_score:.4f}\n\n"
            f"---\n\n"
            f"{chunk.content}"
        )
        elements.append(
            Text(
                name=label,
                content=content,
                display="side",
                language="markdown",
            )
        )

    await cl.Message(
        content=f"📎 **{len(search_result.chunks)} source(s) retrieved** — click to expand:",
        elements=elements,
        author="MedRAG Sources",
        parent_id=parent_msg.id,
    ).send()


# ---------------------------------------------------------------------------
# Sidebar / case summary
# ---------------------------------------------------------------------------

async def _update_sidebar(chain: MedRAGChain, turn: int):
    """Update the patient case sidebar with conversation summary."""
    summary = chain.get_conversation_summary()
    if not summary or summary == "No conversation history yet.":
        return

    sidebar_content = (
        f"### 🩺 Patient Session Summary\n\n"
        f"**Turn**: {turn}\n\n"
        f"**Conversation Summary**:\n\n{summary}"
    )

    await cl.Message(
        content=sidebar_content,
        author="Session Summary",
        type="system_message",
    ).send()


def _is_greeting(text: str) -> bool:
    """Return True if the user message is a greeting."""
    greetings = [
        "السلام عليكم",
        "وعليكم السلام",
        "اهلا",
        "أهلا",
        "مرحبا",
        "مرحبا بك",
        "hello",
        "hi",
    ]
    normalized = text.lower().strip()
    return any(g in normalized for g in greetings)


async def _send_greeting_response() -> None:
    """Send a polite Arabic greeting response."""
    await cl.Message(
        content=(
            "أهلاً بك، أنا المساعد الطبي MedRAG. "
            "كيف يمكنني مساعدتك في استفساراتك الطبية اليوم؟"
        ),
        author="MedRAG",
    ).send()


def _format_sources_summary(search_result: SearchResult) -> str:
    """Format a compact source summary for the retrieval step."""
    if not search_result.chunks:
        return "_No relevant sources found._"

    lines = []
    seen_sources = set()
    for chunk in search_result.chunks:
        key = f"{chunk.source}:p{chunk.page}"
        if key not in seen_sources:
            seen_sources.add(key)
            icon = {"text": "📄", "table": "📊", "figure": "🖼️"}.get(chunk.chunk_type, "📄")
            lines.append(f"- {icon} `{chunk.source}` — Page {chunk.page}")

    return "**Sources**:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# File upload / ingestion handler
# ---------------------------------------------------------------------------

async def _handle_file_upload_wrapper(message: cl.Message):
    """Wrapped to avoid re-registration — handled inside on_message."""
    pass  # Handled by on_message routing


async def _handle_file_upload(message: cl.Message):
    """Handle PDF file uploads and trigger ingestion."""
    pdf_elements = [el for el in message.elements if el.mime == "application/pdf"]

    if not pdf_elements:
        await cl.Message(
            content="⚠️ Please upload PDF files only. Other formats are not supported.",
            author="MedRAG",
        ).send()
        return

    msg = cl.Message(
        content=f"⏳ Ingesting {len(pdf_elements)} PDF(s)… This may take a moment.",
        author="MedRAG",
    )
    await msg.send()

    from src.storage.qdrant_db import IngestionPipeline

    pipeline = IngestionPipeline()
    results = []

    for el in pdf_elements:
        try:
            # Save temp file
            suffix = Path(el.name).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(el.content)
                tmp_path = tmp.name

            result = pipeline.run([tmp_path])
            os.unlink(tmp_path)
            results.append((el.name, result))

        except Exception as exc:
            logger.error(f"Ingestion failed for {el.name}: {exc}")
            results.append((el.name, {"status": "error", "error": str(exc)}))

    # Report results
    report_lines = [f"## 📥 Ingestion Report\n"]
    for name, result in results:
        if result.get("status") == "success":
            report_lines.append(
                f"✅ **{name}**: {result['chunks_generated']} chunks → "
                f"{result['upserted']} vectors stored"
            )
        else:
            report_lines.append(f"❌ **{name}**: {result.get('error', 'Unknown error')}")

    await cl.Message(content="\n".join(report_lines), author="MedRAG").send()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def _handle_reset():
    """Reset the patient session."""
    chain = cl.user_session.get("chain")
    if chain:
        chain.reset_session()
    cl.user_session.set("turn_count", 0)
    cl.user_session.set("sources_history", [])

    await cl.Message(
        content=(
            "🔄 **Session Reset**\n\n"
            "Patient session cleared. Conversation history erased.\n\n"
            "Ready for a new consultation."
        ),
        author="MedRAG",
    ).send()


async def _handle_status():
    """Show knowledge base status."""
    try:
        from src.storage.qdrant_db import QdrantManager

        db = QdrantManager()
        if db.collection_exists():
            info = db.collection_info()
            status_text = (
                f"## 📊 Knowledge Base Status\n\n"
                f"| Field | Value |\n"
                f"|-------|-------|\n"
                f"| Collection | `{info['name']}` |\n"
                f"| Documents | **{info['points_count']:,}** vectors |\n"
                f"| Status | {info['status']} |\n"
                f"| Provider | {settings.llm_provider.value} / {settings.llm_model} |\n"
                f"| Embedding | `{settings.embedding_model}` |\n"
            )
        else:
            status_text = (
                "⚠️ **Knowledge Base Empty**\n\n"
                "No documents have been ingested yet. "
                "Upload PDFs using the 📎 attachment button."
            )
    except Exception as exc:
        status_text = f"❌ Could not connect to Qdrant: {exc}"

    await cl.Message(content=status_text, author="MedRAG").send()
