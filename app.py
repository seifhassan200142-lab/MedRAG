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
import re
import tempfile
from datetime import datetime
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

    # 1. حل جذري للمشكلة رقم 3 (الرسالة الترحيبية الصريحة)
    if _is_greeting(query):
        await _send_greeting_response(query)
        return

    augmentor = _get_augmentor()
    searcher = _get_searcher()
    chain = _get_chain()

    turn = cl.user_session.get("turn_count", 0) + 1
    cl.user_session.set("turn_count", turn)

    # ── Step 1: Query augmentation ───────────────────────────────────
    async with cl.Step(name="Analyzing Query", show_input=False) as step:
        augmented = await augmentor.augment(query)
        is_arabic = QueryAugmentor.is_arabic(query)
        lang_badge = "🇪🇬 Arabic → Clinical EN" if is_arabic else "🇬🇧 English"
        step.output = f"**Language detected**: {lang_badge}\n\n**Clinical terms**: `{augmented.expanded}`"

    # ── Step 2: Hybrid retrieval ─────────────────────────────────────
    async with cl.Step(name="📚 Searching Knowledge Base", show_input=False) as step:
        search_result = await searcher.search(augmented)
        step.output = f"**After RRF + Reranking**: {len(search_result.chunks)} chunks selected"

    # If no sources were retrieved, enforce strict out-of-domain handling
    if not search_result.chunks:
        if is_arabic:
            await cl.Message(
                content=(
                    "عذراً، لا تتوفر لدي معلومات كافية في قاعدة البيانات المخصصة للإجابة على هذا السؤال."
                ),
                author="MedRAG",
            ).send()
        else:
            await cl.Message(
                content=(
                    "I don't have sufficient information in my knowledge base to answer this question."
                ),
                author="MedRAG",
            ).send()
        return

    # ── Step 3 & 4: Streaming generation ──────────
    # We hide the old large source boxes and instead convert inline citations
    # into hoverable tooltips after the response is generated.
    response_msg = cl.Message(content="", author="MedRAG")
    await response_msg.send()

    full_response = ""
    try:
        async for token in chain.astream(query, search_result):
            if token:
                await response_msg.stream_token(token)
                full_response += token

        # Post-process inline citations: replace [n] with clickable anchors and
        # render source chunk expanders for each retrieved citation.
        try:
            from html import escape

            escaped_text = escape(full_response)

            def citation_replacer(match: re.Match[str]) -> str:
                idx = int(match.group(1))
                if idx < 1 or idx > len(search_result.chunks):
                    return match.group(0)

                return (
                    f'<a href="#source-{idx}" '
                    f'style="color:#276ef1; text-decoration:none; font-weight:600;" '
                    f'title="عرض المصدر {idx}">[{idx}]</a>'
                )

            processed = re.sub(r"(?<!\w)\[(\d+)\](?!\w)", citation_replacer, escaped_text)

            # If the model prepended greeting text inadvertently, remove it
            if "أهلاً بك" in processed and not _is_greeting(query):
                processed = processed.split("؟")[-1].strip()

            details_html = []
            for i, chunk in enumerate(search_result.chunks, 1):
                source_name = escape(chunk.source)
                snippet = escape(chunk.content.strip())
                details_html.append(
                    (
                        "<details style='margin:12px 0; padding:10px; border:1px solid #dde2ea; "
                        "border-radius:10px; background:#fafbff;'>"
                        f"<summary id='source-{i}' style='font-size:0.96rem; font-weight:700; "
                        "cursor:pointer; outline:none; color:#1f57c8;'>"
                        f"مصدر [{i}] — {source_name} | الصفحة {chunk.page} | ريرنك {chunk.rerank_score:.4f}"
                        "</summary>"
                        f"<div style='margin-top:10px; white-space:pre-wrap; line-height:1.55;'>"
                        f"{snippet}"
                        "</div>"
                        "</details>"
                    )
                )

            if details_html:
                processed += (
                    "<div style='margin-top:16px; padding-top:8px; border-top:1px solid #e5e8ef;'>"
                    "<strong>مصادر مراجع الاستجابة:</strong>" + "".join(details_html) + "</div>"
                )

            await response_msg.update(content=processed)
        except Exception as e:
            logger.exception("Failed to post-process citations: %s", e)
            await response_msg.update(content=full_response)

    except Exception as exc:
        logger.warning("Streaming failed, falling back to full response: %s", exc)
        full_response = await chain.ainvoke(query, search_result)
        await response_msg.update(content=full_response)

    # ── Step 5: Update sidebar summary ──────────────────────────────
    await _update_sidebar(chain, turn)


# ---------------------------------------------------------------------------
# دالة الـ Source Cards القديمة بنخليها فاضية تماماً عشان متبعتش رسائل تانية تبوظ الدنيا
# ---------------------------------------------------------------------------
async def _send_source_cards(search_result, parent_msg):
    pass
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
        "صباح الخير",
        "مساء الخير",
        "ازيك",
        "ايه الاخبار",
        "كيف حالك",
        "مرحبا",
        "مرحبا بك",
        "اهلا",
        "أهلا",
        "hello",
        "hi",
    ]
    normalized = re.sub(r"[؟?!،.]+", " ", text.lower()).strip()
    if len(normalized) > 60:
        return False
    return any(g in normalized for g in greetings)


def _get_local_time_greeting() -> str:
    now = datetime.now()
    if 5 <= now.hour < 12:
        return "صباح النور"
    return "مساء النور"


async def _send_greeting_response(user_text: str) -> None:
    """Send a polite Arabic greeting response based on the current local time."""
    s = get_settings()
    specialty = getattr(s, "qdrant_collection", "المعرفة الطبية المهيئة")

    if "السلام عليكم" in user_text or "وعليكم السلام" in user_text:
        greeting = "وعليكم السلام ورحمة الله"
    else:
        greeting = _get_local_time_greeting()

    await cl.Message(
        content=(
            f"{greeting}، أنا مساعدك الذكي المتخصص في {specialty}. "
            "كيف يمكنني مساعدتك اليوم؟"
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
            report_lines.append(f"❌ **{name}**")
