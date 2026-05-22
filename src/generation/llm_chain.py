"""
MedRAG — src/generation/llm_chain.py
=======================================
Phase 3 — Generation Chain

Architecture:
  • System prompt enforces strict anti-hallucination constraints.
  • ConversationSummaryBufferMemory keeps recent turns verbatim while
    summarizing older turns — prevents token explosion in long sessions.
  • LangGraph StateGraph for robust, inspectable state management.
  • Streaming-first: yields tokens incrementally via astream().

Clinical Safety Rules (encoded in system prompt):
  1. Never invent drug dosages, lab values, or diagnostic criteria.
  2. If answer not in context → say "I don't have enough information."
  3. Always recommend consulting a physician for clinical decisions.
  4. Distinguish between established guidelines and emerging evidence.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from src.config import get_llm_client, get_settings
from src.retrieval.searcher import SearchResult


# ---------------------------------------------------------------------------
# System Prompt — Anti-Hallucination Medical Assistant
# ---------------------------------------------------------------------------

MEDICAL_SYSTEM_PROMPT = """\
You are MedRAG, a clinical AI assistant designed to answer medical questions \
based strictly on the provided clinical context (retrieved from verified medical \
documents).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL SAFETY RULES — NON-NEGOTIABLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CONTEXT-ONLY ANSWERS: Answer ONLY from the provided context. If the answer \
   is not in the context, say: "I don't have sufficient information in my \
   knowledge base to answer this question. Please consult a healthcare professional."

2. NO FABRICATED DOSAGES: NEVER invent specific drug dosages, concentrations, \
   frequencies, or routes of administration. If dosage is needed and not in \
   context, state it is not available.

3. NO FABRICATED LAB VALUES: Do not generate reference ranges, diagnostic \
   thresholds, or lab interpretations unless explicitly in the context.

4. ALWAYS RECOMMEND PROFESSIONAL CONSULTATION: For any clinical decision, \
   diagnosis, or treatment plan, always append: \
   "⚠️ This information is for educational purposes. Consult a licensed \
   healthcare professional for medical decisions."

5. CITE SOURCES: When answering, reference the source documents \
   (e.g., "According to [Filename, Page X]...").

6. UNCERTAINTY: Express uncertainty clearly using language like "The context \
   suggests...", "Based on available evidence...", or "This may vary by...".

7. LANGUAGE: Respond in the same language as the user's question. \
   If the question is in Arabic, respond in formal Arabic (MSA). \
   If in English, respond in English. Never mix languages unexpectedly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The context below contains excerpts from verified medical documents. \
Each excerpt is labeled with its source, page, and type.

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the user is greeting the assistant, reply immediately with a polite Arabic welcome message such as:
"أهلاً بك، أنا المساعد الطبي MedRAG. كيف يمكنني مساعدتك في استفساراتك الطبية اليوم؟"
For medical queries, maintain a professional clinical persona and answer ONLY from the retrieved document context.
Respond thoughtfully, accurately, and compassionately.
"""


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class MedRAGState(TypedDict):
    """State flowing through the LangGraph pipeline."""
    messages: list[Any]          # Full conversation history (add_messages reducer)
    context: str                 # Retrieved context from Phase 3
    search_result: SearchResult | None  # Full retrieval result for UI display
    response: str                # Final generated response
    memory_summary: str          # Running conversation summary


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------

class ConversationMemoryManager:
    """
    Wraps ConversationSummaryBufferMemory for efficient long-context management.

    Strategy:
      - Last N tokens of conversation kept verbatim (for fresh context).
      - Older messages summarized progressively.
      - Summary is injected into system prompt for continuity.
    """

    def __init__(self) -> None:
        s = get_settings()
        # Use a small fast model for summarization (can reuse main LLM)
        summary_llm = get_llm_client(temperature=0.0, max_tokens=512)

        self.memory = ConversationSummaryBufferMemory(
            llm=summary_llm,
            max_token_limit=s.memory_max_token_limit,
            memory_key="chat_history",
            return_messages=True,
            human_prefix="Patient",
            ai_prefix="MedRAG",
        )
        logger.info(
            f"ConversationSummaryBufferMemory initialized "
            f"(max_token_limit={s.memory_max_token_limit})"
        )

    def add_turn(self, human_input: str, ai_output: str) -> None:
        """Add a completed turn to memory."""
        self.memory.save_context(
            {"input": human_input},
            {"output": ai_output},
        )

    def get_history(self) -> list[Any]:
        """Return LangChain message objects from memory."""
        return self.memory.chat_memory.messages

    def get_summary(self) -> str:
        """Return the running summary of older conversation."""
        return self.memory.moving_summary_buffer or ""

    def clear(self) -> None:
        """Reset memory for a new patient session."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")


# ---------------------------------------------------------------------------
# MedRAG Generation Chain
# ---------------------------------------------------------------------------

class MedRAGChain:
    """
    The full generation chain: context + history → streamed response.

    Integrates:
      • Provider-agnostic LLM (Groq / Ollama via config)
      • Strict medical system prompt
      • ConversationSummaryBufferMemory
      • LangGraph StateGraph for explicit state management

    Usage:
        chain = MedRAGChain()
        # Async streaming
        async for token in chain.astream(query, search_result):
            print(token, end="", flush=True)
        # Sync
        response = chain.invoke(query, search_result)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = get_llm_client(streaming=True)
        self.memory = ConversationMemoryManager()

        # Build the generation prompt
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # Non-streaming LLM for memory summarization
        self._sync_llm = get_llm_client(streaming=False, max_tokens=1024)

        logger.info("MedRAGChain ready.")

    # ------------------------------------------------------------------
    # Async streaming (primary UI path)
    # ------------------------------------------------------------------

    async def astream(
        self,
        question: str,
        search_result: SearchResult,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the LLM response token by token.

        Args:
            question: The user's question.
            search_result: Retrieved context from HybridSearcher.

        Yields:
            String tokens as they arrive from the LLM.
        """
        context = search_result.context_text if search_result.chunks else (
            "No relevant medical information was found in the knowledge base "
            "for this query."
        )

        history = self.memory.get_history()
        summary = self.memory.get_summary()

        # Prepend summary as a system message if it exists
        if summary:
            history = [SystemMessage(content=f"[Conversation Summary]\n{summary}")] + history

        chain = self._prompt | self.llm | StrOutputParser()

        full_response = ""
        async for token in chain.astream({
            "context": context,
            "chat_history": history,
            "question": question,
        }):
            full_response += token
            yield token

        # Save to memory after streaming completes
        self.memory.add_turn(question, full_response)
        logger.debug(f"Turn added to memory. Summary: {self.memory.get_summary()[:100]}")

    async def ainvoke(
        self,
        question: str,
        search_result: SearchResult,
    ) -> str:
        """Async, non-streaming-friendly response generation."""
        try:
            tokens: list[str] = []
            async for token in self.astream(question, search_result):
                tokens.append(token)
            return "".join(tokens)
        except AttributeError as exc:
            logger.warning(
                "Streaming not available for this model; using async fallback: %s",
                exc,
            )

        context = search_result.context_text if search_result.chunks else (
            "No relevant medical information was found in the knowledge base "
            "for this query."
        )
        history = self.memory.get_history()
        summary = self.memory.get_summary()

        if summary:
            history = [SystemMessage(content=f"[Conversation Summary]\n{summary}")] + history

        response = await (
            self._prompt | self.llm | StrOutputParser()
        ).ainvoke({
            "context": context,
            "chat_history": history,
            "question": question,
        })

        response_text = str(response)
        self.memory.add_turn(question, response_text)
        return response_text

    # ------------------------------------------------------------------
    # Sync invoke (for testing / batch processing)
    # ------------------------------------------------------------------

    def invoke(
        self,
        question: str,
        search_result: SearchResult,
    ) -> str:
        """Synchronous invocation — returns full response string."""
        async def _run():
            tokens = []
            async for token in self.astream(question, search_result):
                tokens.append(token)
            return "".join(tokens)

        return asyncio.run(_run())

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Clear conversation history for a new patient session."""
        self.memory.clear()

    def get_conversation_summary(self) -> str:
        """Return a human-readable summary of the current conversation."""
        return self.memory.get_summary() or "No conversation history yet."


# ---------------------------------------------------------------------------
# LangGraph Pipeline (full end-to-end graph)
# ---------------------------------------------------------------------------

class MedRAGGraph:
    """
    LangGraph StateGraph wrapping the full retrieval + generation pipeline.

    Nodes:
      augment  → Phase 2: Query Expansion + HyDE
      retrieve → Phase 3: Hybrid Search + RRF + Reranking
      generate → Phase 3: LLM Generation with memory

    This explicit graph enables:
      • Easy inspection of intermediate states
      • Adding conditional branches (e.g., route Arabic vs English)
      • Future: Human-in-the-loop nodes for clinical review
    """

    def __init__(self) -> None:
        from src.retrieval.expansion import QueryAugmentor
        from src.retrieval.searcher import HybridSearcher

        self.augmentor = QueryAugmentor()
        self.searcher = HybridSearcher()
        self.chain = MedRAGChain()

        self._graph = self._build_graph()
        logger.info("MedRAGGraph compiled.")

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph StateGraph."""
        workflow = StateGraph(MedRAGState)

        # Add nodes
        workflow.add_node("augment", self._augment_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        # Define flow
        workflow.set_entry_point("augment")
        workflow.add_edge("augment", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    async def _augment_node(self, state: MedRAGState) -> dict:
        """Phase 2: Augment the latest user query."""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        augmented = await self.augmentor.augment(last_human)
        return {"messages": state["messages"]}  # State passthrough; augmented stored in searcher

    async def _retrieve_node(self, state: MedRAGState) -> dict:
        """Phase 3a: Retrieve context via hybrid search."""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        augmented = await self.augmentor.augment(last_human)
        search_result = await self.searcher.search(augmented)
        return {
            "context": search_result.context_text,
            "search_result": search_result,
        }

    async def _generate_node(self, state: MedRAGState) -> dict:
        """Phase 3b: Generate response from context + history."""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        search_result = state.get("search_result")
        response = await self.chain.ainvoke(last_human, search_result)
        initial_state: MedRAGState = {
            "messages": [HumanMessage(content=question)],
            "context": "",
            "search_result": None,
            "response": "",
            "memory_summary": "",
        }
        final_state = await self._graph.ainvoke(initial_state)
        return final_state
