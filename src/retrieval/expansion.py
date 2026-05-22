"""
MedRAG — src/retrieval/expansion.py
======================================
Phase 2 — Query Augmentation (The Bridging Layer)

Two augmentation strategies run in parallel:

1. Query Expansion / Rewriting
   Egyptian Arabic dialect patient complaint
     → Precise English clinical terminology
   
   Examples:
     "بطني بتوجعني وعايز أرجع" 
       → "Acute abdominal pain, nausea, vomiting"
     "راسي بتدور وحاسس إني هسقط"
       → "Vertigo, presyncope, dizziness"

2. HyDE — Hypothetical Document Embeddings
   Instead of embedding the (often vague) raw query, we ask the LLM to
   generate a synthetic clinical passage that WOULD answer the query,
   then embed THAT. This dramatically improves semantic recall because
   the embedding is compared against actual clinical text, not a
   patient colloquialism.

Both results are returned and used for the hybrid search in Phase 3.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_llm_client


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AugmentedQuery:
    """
    Holds all augmented forms of the original query.

    Attributes:
        original:     Raw user input (may be Arabic dialect or broken English).
        expanded:     Clinically precise English term(s).
        hyde_passage: Synthetic clinical document for HyDE embedding.
        search_queries: Final list of strings to embed for retrieval.
    """
    original: str
    expanded: str
    hyde_passage: str

    @property
    def search_queries(self) -> list[str]:
        """Return all query forms to use for embedding."""
        queries = [self.expanded]
        if self.hyde_passage and self.hyde_passage != self.expanded:
            queries.append(self.hyde_passage)
        return queries


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

EXPANSION_SYSTEM_PROMPT = """\
You are a bilingual clinical NLP assistant specializing in Arabic-English \
medical translation.

Your task: Convert a patient's complaint (which may be in Egyptian Arabic \
dialect, Modern Standard Arabic, English, or a mix) into precise, formal \
clinical English terminology suitable for medical literature retrieval.

Rules:
- Output ONLY the clinical English translation. No explanation.
- Use standard medical terminology (ICD/SNOMED style).
- Include synonyms if appropriate, separated by commas.
- If the query is already clinical English, clean it up and return it.
- Maximum 30 words in output.

Examples:
  Input: "بطني بتوجعني وعايز أرجع ومش قادر آكل"
  Output: Acute abdominal pain, nausea, vomiting, anorexia

  Input: "راسي بيدور وحاسس إني هسقط لو وقفت"
  Output: Positional vertigo, presyncope, orthostatic hypotension

  Input: "صدري بيوجعني لما بتحرك وبياخد نفس"
  Output: Pleuritic chest pain, dyspnea on exertion
"""

EXPANSION_HUMAN_PROMPT = "Patient complaint: {query}"


HYDE_SYSTEM_PROMPT = """\
You are a medical knowledge assistant. Given a clinical question or patient \
complaint, generate a short hypothetical passage (3-5 sentences) from a \
medical textbook or clinical guideline that would directly answer or \
address the query.

Rules:
- Write as if it is an excerpt from a clinical reference document.
- Use precise medical terminology.
- Include relevant pathophysiology, diagnosis criteria, or treatment if applicable.
- Do NOT say "This is a hypothetical document." Just write the passage directly.
- Keep it factual and grounded. Do not invent specific drug dosages.
- Output only the passage. No preamble.
"""

HYDE_HUMAN_PROMPT = "Clinical query: {query}"


# ---------------------------------------------------------------------------
# QueryAugmentor
# ---------------------------------------------------------------------------

class QueryAugmentor:
    """
    Augments queries through Arabic-to-clinical expansion and HyDE.

    Usage:
        augmentor = QueryAugmentor()
        result = await augmentor.augment("بطني بتوجعني")
        # result.expanded       → "Acute abdominal pain, nausea"
        # result.hyde_passage   → "Abdominal pain is a common presenting..."
        # result.search_queries → [expanded, hyde_passage]
    """

    def __init__(self) -> None:
        # Use a low-temperature LLM for deterministic clinical translations
        self._llm = get_llm_client(temperature=0.1, max_tokens=256)

        self._expansion_chain = (
            ChatPromptTemplate.from_messages([
                ("system", EXPANSION_SYSTEM_PROMPT),
                ("human", EXPANSION_HUMAN_PROMPT),
            ])
            | self._llm
            | StrOutputParser()
        )

        self._hyde_chain = (
            ChatPromptTemplate.from_messages([
                ("system", HYDE_SYSTEM_PROMPT),
                ("human", HYDE_HUMAN_PROMPT),
            ])
            | self._llm
            | StrOutputParser()
        )

        logger.info("QueryAugmentor initialized (expansion + HyDE).")

    # ------------------------------------------------------------------
    # Async API (primary)
    # ------------------------------------------------------------------

    async def augment(self, query: str) -> AugmentedQuery:
        """
        Augment a query asynchronously.

        Expansion and HyDE run concurrently to minimize latency.

        Args:
            query: Raw user query (any language/dialect).

        Returns:
            AugmentedQuery with all augmented forms.
        """
        logger.debug(f"Augmenting query: {query[:80]}…")

        # Run expansion and HyDE in parallel
        expanded_task = asyncio.create_task(self._expand_async(query))
        hyde_task = asyncio.create_task(self._hyde_async(query))

        expanded, hyde_passage = await asyncio.gather(
            expanded_task, hyde_task, return_exceptions=True
        )

        # Handle partial failures gracefully
        if isinstance(expanded, Exception):
            logger.warning(f"Query expansion failed: {expanded}. Using original.")
            expanded = query

        if isinstance(hyde_passage, Exception):
            logger.warning(f"HyDE generation failed: {hyde_passage}. Using expanded.")
            hyde_passage = str(expanded)

        result = AugmentedQuery(
            original=query,
            expanded=str(expanded).strip(),
            hyde_passage=str(hyde_passage).strip(),
        )

        logger.info(
            f"Query augmented:\n"
            f"  Original : {result.original[:80]}\n"
            f"  Expanded : {result.expanded[:80]}\n"
            f"  HyDE     : {result.hyde_passage[:120]}…"
        )
        return result

    def augment_sync(self, query: str) -> AugmentedQuery:
        """Synchronous wrapper for augment()."""
        return asyncio.run(self.augment(query))

    # ------------------------------------------------------------------
    # Internal chain runners
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
    async def _expand_async(self, query: str) -> str:
        """Run query expansion chain."""
        result = await self._expansion_chain.ainvoke({"query": query})
        return result.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
    async def _hyde_async(self, query: str) -> str:
        """Run HyDE generation chain."""
        result = await self._hyde_chain.ainvoke({"query": query})
        return result.strip()

    # ------------------------------------------------------------------
    # Utility: detect if query is Arabic (for logging/routing)
    # ------------------------------------------------------------------

    @staticmethod
    def is_arabic(text: str) -> bool:
        """Return True if the text contains Arabic characters."""
        return any("\u0600" <= ch <= "\u06FF" for ch in text)
