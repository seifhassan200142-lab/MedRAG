"""
MedRAG — src/config.py
======================
Central configuration hub. Reads .env, validates settings, and exposes
typed factories for every external client (LLM, Qdrant, Embeddings).

Design decisions
----------------
* Pydantic-Settings v2 for strict typing + auto-casting.
* LLM is resolved through get_llm_client() — a pure factory that checks
  LLM_PROVIDER and returns the correct LangChain BaseChatModel, making
  every downstream module completely provider-agnostic.
* All clients are singletons (via @lru_cache) to avoid repeated cold-starts
  across the module graph.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient, QdrantClient

load_dotenv()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"


class VLMProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class MedRAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:9b"

    # API Keys
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    # VLM
    vlm_provider: VLMProvider = VLMProvider.GEMINI
    vlm_ollama_model: str = "llama3.2-vision"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "medrag_v1"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_top_k: int = 4

    # Retrieval
    hybrid_dense_top_k: int = 20
    hybrid_sparse_top_k: int = 20
    final_top_k: int = 4

    # Generation
    max_tokens: int = 2048
    temperature: float = 0.1
    memory_max_token_limit: int = 3000

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _groq_key_from_env(cls, v: str) -> str:
        return v or os.getenv("GROQ_API_KEY", "")

    @field_validator("google_api_key", mode="before")
    @classmethod
    def _google_key_from_env(cls, v: str) -> str:
        return v or os.getenv("GOOGLE_API_KEY", "")


@lru_cache(maxsize=1)
def get_settings() -> MedRAGSettings:
    """Return the singleton settings object."""
    settings = MedRAGSettings()
    logger.info(
        f"MedRAG Config loaded — LLM: {settings.llm_provider.value}/{settings.llm_model} | "
        f"VLM: {settings.vlm_provider.value} | Collection: {settings.qdrant_collection}"
    )
    return settings


# ---------------------------------------------------------------------------
# LLM Factory  (THE critical abstraction layer)
# ---------------------------------------------------------------------------

def get_llm_client(
    *,
    streaming: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    LLM factory — returns a LangChain BaseChatModel.

    Provider selection is driven solely by the LLM_PROVIDER env-var:
        groq  → ChatGroq   (cloud, fast inference)
        ollama → ChatOllama (local, private, HIPAA-friendly)

    Switching providers requires ZERO code changes in any other module.
    """
    s = get_settings()
    temp = temperature if temperature is not None else s.temperature
    tokens = max_tokens if max_tokens is not None else s.max_tokens

    if s.llm_provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq

        if not s.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        logger.debug(f"Instantiating ChatGroq — model={s.llm_model}")
        return ChatGroq(
            model=s.llm_model,
            api_key=s.groq_api_key,
            temperature=temp,
            max_tokens=tokens,
            streaming=streaming,
        )

    elif s.llm_provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        logger.debug(
            f"Instantiating ChatOllama — model={s.ollama_model} @ {s.ollama_base_url}"
        )
        return ChatOllama(
            model=s.ollama_model,
            base_url=s.ollama_base_url,
            temperature=temp,
            num_predict=tokens,
        )

    raise ValueError(f"Unknown LLM provider: {s.llm_provider}")


# ---------------------------------------------------------------------------
# Qdrant Client Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Synchronous Qdrant client (used during ingestion)."""
    s = get_settings()
    kwargs: dict = {"url": s.qdrant_url}
    if s.qdrant_api_key:
        kwargs["api_key"] = s.qdrant_api_key
    logger.debug(f"Connecting to Qdrant at {s.qdrant_url}")
    return QdrantClient(**kwargs)


@lru_cache(maxsize=1)
def get_async_qdrant_client() -> AsyncQdrantClient:
    """Async Qdrant client (used at query-time for non-blocking I/O)."""
    s = get_settings()
    kwargs: dict = {"url": s.qdrant_url}
    if s.qdrant_api_key:
        kwargs["api_key"] = s.qdrant_api_key
    return AsyncQdrantClient(**kwargs)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure loguru for the application."""
    s = get_settings()
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=s.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        "logs/medrag.log",
        rotation="50 MB",
        retention="14 days",
        level="DEBUG",
        encoding="utf-8",
    )
