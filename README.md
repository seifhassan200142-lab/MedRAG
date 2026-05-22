# MedRAG — Clinical AI Assistant

A **production-grade, clinical-grade RAG system** for medical document retrieval and Q&A, supporting Egyptian Arabic and English, with hybrid search, cross-encoder reranking, and hallucination-constrained generation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MedRAG Architecture                             │
├──────────────┬──────────────────┬──────────────────┬────────────────────┤
│  PHASE 1     │   PHASE 2        │   PHASE 3        │   UI LAYER         │
│  INGESTION   │   AUGMENTATION   │   RETRIEVAL +    │                    │
│              │                  │   GENERATION     │                    │
├──────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Docling PDF  │ Query Expansion  │ Hybrid Search    │ Chainlit           │
│ Parser       │ Arabic→Clinical  │ Dense (BGE-M3)   │ • Token streaming  │
│              │ EN               │ Sparse (BM25)    │ • Source cards     │
│ VLM Cropper  │                  │                  │ • Session sidebar  │
│ Gemini /     │ HyDE — Synthetic │ RRF Fusion       │ • PDF upload       │
│ Llama-Vision │ Doc Embedding    │                  │ • Arabic support   │
│              │                  │ Cross-Encoder    │                    │
│ HybridChunker│                  │ bge-reranker     │                    │
│ Semantic +   │                  │ -large           │                    │
│ Recursive    │                  │                  │                    │
│              │                  │ LangGraph Chain  │                    │
│ Qdrant       │                  │ + Memory         │                    │
│ Dense+Sparse │                  │                  │                    │
└──────────────┴──────────────────┴──────────────────┴────────────────────┘
```

## Project Structure

```
MedRAG/
├── src/
│   ├── config.py              # Settings + LLM/DB factory (provider-agnostic)
│   ├── ingestion/
│   │   ├── parser.py          # Docling PDF + VLM figure captioning
│   │   └── chunker.py         # Semantic + Recursive hybrid chunker
│   ├── embeddings/
│   │   └── embedder.py        # BAAI/bge-m3 dense + BM25 sparse
│   ├── storage/
│   │   └── qdrant_db.py       # Qdrant collection + ingestion pipeline
│   ├── retrieval/
│   │   ├── expansion.py       # Arabic→Clinical expansion + HyDE
│   │   └── searcher.py        # Hybrid search + RRF + reranking
│   ├── generation/
│   │   └── llm_chain.py       # LangGraph chain + ConversationSummaryBufferMemory
│   └── app.py                 # Chainlit UI
├── ingest.py                  # CLI ingestion tool
├── requirements.txt
├── .env                       # Configuration
└── chainlit.md                # Welcome page
```

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 2. Configure `.env`

```bash
cp .env .env.local
# Edit .env with your GROQ_API_KEY and GOOGLE_API_KEY
```

### 3. Ingest Medical PDFs

```bash
# Ingest a directory of PDFs
python ingest.py --dir /path/to/medical_pdfs

# Ingest specific files
python ingest.py --files cardiology.pdf nephrology.pdf

# Recreate collection (fresh start)
python ingest.py --recreate --dir /path/to/pdfs
```

### 4. Launch the UI

```bash
chainlit run src/app.py -w --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Switching LLM Provider

To switch from Groq to local Ollama (e.g., Gemma 9B):

```bash
# In .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:9b
OLLAMA_BASE_URL=http://localhost:11434
```

**No code changes required.** All downstream modules consume `get_llm_client()` which resolves the provider at runtime.

---

## Key Design Decisions

### Why Hybrid Search?
Dense vectors excel at semantic similarity ("myocardial infarction" ↔ "heart attack") but miss rare medical abbreviations. Sparse BM25 catches exact terminology ("AST", "eGFR", "NYHA Class II"). RRF blends both rankings optimally.

### Why HyDE?
Patient queries like "بطني بتوجعني" (my stomach hurts) produce poor dense embeddings when compared directly against clinical text. HyDE generates a synthetic clinical answer first, then embeds that — the embedding is compared against similar clinical prose, dramatically improving recall.

### Why ConversationSummaryBufferMemory?
Long medical consultations can exceed LLM context windows. This memory compresses older turns into a summary while keeping recent turns verbatim, maintaining clinical continuity without token overflow.

### Why Cross-Encoder Reranking?
Bi-encoder similarity (dot product) is fast but approximate. The cross-encoder (bge-reranker-large) reads the full (query, passage) pair together, enabling far more accurate relevance scoring as the final gate.

---

## Clinical Safety

MedRAG enforces strict safety constraints at the prompt level:
- Never fabricates drug dosages or lab values
- Refuses to answer if context is insufficient (no hallucination)
- Always appends physician consultation recommendation
- Cites sources with filename and page number
- Responds in the user's language (Arabic or English)

---

## Performance Benchmarks (expected)

| Stage | Latency |
|-------|---------|
| Query Expansion (Groq) | ~300ms |
| HyDE Generation (Groq) | ~500ms |
| Hybrid Search (Qdrant) | ~50ms |
| Cross-Encoder Reranking (CPU, 20 candidates) | ~400ms |
| LLM Generation (Groq, streaming) | ~1-3s TTFT |
| **Total E2E** | **~2-4s to first token** |
