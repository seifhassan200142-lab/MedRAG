**MedRAG — الدليل المرجعي الشامل (Arabic)
=================================================

**مقدمة سريعة**: هذا الدليل يشرح مشروع MedRAG خطوة بخطوة — من ingestion إلى retrieval و generation — ليجهّزك للمناقشة التقنية في مقابلات العمل.

**الفهرس**
- **الفصل الأول: System Architecture**
- **الفصل الثاني: Project Structure & Tech Stack**
- **الفصل الثالث: Code Deep Dive**
- **الفصل الرابع: Interview & Industry Focus**


**الفصل الأول: System Architecture**

- **ما هي المشكلة التي يحلها هذا المشروع؟**
  - MedRAG هدفه بناء نظام Retrieval-Augmented Generation (RAG) موجه للوثائق السريرية: يمكنه الإجابة على أسئلة طبية مستندة إلى مستندات PDF طبية مفهرسة، مع قيود سلامة سريرية (no-hallucination).
  - يحلّ مشكلة: البحث السريع عبر كم كبير من المنشورات/الكتب السريرية، استخراج مقاطع سياقية دقيقة، وتقديم إجابات مقتبسة مع رقابة على الاختلاق (hallucination).

- **شرح مبسط وعميق للـ Pipeline**
  - مدخلات: PDF(s) → (Docling OCR + structural parsing) → Parse
  - Phase 1 — Ingestion (Parsing → Chunking → Embedding → Store):
    - Parsing: [src/ingestion/parser.py](src/ingestion/parser.py) يقرأ صفحات PDF، يستخرج نصوص، جداول، وأشكال (figures). يدعم OCR ويدعم توليد caption للصور باستخدام VLM عبر Gemini أو Ollama.
    - Chunking: [src/ingestion/chunker.py](src/ingestion/chunker.py) يستخدم "HybridChunker":
      - Pass 1: SemanticChunker (بناء على embeddings) لتجميع جمل موضوعياً.
      - Pass 2: RecursiveCharacterTextSplitter كحد أمان لحجم الشظايا.
      -يحافظ على metadata لكل chunk مثل `source`, `filename`, `page`, `chunk_type`, `chunk_index`.
    - Embedding: [src/embeddings/embedder.py](src/embeddings/embedder.py) يُنتج متجهات **dense (BGE-M3 1024d)** و **sparse (BM25 via FastEmbed)**.
    - Storage: [src/storage/qdrant_db.py](src/storage/qdrant_db.py) ينشئ collection ذات named vectors (`dense`, `sparse`) ويقوم بـ upsert دفعات نقاط (points) مع payload metadata.
  - Phase 2 — Query Augmentation (قُبل الاستعلام):
    - [src/retrieval/expansion.py] (ملف التحويل/التوسيع موجود في المشروع) يوسّع الاستعلام: استخرج مصطلحات سريرية، HyDE passage، واجراء توليد استعلامات بحث متعددة.
  - Phase 3 — Retrieval & Rerank:
    - Hybrid search: [src/retrieval/searcher.py](src/retrieval/searcher.py) يجري بحثًا متوازياً على الـ dense و الـ sparse vectors (بـ top_k مُختلف).
    - RRF (Reciprocal Rank Fusion): يُدمج قوائم الترتيب من المصادر المتعددة لرفع الاستقرار.
    - Cross-encoder Reranking: BGE Reranker (FlagReranker) يُعيد ترتيب المرشحين بدقّة أعلى.
  - Phase 4 — Generation (RAG):
    - [src/generation/llm_chain.py](src/generation/llm_chain.py) يبني prompt نظامي (medical system prompt) ويستخدم LLM عبر factory في [src/config.py](src/config.py) لتوليد الرد شهرًا-ب-شهر.
    - الذاكرة: ConversationSummaryBufferMemory للحفاظ على ملخّص حواري وتقليل tokens.
    - Streaming: واجهة Chainlit في [app.py](app.py) تبث الإجابة token-by-token.

- **رسم بياني نصي (Text-based Diagram)**

  User → [Chainlit UI: app.py] → Query
                 │
                 ├─ If upload PDF → IngestionPipeline (src/storage/qdrant_db.py)
                 │     ├─ Parser (src/ingestion/parser.py) — Docling + VLM captions
                 │     ├─ Chunker (src/ingestion/chunker.py) — Semantic + Recursive
                 │     ├─ Embedder (src/embeddings/embedder.py) — dense + sparse
                 │     └─ QdrantManager (src/storage/qdrant_db.py) — store vectors
                 │
                 └─ Main Q&A flow:
                       ├─ QueryAugmentor (src/retrieval/expansion.py) → produces search_queries + HyDE passage
                       ├─ HybridSearcher (src/retrieval/searcher.py):
                       │     ├─ dense search (BGE embeddings)
                       │     ├─ sparse search (BM25 sparse vectors)
                       │     ├─ RRF fusion → candidate list
                       │     └─ Cross-encoder reranker → Top-K chunks
                       ├─ MedRAGChain (src/generation/llm_chain.py) → LLM generation (system prompt + context + memory)
                       └─ Chainlit streams response + source cards (app.py)


**الفصل الثاني: Project Structure & Tech Stack**

- **بنية المشروع (الملفات الأساسية)**
  - **`app.py`**: واجهة Chainlit للتفاعل المباشر — رسائل، تحميل ملفات، تحديث الشريط الجانبي، تشغيل pipeline. لو حُذف: تفقد واجهة المستخدم التّفاعلية، لكن يمكن تشغيل النظام عبر CLI (ingest.py) و API.
  - **`ingest.py`**: سكربت CLI لاستهلال ingestion على دفعات (dir/files) ويتيح خيار recreate collection. مفيد للتشغيل الآلي.*
  - **`src/config.py`**: مركز الإعدادات — يقرأ `.env`، يُنشئ factories لـ LLM و Qdrant (synchronous + async)، ويضبط logging. حاسم لمرونة تغيير المزود (Groq / Ollama).
  - **`src/ingestion/parser.py`**: Docling-based PDF parser + VLM captioning للصور.
  - **`src/ingestion/chunker.py`**: HybridChunker يطبّق SemanticChunker ثم RecursiveCharacterTextSplitter.
  - **`src/embeddings/embedder.py`**: EmbedderService (dense BGE-M3 + sparse FastEmbed BM25) و APIs للمزيد.
  - **`src/storage/qdrant_db.py`**: QdrantManager و IngestionPipeline: إدارة collection، upsert، وإجراء تخزين vectors.
  - **`src/retrieval/searcher.py`**: HybridSearcher + CrossEncoderReranker و SearchResult dataflow.
  - **`src/generation/llm_chain.py`**: MedRAGChain: system prompt، memory manager، streaming generation، و LangGraph orchestration.
  - ملفات إضافية: `eval_medrag.py`, `test_pipeline.py`, و `requirements.txt`.

- **لماذا هذه التقنيات؟ (تبرير اختيارات الـ Tech Stack)**
  - **Qdrant**: تصميم hybrid vectors (named vectors) يدعم كلًّا dense + sparse ضمن نفس collection → مناسب للـ hybrid retrieval. Qdrant يوفر أداء عالي، واجهة سهلة، وميزات payload indexing للـ metadata filters.
  - **BAAI/bge-m3 (Embeddings)**: 1024-d dense, multilingual — مهم لأن المشروع يحتاج فهم عربي/إنجليزي سريري في نفس الفضاء. دقّة عالية في semantic retrieval.
  - **FastEmbed (BM25 sparse)**: خيار فعال لإضافة استعانة بالكلمات المفتاحية (lexical matching) خصوصًا للمصطلحات الطبية الدقيقة.
  - **Docling**: مكتبة قوية لاستخراج بنية PDF (نصوص، جداول، أشكال) مع OCR، مهم للمصادر القديمة أو الممسوحة ضوئياً.
  - **VLM (Gemini / Ollama)**: لاستخراج وصف طبي للصور (figures) — يرفع دقة السياق.
  - **LangChain / LangGraph / LangGraph StateGraph**: نمذجة السلاسل القابلة للتركيب (prompt pipelines)، وإدارة الذاكرة و streaming.
  - **Chainlit**: واجهة UI سريعة للبروتوتايب مع دعم البثّ token-by-token والعناصر الغنية (source cards).

- **فوائد هذه التقنيات في Production**
  - **قابلية التبديل (Provider-agnostic)** عبر factories في `src/config.py`: يمكن تبديل LLM/ VLM بدون تغيير المنطق.
  - **الهجين (dense+sparse)** يوفّر أفضل توازن بين recall للكلمات المفتاحية والدلالة السياقية.
  - **التجزئة الذكية** (semantic + safety split) تمنع فقدان السياق وتقلل من احتمال تجاوز حدود التوكن.
  - **البنية الآمنة**: system prompts صارمة وقواعد سريرية تقلّل الهلاوس (hallucinations).


**الفصل الثالث: Code Deep Dive**

ملاحظة: سأشرح كل ملف أساسي "قطعة بقطعة" مع منطق التنفيذ.

- **`src/ingestion/parser.py` — المنطق**
  - الهدف: تحويل PDF → مجموعة `Document` objects مع metadata مفيدة.
  - مكوّنات رئيسية:
    - `ParsedDocument` dataclass: حامل مؤقت للنواتج (documents, figure_count, page_count).
    - `VLMCaptioner.caption(image)`: يستدعِي Gemini أو Ollama لتوليد caption طبي للصورة. تم تصميمه مع retry عبر `tenacity` لزيادة الصلابة.
    - `MedicalDocumentParser.parse(path)`: يستدعي Docling converter (مُعدّ لعمل OCR وtable detection). بعدها:
      - `_extract_text_blocks`: يمر عبر عناصر docling، يستخرج نصوص وجداول، يتجاهل العناصر الصغيرة جداً.
      - `_extract_figures`: يرسم صفحات PDF (pypdfium2)، يقتطع bounding box لكل figure، يُرسله للـ VLMCaptioner، ويخزّن وصف الصورة كـ Document من نوع `figure`.
  - لماذا هكذا؟
    - استخدام Docling يعطي بنيةً أكثر دقة من مجرد تهيئة نص خام (يحترم الجداول و bounding boxes).
    - فصل captions للصور يزود النظام بسياق بصري مهم قد لا يظهر في النص.

- **`src/ingestion/chunker.py` — المنطق**
  - هدفه ضمان "وحدات سياقية" ذات طول مناسب وموضوعية.
  - Pass 1 (SemanticChunker): يجمع نصوصًا متقاربة دلالياً عبر embeddings لتجميع فقرات مرتبطة.
  - Pass 2 (RecursiveCharacterTextSplitter): يفرض حد length (default 1000 chars) ويقسم إذا لزم.
  - التعامل مع non-text (tables/figures): يعبر دون تقسيم لضمان سلامة المعلومات.
  - لماذا هكذا؟
    - Semantic grouping يحافظ على السرد السريري بدل تقسيمه عشوائياً؛ recursive pass هو safety boundary لمنع تجاوز توكن.

- **`src/embeddings/embedder.py` — المنطق**
  - يوفر singleton لـ HuggingFaceEmbeddings (BGE-M3) و FastEmbed sparse encoder.
  - `EmbedderService.embed_documents(documents)`: يعيد (dense_vectors, sparse_vectors).
  - يحول مخرجات FastEmbed إلى `qdrant_client.models.SparseVector` لرفعها إلى Qdrant.
  - لماذا dense+sparse؟
    - dense → فهم دلالي عميق.
    - sparse → مطابقات لفظية حساسة (مهم في المصطلحات الطبية).

- **`src/storage/qdrant_db.py` — المنطق**
  - `QdrantManager.create_collection()`: يعرّف named vectors `dense` و `sparse`، وضبط `Distance.COSINE` و payload indexes.
  - `upsert_documents(...)`: يحوّل Documents+vectors → `PointStruct` مع payload يحتوي `page_content` + metadata.
  - `IngestionPipeline.run(pdf_paths)`: يتكامل parser -> chunker -> embedder -> upsert.
  - مقاومة الفشل: retry على upsert و batch upsert بحجم `UPSERT_BATCH_SIZE`.
  - لماذا Qdrant هنا؟
    - يدعم named vectors و sparse vectors ضمن نفس collection، وتمكين payload filtering.

- **`src/retrieval/searcher.py` — المنطق**
  - `HybridSearcher.search(augmented_query)`: يجري على كل `augmented_query.search_queries` بحثاً هجيناً (dense + sparse)
  - `_hybrid_search_single`: 
    - يحسب dense_vec و sparse_vec بشكل متزامن (run_in_executor)
    - يجري `_dense_search` و `_sparse_search` عبر العميل المتزامن (client.query_points)
  - RRF fusion (`_rrf_fuse`): يجمع الدرجات من القوائم المتعددة باستخدام معادلة RRF (1/(k+rank)) ويرتّب.
  - Cross-encoder (`CrossEncoderReranker`): يُعيد ترتيب القائمة النهائية بدقّة عبر نموذج reranker.
  - مخرجات `SearchResult` تحوي `chunks` مع حقول `rrf_score` و `rerank_score` و `context_text` المُنسق لاستهلاك LLM.

- **`src/generation/llm_chain.py` — المنطق**
  - `MEDICAL_SYSTEM_PROMPT`: قواعد صارمة تمنع fabrication (dosages، lab values، إلخ) وتفرض الاقتباس من المصادر.
  - `ConversationMemoryManager`: يستخدم `ConversationSummaryBufferMemory` لتلخيص الرسائل القديمة وإبقاء أحدث اللفظيّات.
  - `MedRAGChain.astream`: يولّد سياق (context = concatenated retrieved chunks) + history/summary ثم يمرِّره إلى الـ LLM عبر prompt template، ويبثّ tokens إلى الواجهة.
  - LangGraph: يهيكل nodes لتمكين فحص/تعديل التدفق retrieval→generate.

- **`app.py` — منطق التشغيل UI**
  - يعرّف handlers لـ `/reset`, `/status`, file upload و on_message pipeline.
  - يهيّئ singletons per-session (`MedRAGChain`, `HybridSearcher`, `QueryAugmentor`) داخل `cl.user_session`.
  - يعرض source cards مع metadata مفصلة بعد الإجابة.


**الفصل الرابع: Interview & Industry Focus**

- **أسئلة متوقعة من Interviewer و إجابات موجزة ومهنية**
  1. سؤال: "لماذا اخترت استخدام dense + sparse hybrid retrieval؟"
     - إجابة: **Hybrid** يجمع بين فهم دلالي عالي (dense) وحساسية لفظية دقيقة (sparse/BM25). في المجال الطبي المصطلحات الدقيقة قد تكون حاسمة؛ hybrid يزيد recall ويقلل false negatives.
  2. سؤال: "كيف تمنع النظام من اختلاق (hallucination) معلومات طبية؟"
     - إجابة: طبقتُ قواعد صارمة في `MEDICAL_SYSTEM_PROMPT`، أُجبر الـ LLM على الإجابة فقط من `context` المسترجع، أُجري citation دائمًا، وأستخدم memory buffer لتتبع مصادر الإجابات.
  3. سؤال: "متى تستخدم reranker cross-encoder؟ وما تكلفته؟"
     - إجابة: أستخدمه بعد RRF على مرشّحين محددين (top-n) لأن cross-encoder دقته أعلى لكنه مكلف حسابيًا (O(n))؛ لذلك أطبق RRF لتقليل n أولًا.
  4. سؤال: "كيف تدير تحديثات المعرفة (new docs) بدون downtime؟"
     - إجابة: أستخدم عملية ingestion تُحدِث collection أو تـ upsert نقاط جديدة. للإنتاج سأستخدم استراتيجيات blue/green: ingest إلى collection جديدة أو partition ثم تبديل alias بدون توقف.

- **نقاط القوة التي يجب إبرازها**
  - **Provider-agnostic design** عبر factories (`get_llm_client`) يسمح بالتبديل السريع (cloud vs local) لأغراض خصوصية/امتثال.
  - **Hybrid index** (dense+sparse) مصمّم لتحسين الدقّة في المجالات ذات المصطلحات الحساسة.
  - **Semantic chunking** يقلل فقدان السياق ويعطي إجابات أكثر تلاؤمًا.
  - **Safety-first system prompt** و memory summary يحدان من الهلاوس ويُحسنان الاستمرارية الحوارية.

- **التحديات (Bottlenecks) و حلولها عند التوسعة (Scaling)**
  1. Bottleneck: **Embedding throughput** (BGE-M3 model heavy / GPU demand)
     - حلول:
       - Use batching + async workers, run embedding service on GPU nodes, scale horizontally with a microservice embedding API (e.g., Triton / TorchServe / Hugging Face inference endpoint).
       - Cache embeddings for identical chunks.
  2. Bottleneck: **Reranker compute cost** (cross-encoder expensive)
     - حلول:
       - Apply RRF/approx filters to reduce candidate set before reranking.
       - Use distilled rerankers or cascade models (fast lightweight re-ranker then heavy model for top-3).
       - Offload reranking to GPU workers with batching.
  3. Bottleneck: **Qdrant memory & indexing** for very large corpora
     - حلول:
       - Use Qdrant cloud or sharded deployments, configure on-disk indices, tune HNSW parameters, maintain warm caches.
       - Partition by domain/time and search relevant partitions first.
  4. Bottleneck: **Latency for streaming generation**
     - حلول:
       - Use smaller streaming-optimized LLM for low-latency responses and fallback to larger model for long/complex queries.
       - Precompute HyDE passages and relevant embeddings for frequent queries.
  5. Bottleneck: **Ingestion concurrency & consistency**
     - حلول:
       - Use ingest queue (e.g., Celery / Kafka) and transactional writes to vector DB.
       - Maintain document-level versions; support delete-by-source.

- **نصائح للاستعداد للمقابلة**
  - احفظ بنية الملفات الرئيسية والمسوؤليات (app.py, ingest.py, src/config.py, src/ingestion/*, src/embeddings/*, src/storage/*, src/retrieval/*, src/generation/*).
  - كن مستعدًا لرسم مسار البيانات من upload → parser → chunker → embedder → qdrant → searcher → llm.
  - اشرح trade-offs بوضوح: دقة vs. latency، تكلفة reranker vs. recall، GPU budget للموديلات الكبيرة.


**خاتمة سريعة**
- هذا الدليل يختصر الأفكار الرئيسية لبناء واختبار وتشغيل نظام MedRAG. إذا رغبت، أستطيع: 
  - توسيع "الفصل الثالث" ليشمل مقتطفات كود مفصلة لكل دالة (line-by-line explanations).
  - توليد نسخة PDF مُنسقة من هذا الملف.
  - إعداد ملخص مختصر للـ architecture على صفحة واحدة للطباعة.


---
*انتهى الجزء الأول من الدليل. أخبرني أي فصل تريد توسيع شرحه (مثلاً: تفصيل كل دالة في `src/retrieval/searcher.py` خطوة بخطوة) وسأتابع التنفيذ.*
