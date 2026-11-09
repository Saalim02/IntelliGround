# IntelliGround

IntelliGround is an AI-powered multilingual documentation assistant. Users upload PDF documents and ask questions about them in text or voice. The application retrieves relevant passages from the user's active document, generates grounded answers, and returns page-level evidence and visual page previews.

Instead of manually reading hundreds of pages, users can request summaries, important topics, suggested questions, page explanations, and contextual follow-ups. The voice agent distinguishes normal conversation from PDF questions.

> This README documents the implementation currently present in the repository. Optional and partially integrated features are identified explicitly.

## Why IntelliGround exists

Long PDFs are difficult to search when information is spread across pages, tables, or scanned images. Keyword search misses meaning, while an unconstrained LLM can produce plausible but unsupported answers. Language differences make document understanding harder, and users need traceable evidence.

IntelliGround addresses these problems with page-aware extraction, OCR fallback, multilingual embeddings, BM25 keyword retrieval, hybrid ranking, evidence citations, authenticated user-scoped storage, multilingual responses, and voice interaction.

## Key features

### PDF upload and processing

- PDF-only validation.
- Maximum configured file size: 200 MB.
- Maximum configured page count: 1,400 pages.
- Native PyMuPDF extraction.
- Tesseract OCR fallback for blank or corrupted pages.
- OCR language selection and extraction-method metadata.
- Adaptive page-aware chunking with overlap.
- Page image generation for evidence previews.
- Upload progress jobs.
- Atomic vector-store and active-document registry updates.
- Per-user active-document isolation.

### Document intelligence

The React client exposes on-demand document actions:

- grounded summaries;
- key topics and important points;
- suggested questions;
- response-language selection.

These operations read authenticated document evidence, validate references against real chunk identifiers, and validate language and citation consistency. They are generated on demand rather than being claimed as automatic upload-time outputs.

### Retrieval and grounded answers

- Dense semantic retrieval with intfloat/multilingual-e5-small.
- FAISS vector storage.
- Persistent BM25 sparse retrieval using rank-bm25, with an internal fallback.
- Hybrid dense and sparse retrieval.
- Reciprocal Rank Fusion.
- Optional multilingual cross-encoder reranking.
- Dynamic candidate sizing based on document size.
- Exact page retrieval for requests such as “Explain page 40.”
- Page, chunk, paragraph, source, snippet, and image metadata where available.
- Grounded fallback when evidence is insufficient.
- Reranking diagnostics in query responses.

### Conversation and voice

The intent layer recognizes greetings, audio checks, casual conversation, acknowledgments, continuation, repetition, clarification, page queries, document overviews, document questions, contextual follow-ups, application help, out-of-scope requests, and ambiguous references.

Non-document messages do not invoke RAG or LLM-as-a-Judge evaluation. Recent user-scoped context is used for follow-ups, while continuation and repetition avoid duplicate retrievals and evaluations.

Voice uses Gnani Prisma speech-to-text, Gnani Timbre text-to-speech, buffered voice turns, a WebSocket streaming route, interruption handling, and the same shared query orchestration used by typed questions.

### Evaluation and trust

Eligible document answers can be evaluated with the existing LLM-as-a-Judge service. It measures retrieval quality, source relevance, grounding, claim support, answer correctness, citation quality, overall confidence, and decision status.

Greetings, audio checks, casual conversation, acknowledgments, continuation, repetition, clarification, out-of-scope refusals, and evidence-free answers are excluded from evaluation. Existing ownership, eligibility, rate-limit, and duplicate-evaluation checks remain active.

The project includes input guardrails, semantic guardrails, document prompt-injection inspection, output guardrails, safe fallbacks, protected human review, Developer Settings, and audit logging.

LangSmith configuration and a metadata-only service foundation are present. Sensitive metadata such as questions, answers, prompts, PDF text, audio, passwords, and API keys is rejected. The service is not yet wired into every normal RAG request, so configuration alone does not prove that every request is traced.

## How IntelliGround works

### PDF ingestion pipeline

~~~text
PDF upload
    ↓
File, MIME, size, page-count, and OCR-language validation
    ↓
Native extraction with OCR fallback
    ↓
Page-aware adaptive chunking
    ↓
Document/page/chunk/paragraph/offset/language metadata
    ↓
Multilingual E5 embeddings and FAISS index
    ↓
Persistent BM25 index
    ↓
Page images, summary source, and user-scoped active registry
    ↓
Document ready
~~~

Extraction preserves page boundaries and records whether content came from native extraction, OCR, or a blank page. Chunk metadata is carried into both indexes so answers can cite the original page and chunk.

### Question-answering pipeline

~~~text
User text or voice transcript
    ↓
Input guardrail and language detection
    ↓
Intent classification and conversation-context resolution
    ↓
Direct page retrieval or hybrid retrieval
    ↓
Dense FAISS candidates plus BM25 candidates
    ↓
Reciprocal Rank Fusion
    ↓
Optional cross-encoder reranking
    ↓
Evidence and citation construction
    ↓
Grounded LLM answer generation
    ↓
Output and response-language validation
    ↓
Eligible-answer evaluation and persistence
    ↓
Answer, evidence, page preview, confidence, and evaluation status
~~~

Dense retrieval finds passages with similar meaning. BM25 contributes exact words, names, numbers, and terminology. RRF combines their rankings without assuming either method is always correct. The optional reranker scores fused candidates and reports an explicit disabled or failed status when unavailable.

Page questions use the page retrieval service instead of silently substituting a different page. It reads authenticated user records, preserves page and chunk order, and can use authorized OCR fallback when stored text is unavailable. Invalid or empty pages return a truthful message rather than fabricated evidence.

## Intelligent voice agent

Each authenticated voice turn is transcribed, language-checked, classified, passed to shared query orchestration when appropriate, persisted, and synthesized back to speech.

~~~text
User: Hello, how are you?
Assistant: I'm doing well. What would you like to know about your PDF?

User: Explain page 40.
Assistant: Retrieves page 40, explains supported content, and returns page evidence.

User: Continue speaking.
Assistant: Continues without starting a duplicate document query.

User: Can you hear me?
Assistant: Yes, I can hear you clearly.
~~~

A greeting is not a PDF question. Sending it through retrieval could produce an incorrect “not found in PDF” response and an unnecessary evaluation record. A later document question still uses the active document and recent conversation context.

The frontend supports microphone capture, PCM audio worklet processing, WebSocket voice sessions, mute/output controls, speaking/listening states, and interruption handling. It uses the configured Gnani integrations rather than claiming provider-independent streaming.

## Multilingual architecture

| Concept | Meaning |
|---|---|
| PDF/evidence language | Language detected or recorded for extracted page content. |
| Query language | Language detected from the current typed question or transcript. |
| Response language | Language selected by the user and passed explicitly to generation. |
| STT language | Provider language used or detected for speech recognition. |
| TTS language | Verified provider mapping used for speech synthesis. |

The broader language registry includes English, Kannada, Tamil, Telugu, Hindi, Malayalam, Marathi, Gujarati, Bengali, Punjabi, and Urdu. Gnani mappings are separately validated, so not every text language should be assumed to support voice output. Original evidence is preserved rather than translated merely to simplify retrieval.

## Evaluation and security architecture

The evaluator combines deterministic and optional LLM-assisted checks. Retrieved document content is treated as untrusted data: instructions found inside a PDF are never promoted to system or developer instructions.

Human review is available through protected evaluation-review endpoints and Developer Settings. The user-facing evaluation drawer can evaluate eligible stored answers and save helpful/not-helpful feedback. The developer dashboard exposes evaluation counts, pending reviews, guardrail testing, and observability configuration.

## System architecture

~~~mermaid
flowchart LR
    U[User] --> FE[React TypeScript Vite]
    FE --> AUTH[FastAPI authentication]
    FE --> UP[Upload routes]
    FE --> Q[Query routes]
    FE --> V[Voice routes]
    FE --> DI[Document intelligence]
    FE --> DEV[Developer Settings]

    UP --> PDF[PyMuPDF extraction]
    PDF --> OCR[Tesseract OCR fallback]
    PDF --> CH[Adaptive page-aware chunks]
    CH --> EMB[Multilingual E5 embeddings]
    EMB --> FAISS[FAISS vector index]
    CH --> BM25[BM25 sparse index]
    PDF --> IMG[Page images]
    PDF --> REG[User-scoped document registry]

    Q --> ORCH[Shared RAG orchestration]
    V --> INTENT[Intent and context resolution]
    INTENT --> ORCH
    ORCH --> DENSE[Dense retrieval]
    ORCH --> SPARSE[Sparse retrieval]
    DENSE --> RRF[Reciprocal Rank Fusion]
    SPARSE --> RRF
    RRF --> RR[Optional reranker]
    RR --> EVID[Evidence and citations]
    EVID --> LLM[Grounded LLM generation]
    LLM --> EVAL[Evaluation services]
    EVAL --> REVIEW[Human review]

    V --> STT[Gnani Prisma STT]
    V --> TTS[Gnani Timbre TTS]
    AUTH --> DB[(SQLite by default)]
    ORCH --> DB
    EVAL --> STORE[(Persistent user storage)]
    FE -. isolated foundation .-> LS[LangSmith]
~~~

## Technology stack

| Technology | Purpose | Location |
|---|---|---|
| Python 3.11 | Backend runtime and image base | Dockerfile.backend |
| FastAPI | HTTP and WebSocket API | app/main.py and app/api/routes |
| Uvicorn | ASGI server | Dockerfile.backend |
| React 19 and TypeScript | Single-page application | frontend/intelliground-web/src |
| Vite | Frontend development and build | frontend/intelliground-web/package.json |
| PyMuPDF | PDF extraction and page access | app/services/pdf_service.py |
| Tesseract and pytesseract | OCR fallback | app/services/pdf_service.py |
| LangChain | LLM and embedding integration | requirements.backend.txt |
| OpenAI-compatible LLM | Answers, intelligence, and judge calls | app/services/llm_service.py |
| intfloat/multilingual-e5-small | Multilingual embeddings | app/services/embedding_service.py |
| FAISS | Dense vector index | app/services/vector_store_service.py |
| rank-bm25 | Sparse retrieval | app/services/bm25_service.py |
| Reciprocal Rank Fusion | Dense/sparse fusion | app/services/rrf_service.py |
| sentence-transformers | Optional cross-encoder reranking | app/services/reranker_service.py |
| SQLAlchemy and SQLite | Auth and conversation persistence | app/core/database.py |
| PyJWT, pwdlib, Authlib | Sessions, hashing, OAuth | app/services/security and auth routes |
| Gnani Prisma | Speech-to-text | app/services/gnani_prisma_service.py |
| Gnani Timbre | Text-to-speech | app/services/gnani_timbre_service.py |
| LangSmith SDK | Metadata-only observability foundation | app/services/observability |
| pytest | Backend tests | tests |

## Project structure

~~~text
.
├── app/
│   ├── api/routes/              FastAPI auth, upload, query, voice, intelligence, developer APIs
│   ├── core/                    Configuration, database, logging
│   ├── models/                  Pydantic and SQLAlchemy models
│   ├── services/                PDF, retrieval, RAG, voice, evaluation, security
│   ├── utils/                   File handling, metadata, debug export
│   └── main.py                  FastAPI application and middleware
├── frontend/intelliground-web/
│   ├── src/App.tsx              Main workspace, upload, chat, evidence, evaluation
│   ├── src/VoiceAgent.tsx       WebSocket voice UI and audio controls
│   ├── src/EvaluationDrawer.tsx User-facing answer evaluation drawer
│   ├── src/App.css              Application styling
│   └── package.json              Vite scripts and dependencies
├── storage/                    Uploads, indexes, page images, summaries, evaluations, database
├── logs/                       Application logs
├── tests/                      Backend unit and integration tests
├── benchmarks/                 Retrieval, reranker, OCR, and multilingual utilities
├── Dockerfile.backend           Backend image
├── docker-compose.yml           Backend Compose service and persistent mounts
├── requirements.backend.txt     Backend dependencies
└── README.md
~~~

Legacy and backup artifacts are intentionally omitted from this simplified tree.

## Installation and setup

### Prerequisites

- Python 3.11 or compatible.
- Node.js and npm compatible with the Vite toolchain.
- Tesseract OCR and required language packs for local non-Docker OCR.
- An OpenAI-compatible API key.
- Optional Gnani credentials for voice features.
- Optional Docker Engine and Docker Compose for the backend container.

### Install

~~~bash
git clone <repository-url>
cd IntelliGroud-

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.backend.txt

cd frontend/intelliground-web
npm install
cd ../..
~~~

Frontend scripts are npm run dev, npm run build, and npm run preview.

### Configure the environment

Create a local .env in the project root. Never commit real credentials:

~~~dotenv
OPENAI_API_KEY=<provider-key>
RAG_ACCESS_PASSWORD=<rag-access-password>
JWT_SECRET_KEY=<long-random-secret>
FRONTEND_URL=http://localhost:5175
VITE_API_BASE_URL=http://localhost:8000/api
~~~

A checked-in .env.example was not present during inspection. Use placeholders and keep the real .env local.

### Run locally

Backend:

~~~bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
~~~

Frontend in a second terminal:

~~~bash
cd frontend/intelliground-web
npm run dev -- --host 0.0.0.0 --port 5175
~~~

URLs:

- UI: http://localhost:5175
- Health: http://localhost:8000/health
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker backend

The current Compose file defines only a backend service:

~~~bash
docker compose build backend
docker compose up -d --force-recreate backend
~~~

It maps 8000:8000, loads .env, and mounts ./storage and ./logs into the container. The frontend is not a Compose service and should be run with Vite unless deployment-specific configuration is added.

## Environment variables

### Required backend variables

| Variable | Purpose |
|---|---|
| OPENAI_API_KEY | LLM/provider authorization |
| RAG_ACCESS_PASSWORD | Protected RAG authorization |
| JWT_SECRET_KEY | Signs browser access tokens |

### Application and authentication

| Variable | Purpose | Default or status |
|---|---|---|
| FRONTEND_URL | Primary allowed frontend origin | Backend default is http://localhost:8501; local Vite commonly uses 5175 |
| COOKIE_SECURE | Secure-cookie behavior | Deployment-specific |
| DATABASE_URL | SQLAlchemy database URL | SQLite at storage/app.db |
| LOCAL_DEV_AUTH_BYPASS | Optional local development identity | false |
| DEVELOPER_ACCESS_PASSWORD | Developer Settings access | Optional, required for that feature |
| GOOGLE_CLIENT_ID | Google OAuth client ID | Optional |
| GOOGLE_CLIENT_SECRET | Google OAuth client secret | Optional |
| GOOGLE_REDIRECT_URI | Google OAuth callback | Optional |
| EVALUATION_MODEL | Evaluation model | gpt-4o-mini |

### Voice

| Variable | Purpose |
|---|---|
| GNANI_API_KEY | Gnani authorization; optional until voice is configured |
| GNANI_STT_MODEL | Prisma model; default prisma-v3.0 |
| GNANI_TTS_MODEL | Timbre model; default timbre-v2.5 |
| GNANI_TTS_VOICE | Timbre voice; default Kaveri |
| GNANI_STT_URL | Buffered STT endpoint |
| GNANI_TTS_URL | Buffered TTS endpoint |
| GNANI_STT_WS_URL | Streaming STT WebSocket |
| GNANI_TTS_WS_URL | Streaming TTS WebSocket |
| GNANI_REQUEST_TIMEOUT_SECONDS | Provider timeout; default 60 |
| GNANI_MAX_AUDIO_BYTES | Maximum audio payload; default 15 MiB |

### Retrieval, OCR, observability, and frontend

| Variable | Purpose |
|---|---|
| RERANKER_ENABLED | Enable optional cross-encoder reranking; default false |
| RERANKER_MODEL | Reranker model; default BAAI/bge-reranker-v2-m3 |
| RERANKER_MAX_LENGTH | Reranker sequence length; default 512 |
| RRF_CONSTANT | RRF rank constant; default 60 |
| RRF_CANDIDATE_LIMIT | Fused candidate limit |
| FINAL_EVIDENCE_COUNT | Final evidence limit |
| DENSE_CANDIDATE_COUNT | Dense candidate tuning |
| SPARSE_CANDIDATE_COUNT | Sparse candidate tuning |
| PDF_OCR_DEFAULT_LANG | OCR fallback language preference |
| LANGSMITH_TRACING | Enable isolated LangSmith foundation |
| LANGSMITH_ENDPOINT | LangSmith API endpoint |
| LANGSMITH_API_KEY | LangSmith authorization |
| LANGSMITH_PROJECT | LangSmith project |
| VITE_API_BASE_URL | Frontend backend base URL; default http://localhost:8000/api |

Do not place secrets in frontend source files. The frontend uses authenticated browser cookies for API requests.

## API overview

All routes below are under /api unless noted otherwise. Protected routes use the authenticated browser session; document and answer operations also enforce ownership and active-document checks where applicable.

| Method | Endpoint | Purpose | Auth |
|---|---|---|---|
| GET | /health | Backend health check | No |
| POST | /api/auth/register | Create account | No |
| POST | /api/auth/login | Create browser session | No |
| GET | /api/auth/me | Read current user | Yes |
| POST | /api/auth/logout | End browser session | Session |
| GET | /api/auth/google/login | Begin Google OAuth | No |
| GET | /api/auth/google/callback | Complete OAuth | Callback |
| POST | /api/upload-pdf | Upload and index PDF | Yes |
| GET | /api/upload-pdf/{job_id}/progress | Upload progress | Yes |
| POST | /api/ask | Shared grounded document QA | Yes + RAG authorization |
| POST | /api/evaluation/answer | Evaluate eligible stored answer | Yes |
| POST | /api/evaluation/answer/feedback | Save answer feedback | Yes |
| GET | /api/evaluation/{evaluation_id} | Read owned evaluation | Yes |
| POST | /api/evaluation/{evaluation_id}/review | Submit human review | Protected |
| POST | /api/document/language | Change response language | Yes |
| POST | /api/document/summary | Generate summary | Yes |
| POST | /api/document/topics | Generate topics/key points | Yes |
| POST | /api/document/questions | Generate suggested questions | Yes |
| POST | /api/summary | Legacy summary route | Yes |
| POST | /api/suggested-questions | Legacy questions route | Yes |
| GET | /api/page-images/{doc_id}/{page_filename} | Serve owned page image | Ownership checked |
| POST | /api/voice/onboarding-speech | Onboarding speech | Yes |
| POST | /api/voice/transcribe | Transcribe audio | Yes |
| POST | /api/voice/sessions | Create voice session | Yes |
| GET | /api/voice/conversations | List voice conversations | Yes |
| DELETE | /api/voice/conversations/{conversation_id} | Delete conversation | Yes |
| POST | /api/voice/sessions/{conversation_id}/turn | Buffered voice turn | Yes |
| POST | /api/voice/sessions/{conversation_id}/synthesize | Synthesize session text | Yes |
| WS | /api/voice/sessions/{conversation_id}/stream | Streaming voice session | Session auth |
| GET | /api/developer/dashboard | Developer dashboard | Developer access |
| POST | /api/developer/evaluations/{evaluation_id}/review | Developer review | Developer access |
| POST | /api/developer/guardrails/test | Guardrail smoke test | Developer access |

FastAPI's interactive schema is available at /docs.

## Testing

Backend tests:

~~~bash
source .venv/bin/activate
pytest -q
~~~

Focused examples:

~~~bash
pytest -q tests/test_conversational_intents.py \
  tests/test_shared_intent.py \
  tests/test_intent_orchestration.py \
  tests/test_voice_orchestration.py \
  tests/test_page_retrieval_service.py \
  tests/test_evaluation_service.py
~~~

Frontend build:

~~~bash
cd frontend/intelliground-web
npm run build
~~~

Tests cover authentication, ownership isolation, upload validation, OCR, chunking, embeddings, FAISS/BM25/RRF retrieval, reranking, page retrieval, document intelligence, voice services, intent routing, guardrails, evaluation, persistence, and observability helpers.

Focused validation has passed in this repository, including voice/orchestration suites and the frontend build. The full suite should be run in the target environment; model, OCR, and indexing tests can take substantially longer. Focused results must not be presented as production-readiness evidence.

## Security and privacy

Implemented protections include:

- password hashing and signed browser sessions;
- authentication and request rate limiting;
- per-user active-document registries and storage paths;
- ownership checks for conversations, answers, evaluations, and page images;
- active-document checks before retrieval;
- file type, size, page-count, and upload-job validation;
- input and semantic guardrails;
- document-content prompt-injection inspection;
- output guardrails;
- safe fallback responses when evidence is insufficient;
- protected Developer Settings and human-review operations;
- redacted operational logging;
- rejection of secrets and user content from LangSmith metadata fields.

Limitations remain: local development can intentionally enable an auth bypass, external provider security depends on configuration, and no evaluation system guarantees correctness. Keep .env, uploaded documents, indexes, databases, and logs outside source control.

## Limitations and roadmap

### Current limitations

- LangSmith is a safe configuration foundation, not automatic tracing for every RAG request.
- Reranking is disabled by default and depends on local model availability.
- Voice output support depends on verified Gnani mappings, not only the broader text registry.
- The current Compose file defines no frontend service; the frontend runs with Vite.
- Full-suite runtime can be long because of model, OCR, and indexing tests.
- Conversation context is bounded to recent persisted turns.
- Evaluation quality depends on retrieved evidence, configured models, and judge availability.
- No license file was present during inspection.

### Near-term roadmap

1. Connect metadata-only LangSmith traces to shared orchestration.
2. Add deployment packaging for the Vite frontend.
3. Expand browser and live-provider tests for upload, evidence, evaluation, and voice permissions.
4. Expand multilingual intent examples and provider-verified voice mappings.
5. Improve structured memory for topic references and unfinished explanations.
6. Add operational metrics and documented index/model refresh procedures.

## Project status

IntelliGround is an active engineering project. Its strongest guarantees are architectural and validation-oriented; model quality, external provider availability, and deployment configuration still require environment-specific verification.
