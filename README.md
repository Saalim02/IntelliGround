# IntelliGround

### Evidence-grounded multilingual PDF intelligence for text and voice

IntelliGround is a React and FastAPI application for asking questions about uploaded PDF documents. It extracts page-aware evidence, indexes each user's active document with dense and sparse retrieval, generates answers in a selected response language, and exposes citations back to source-page and paragraph metadata.

The application also includes document summaries, key topics, suggested questions, persisted conversations, answer evaluation, guardrails, and an optional Gnani Prisma/Timbre voice interface.

> This README describes the implementation currently present in this repository. Provider credentials, model downloads, OCR availability, browser permissions, and deployment configuration require environment-specific verification.

## Contents

- [What it provides](#what-it-provides)
- [Feature status](#feature-status)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Repository layout](#repository-layout)
- [Local setup](#local-setup)
- [Environment variables](#environment-variables)
- [API overview](#api-overview)
- [RAG pipeline](#rag-pipeline)
- [Voice agent](#voice-agent)
- [Testing and validation](#testing-and-validation)
- [Deployment](#deployment)
- [Security](#security)
- [Limitations and roadmap](#limitations-and-roadmap)
- [Contributing and license](#contributing-and-license)
- [Contact](#contact)

## What it provides

IntelliGround is designed for long, multilingual and occasionally scanned PDFs where keyword search alone is insufficient and unsupported LLM answers are unacceptable. Its differentiators are:

- page- and chunk-preserving document processing;
- dense, BM25 and hybrid retrieval with Reciprocal Rank Fusion (RRF);
- optional cross-encoder reranking with explicit disabled/failed states;
- document-scoped answer generation with evidence and citation metadata;
- separate text-language, query-language, response-language and provider-language handling;
- user-scoped documents, conversations, evaluations and page previews;
- a shared orchestration path for typed questions and eligible voice turns.

Typical uses include policy and legal-document exploration, multilingual study material, internal reports, scanned records, evidence-backed document Q&A, and hands-free document interaction.

## Feature status

| Area | Status | Implementation notes |
|---|---|---|
| PDF upload and validation | Implemented | PDF checks, size/page limits, upload progress and per-user active-document tracking. |
| Native extraction and OCR | Implemented | PyMuPDF extraction with Tesseract/PyTesseract fallback and configurable OCR languages. |
| Multilingual processing | Implemented with provider limits | Text registry includes English, Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Gujarati, Bengali, Punjabi and Urdu. |
| Dense retrieval | Implemented | intfloat/multilingual-e5-small embeddings and FAISS indexes. |
| Sparse retrieval | Implemented | Persistent BM25 using rank-bm25, with an internal fallback. |
| Hybrid retrieval and RRF | Implemented | Dense and sparse candidates are fused using Reciprocal Rank Fusion. |
| Reranking | Optional | Cross-encoder reranking is disabled by default and depends on local model availability. |
| Grounded answers and citations | Implemented | Active-document evidence returns page/chunk/paragraph metadata where available. |
| Summaries, topics and questions | Implemented | On-demand document intelligence uses the same evidence boundary. |
| LLM providers | Implemented/configuration-dependent | OpenAI is supported; an Evon adapter exists for an explicitly configured compatible endpoint. |
| Provider fallback | Implemented | Retryable primary-provider failures can use a separately configured fallback. |
| Voice STT/TTS | Implemented/configuration-dependent | Gnani Prisma STT and Timbre TTS adapters support REST and WebSocket paths. |
| Voice streaming | Implemented in application code | Browser PCM capture, FastAPI WebSocket sessions, interruption handling and streamed playback. |
| Emotion-aware behavior | Implemented as response state | Emotion affects delivery guidance and UI state. A generic Timbre emotion field is not claimed or sent. |
| Voice selection | Implemented | Voice catalogue and preference routes expose the configured Timbre v2.5 catalogue. |
| Conversation persistence | Implemented | Conversations, turns, document metadata, response language and length are persisted. |
| Evaluation and guardrails | Implemented with dependencies | Retrieval, grounding, source relevance, correctness, citation, confidence, human review and guardrails are present. |
| LangSmith | Partial foundation | Metadata-only tracing utilities and controls exist, but they are not wired into every normal RAG request. |

Urdu is supported by the text-language registry and text response validation. It is deliberately excluded from the current verified Gnani voice-language allow-list until provider support is confirmed; the voice path fails explicitly rather than silently substituting English.

## Architecture

The React/Vite client communicates with FastAPI over HTTP and WebSocket. Uploaded PDFs are stored under user-scoped paths. Extracted pages are chunked, embedded and indexed in dense and sparse stores. Query and voice orchestration resolve intent and context before retrieval. Only eligible document questions proceed through evidence construction and grounded generation.

~~~mermaid
flowchart LR
    User[User] --> React[React + TypeScript + Vite]
    React --> HTTP[FastAPI HTTP API]
    React --> WS[FastAPI Voice WebSocket]

    HTTP --> Auth[Authentication and ownership]
    HTTP --> Upload[PDF upload]
    Upload --> Extract[PyMuPDF extraction]
    Extract --> OCR[Tesseract fallback]
    Extract --> Chunk[Page-aware chunks and metadata]
    Chunk --> Embed[Multilingual E5 embeddings]
    Embed --> FAISS[FAISS index]
    Chunk --> BM25[BM25 index]
    Extract --> Images[Page images]

    HTTP --> Query[Query orchestration]
    WS --> Prisma[Gnani Prisma STT]
    Prisma --> Query
    Query --> Intent[Intent and context resolution]
    Intent --> Retrieve[Active-document retrieval]
    Retrieve --> Dense[Dense candidates]
    Retrieve --> Sparse[Sparse candidates]
    Dense --> RRF[RRF fusion]
    Sparse --> RRF
    RRF --> Rerank[Optional cross-encoder]
    Rerank --> Evidence[Evidence and citations]
    Evidence --> LLM[OpenAI or configured Evon LLM]
    LLM --> Validate[Language, output and guardrails]
    Validate --> Persist[Conversation and evaluation storage]
    Validate --> React
    Validate --> Timbre[Gnani Timbre TTS]
    Timbre --> React

    Auth --> DB[(SQLite by default)]
    Persist --> DB
~~~

## Technology stack

| Category | Technologies confirmed in the repository |
|---|---|
| Frontend | React 19, TypeScript, Vite 8, React DOM, Oxlint |
| Backend | Python 3.11 container target, FastAPI, Uvicorn, Pydantic, python-multipart |
| LLM | LangChain, langchain-openai, OpenAI adapter, Evon adapter and retryable fallback gateway |
| Embeddings | Hugging Face intfloat/multilingual-e5-small, langchain-huggingface |
| Retrieval | FAISS, rank-bm25, NumPy, RRF, optional sentence-transformers cross-encoder |
| PDF/OCR | PyMuPDF, Tesseract, PyTesseract, Pillow |
| Speech | Gnani Prisma STT, Gnani Timbre TTS, websockets, browser AudioWorklet/PCM capture |
| Persistence | SQLAlchemy, SQLite by default, filesystem storage |
| Authentication | PyJWT, pwdlib[argon2], Authlib, cryptography |
| Operations | Dockerfile, Docker Compose backend service, environment configuration and logging |
| Testing/observability | pytest, LangSmith SDK foundation, evaluation and benchmark utilities |

## Repository layout

~~~text
.
├── app/
│   ├── api/routes/          FastAPI auth, upload, query, document, voice and developer routes
│   ├── core/                Configuration, database and logging
│   ├── models/              Pydantic and SQLAlchemy models
│   ├── services/
│   │   ├── llm/             OpenAI/Evon adapters and fallback gateway
│   │   ├── observability/   Metadata-only LangSmith foundation
│   │   ├── security/        Authentication, authorization and guardrails
│   │   └── evaluation/      Retrieval, grounding, citation and confidence evaluation
│   ├── utils/               File, metadata and debug helpers
│   └── main.py              FastAPI app, middleware and route registration
├── frontend/intelliground-web/
│   ├── src/App.tsx          Text workspace, upload, conversation and evaluation UI
│   ├── src/VoiceAgent.tsx   Voice UI, AudioWorklet capture and WebSocket lifecycle
│   ├── src/EvaluationDrawer.tsx
│   ├── src/App.css
│   └── package.json
├── benchmarks/              Retrieval, reranker, OCR and multilingual utilities
├── tests/                   Backend unit/integration and benchmark tests
├── storage/                 Runtime uploads, indexes, images, summaries and database
├── logs/                    Runtime logs
├── Dockerfile.backend
├── docker-compose.yml       Backend-only Compose service
├── requirements.backend.txt
├── requirements.txt
├── pytest.ini
└── README.md
~~~

Runtime directories are created by the application and should not be treated as source-controlled data.

## Local setup

### Prerequisites

- Python 3.11 or a compatible Python environment.
- Node.js and npm compatible with the Vite toolchain.
- Tesseract OCR and required language packs for local non-Docker OCR.
- An OpenAI API key for the default LLM path.
- Optional Gnani credentials for voice features.
- Optional Docker Engine and Docker Compose for the backend container.

### Clone and install

~~~bash
git clone https://github.com/Saalim02/IntelliGround.git
cd IntelliGround

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.backend.txt

cd frontend/intelliground-web
npm install
cd ../..
~~~

The repository does not contain a checked-in .env.example; create .env locally from the variable list below. Never commit real credentials.

### Configure and run

The backend validates OPENAI_API_KEY, RAG_ACCESS_PASSWORD and JWT_SECRET_KEY at startup.

~~~bash
# Terminal 1, from the repository root
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
cd frontend/intelliground-web
npm run dev -- --host 0.0.0.0 --port 5173
~~~

Open http://localhost:5173. The API is available at http://localhost:8000, with health at /health, OpenAPI at /docs, and ReDoc at /redoc.

### Docker Compose

The checked-in Compose file defines a backend-only service. It loads .env, exposes port 8000, and mounts ./storage and ./logs.

~~~bash
docker compose up --build backend
~~~

The frontend is not a Compose service and must be run separately with Vite unless deployment-specific packaging is added.

## Environment variables

Use placeholders in local configuration:

~~~dotenv
OPENAI_API_KEY=your_openai_api_key_here
RAG_ACCESS_PASSWORD=your_rag_access_password_here
JWT_SECRET_KEY=replace_with_a_long_random_secret
FRONTEND_URL=http://localhost:5173
VITE_API_BASE_URL=http://localhost:8000/api
~~~

### Required and core variables

| Variable | Required | Purpose/default |
|---|---:|---|
| OPENAI_API_KEY | Yes | Default LLM and authorization path. |
| RAG_ACCESS_PASSWORD | Yes | Protected RAG authorization. |
| JWT_SECRET_KEY | Yes | Signs browser access tokens. |
| FRONTEND_URL | No | Primary allowed frontend origin; backend default is http://localhost:8501. |
| DATABASE_URL | No | SQLAlchemy URL; defaults to SQLite under storage/app.db. |
| COOKIE_SECURE | No | Secure-cookie flag; defaults to false. |
| LOCAL_DEV_AUTH_BYPASS | No | Explicit local development identity; defaults to false. |
| DEVELOPER_ACCESS_PASSWORD | No | Enables protected developer access. |
| GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI | No | Google OAuth configuration. |
| EVALUATION_MODEL | No | Evaluation model; defaults to gpt-4o-mini. |

### LLM provider variables

| Variable | Purpose/default |
|---|---|
| LLM_PRIMARY_PROVIDER | openai or evon; defaults to openai. |
| LLM_FALLBACK_PROVIDER | openai or evon; defaults to openai. |
| OPENAI_MODEL | Defaults to gpt-4o-mini. |
| EVON_API_KEY, EVON_BASE_URL, EVON_MODEL | Required when Evon is selected; the endpoint is not inferred. |
| LLM_REQUEST_TIMEOUT_SECONDS | Defaults to 60. |
| LLM_MAX_RETRIES | Defaults to 1, validated to 0–3. |

### Voice variables

| Variable | Purpose/default |
|---|---|
| GNANI_API_KEY | Gnani authorization; required for live voice calls. |
| GNANI_STT_MODEL | Defaults to prisma-v3.0. |
| GNANI_TTS_MODEL | Defaults to timbre-v2.5. |
| GNANI_TTS_VOICE | Defaults to Kaveri. |
| GNANI_STT_URL, GNANI_TTS_URL | REST provider endpoints with repository defaults. |
| GNANI_STT_WS_URL, GNANI_TTS_WS_URL | Provider WebSocket endpoints with repository defaults. |
| GNANI_REQUEST_TIMEOUT_SECONDS | Defaults to 60. |
| GNANI_MAX_AUDIO_BYTES | Defaults to 15 MiB. |

### Retrieval, OCR and observability variables

DENSE_CANDIDATE_COUNT, SPARSE_CANDIDATE_COUNT, RRF_CONSTANT, RRF_CANDIDATE_LIMIT, RERANKER_CANDIDATE_LIMIT, FINAL_EVIDENCE_COUNT, RERANKER_ENABLED, RERANKER_MODEL, RERANKER_MAX_LENGTH, EMBEDDING_MODEL_VERSION, EMBEDDING_DIMENSIONS and EMBEDDING_BATCH_SIZE tune retrieval and model loading. PDF_OCR_DEFAULT_LANG controls the OCR fallback preference.

LangSmith configuration uses LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_API_KEY and LANGSMITH_PROJECT. These variables enable the isolated observability foundation; they do not imply that every normal request is automatically traced.

The frontend reads VITE_API_BASE_URL and defaults to http://localhost:8000/api. Do not put backend secrets in Vite variables or frontend source.

## API overview

The FastAPI application mounts these routes under /api unless noted. Protected endpoints use the authenticated browser session; document operations also enforce ownership and active-document/RAG authorization where applicable. FastAPI's generated documentation is authoritative at /docs.

### Health and authentication

| Method | Route | Purpose |
|---|---|---|
| GET | /health | Health check; public. |
| POST | /api/auth/register | Create a local account. |
| POST | /api/auth/login | Create the browser session. |
| POST | /api/auth/logout | Clear the browser session. |
| GET | /api/auth/me | Return the current user. |
| POST | /api/auth/forgot-password | Start password recovery. |
| POST | /api/auth/reset-password | Complete password recovery. |
| GET | /api/auth/google/login | Start Google OAuth when configured. |
| GET | /api/auth/google/callback | Complete Google OAuth. |
| POST | /api/auth/onboarding/complete | Mark authenticated onboarding complete. |

### Documents, queries and persistence

| Method | Route | Purpose |
|---|---|---|
| POST | /api/upload-pdf | Validate, store, extract and index a PDF. |
| GET | /api/upload-pdf/{job_id}/progress | Read upload-job progress. |
| POST | /api/ask | Run shared grounded document question answering. |
| GET | /api/document/current | Read the active document. |
| POST | /api/document/language | Save response language. |
| POST | /api/document/summary | Generate a grounded summary. |
| POST | /api/document/topics | Generate grounded topics/key points. |
| POST | /api/document/questions | Generate grounded suggested questions. |
| POST | /api/summary | Legacy summary route. |
| POST | /api/suggested-questions | Legacy suggested-question route. |
| GET | /api/conversations | List owned conversations. |
| GET | /api/conversations/{conversation_id} | Restore an owned conversation and document metadata. |
| PATCH | /api/conversations/{conversation_id} | Update title/language/length preferences. |
| GET | /api/page-images/{doc_id}/{page_filename} | Serve an owned generated page image. |

### Evaluation and developer operations

| Method | Route | Purpose |
|---|---|---|
| GET | /api/evaluation | List the user's evaluations. |
| POST | /api/evaluation/answer | Evaluate an eligible stored answer. |
| POST | /api/evaluation/answer/feedback | Save answer feedback. |
| GET | /api/evaluation/{evaluation_id} | Read an owned evaluation. |
| POST | /api/evaluation/{evaluation_id}/review | Submit a protected human review. |
| GET | /api/developer/dashboard | Developer evaluation dashboard. |
| GET/PUT | /api/developer/evaluation-config | Read/update evaluation settings. |
| GET/PUT | /api/developer/emotion-settings | Read/update response-emotion settings. |
| GET | /api/developer/observability | Read observability status and metrics. |
| GET | /api/developer/observability/traces | Read retained trace metadata. |
| GET | /api/developer/providers | Read provider status without secrets. |
| POST | /api/developer/guardrails/test | Run a protected guardrail test. |

Developer routes require developer access, not merely an ordinary user session.

### Voice

| Method | Route | Purpose |
|---|---|---|
| GET | /api/voice/voices | List the application voice catalogue. |
| POST | /api/voice/voices/preview | Preview a validated voice/language pair. |
| GET/PUT | /api/voice/preferences | Read/save voice and response-emotion preferences. |
| POST | /api/voice/transcribe | Transcribe uploaded audio with Prisma. |
| POST | /api/voice/onboarding-speech | Synthesize onboarding speech. |
| POST | /api/voice/sessions | Create a persisted voice conversation. |
| GET | /api/voice/conversations | List owned voice conversations. |
| DELETE | /api/voice/conversations/{conversation_id} | Delete an owned voice conversation. |
| POST | /api/voice/sessions/{conversation_id}/turn | Process one buffered audio turn. |
| POST | /api/voice/sessions/{conversation_id}/synthesize | Synthesize session text. |
| POST | /api/voice/synthesize-answer | Synthesize an already-generated answer. |
| WS | /api/voice/sessions/{conversation_id}/stream | Authenticate and stream a voice session. |

## RAG pipeline

1. **Upload:** validate PDF type, size and page-count limits; save under the authenticated user's storage and create upload progress.
2. **Extraction:** PyMuPDF extracts page text; unusable pages can use Tesseract OCR. Metadata records page, language and extraction method.
3. **Chunking:** adaptive page-aware chunks preserve document, page, paragraph and character-offset metadata.
4. **Embeddings:** intfloat/multilingual-e5-small creates normalized 384-dimensional embeddings with passage and query prompts.
5. **Indexing:** FAISS stores dense vectors; a JSON-persisted BM25 index stores Unicode-tokenized sparse records.
6. **Retrieval:** the active document is enforced before candidates are accepted. Dense and sparse searches use configurable limits.
7. **Fusion:** RRF merges candidates by stable document/chunk identity.
8. **Reranking:** an optional cross-encoder reranks fused candidates; disabled/failed states are explicit.
9. **Context:** selected chunks become bounded evidence with page, citation, source, snippet and image metadata.
10. **Generation:** the LLM gateway generates from authorized document context for document intents. Non-RAG intents do not bypass the document policy.
11. **Validation:** output language, citation consistency, evidence and output guardrails are checked.
12. **Persistence/evaluation:** eligible answers, turns and evaluation results are stored under the authenticated user.

Page-specific requests use the page retrieval service rather than silently substituting a different page. Selected response language is separate from detected document/query language, and Urdu remains text-supported but not currently voice-verified.

## Voice agent

The voice path is:

1. The browser requests microphone access and captures mono audio through an AudioWorklet.
2. PCM frames are sent over the authenticated FastAPI WebSocket session.
3. Prisma STT transcribes the turn and reports provider language metadata when available.
4. Voice intent and recent context determine whether shared RAG orchestration is needed.
5. Grounded answers are persisted and streamed through the voice session.
6. Timbre TTS produces buffered or streamed audio, which the browser schedules for playback.
7. Interruption handling can stop playback and resume listening; cleanup closes audio contexts, stops media tracks, clears timers and closes sockets.

Response-emotion states such as HAPPY, CALM and PROFESSIONAL affect response delivery and UI behavior. The application does **not** claim that Timbre receives a generic emotion parameter or produces verified emotional audio. Provider-level expressive behavior depends on the configured account and documented provider contract.

Unit and route tests cover voice orchestration, context preservation, provider payloads and cancellation. Browser microphone permissions, AudioWorklet behavior and live Prisma/Timbre sessions require separate validation.

## Testing and validation

### Backend

~~~bash
source .venv/bin/activate
python -m pytest -q
~~~

Tests cover authentication, ownership isolation, upload validation, OCR, chunking, embeddings, FAISS/BM25/RRF retrieval, reranking, page retrieval, document intelligence, multilingual language handling, voice services, intent routing, guardrails, evaluation, persistence and observability helpers.

At the time of this README update, the targeted voice/RAG validation selection passed **178 tests** locally. This is not a claim that the full suite or live external providers have been verified in every environment.

### Frontend

~~~bash
cd frontend/intelliground-web
npm run build
npm run lint
~~~

The frontend build is TypeScript-checked and bundled with Vite. Lint currently reports warnings in existing code; warnings are not browser verification.

## Deployment

The repository provides a Python 3.11 backend Dockerfile and a Docker Compose file with one backend service:

~~~bash
docker compose up --build backend
~~~

The container installs Tesseract language packs, runs Uvicorn on port 8000 as a non-root user, and mounts storage and logs. The frontend is not packaged by the checked-in Compose file. No EC2-specific scripts, infrastructure definitions or production reverse-proxy configuration are present, so EC2 is not documented here as a verified workflow.

## Security

Implemented controls include:

- password hashing and signed browser access tokens;
- optional Google OAuth configuration;
- authentication, developer-access and authentication-specific rate limiting;
- per-user document, conversation, evaluation and page-image ownership checks;
- active-document and RAG authorization before retrieval;
- PDF type, size, page-count and upload validation;
- semantic/input guardrails, document prompt-injection inspection and output guardrails;
- protected human-review and developer endpoints;
- sanitized provider errors and redacted operational logging;
- LangSmith metadata allow-listing that rejects questions, answers, prompts, PDF text, audio, credentials and tokens.

Keep .env, uploaded documents, indexes, databases and logs out of commits. LOCAL_DEV_AUTH_BYPASS=true is intended only for explicitly local development.

## Limitations and roadmap

Current limitations:

- LangSmith is a metadata-only foundation and is not automatically attached to every normal RAG request.
- Cross-encoder reranking is optional, disabled by default and model-download dependent.
- Evon requires official credentials and an explicit compatible base URL; the repository does not infer one.
- Gnani voice mappings are provider-dependent; Urdu voice is intentionally unsupported in the current allow-list.
- Live provider, browser microphone and end-to-end multi-viewport validation are not represented by unit-test results.
- Compose does not include the frontend.
- No LICENSE file was found in the repository.

Planned improvements:

1. Add browser-level tests for authentication, upload, scrolling, text/voice switching and microphone cleanup.
2. Add live-provider smoke tests behind explicit opt-in credentials.
3. Integrate metadata-only LangSmith tracing into shared orchestration after privacy review.
4. Add a documented frontend deployment target and production reverse-proxy configuration.
5. Publish repeatable retrieval/OCR benchmark procedures and model/index refresh guidance.

## Contributing and license

Before opening a change:

1. Inspect the relevant route, service, model and test boundaries.
2. Add or update focused tests for behavior changes.
3. Run backend tests and frontend build/lint.
4. Run git diff --check.
5. Keep secrets, runtime storage and generated model artifacts out of commits.

No LICENSE file was present during repository inspection. Do not assume or advertise an open-source license until the project owner adds one.

## Contact

Repository: [https://github.com/Saalim02/IntelliGround](https://github.com/Saalim02/IntelliGround)
