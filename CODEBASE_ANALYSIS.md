# VibeRAG Codebase Analysis: Operation Autistic Deep Dive

Yo. We dove headfirst into this digital labyrinth. Here's the fucking sitrep. Feels good man.

## TL;DR - The High-Level Gist

This is a **React/TypeScript frontend** talking to a **Python/FastAPI backend**. The backend is a **sophisticated RAG (Retrieval-Augmented Generation) system** designed for chat, presentation generation, and research tasks. It leverages **Milvus** for vector storage, **Sentence Transformers** for embeddings, and can integrate with **Ollama** and/or **OpenAI** for LLM generation. It's containerized via **Docker Compose** and includes multi-user **JWT authentication** and **SQLAlchemy** for DB persistence (users, chat history).

## Directory Structure - The Lay of the Land

```
.
├── backend/          # Python/FastAPI RAG core, ML/NLP logic
├── frontend/         # React/TypeScript UI, Node.js dev server
├── docs/             # Documentation (hopefully not lies)
├── media/            # Static assets?
├── volumes/          # Docker volume persistence (DBs, etc.)
├── .cursor/          # Our command center
├── .git/             # Version control autism
├── .venv/            # Python virtual env
├── CODEBASE_ANALYSIS.md # This fucking masterpiece
├── docker-compose.yml # Service orchestration
├── requirements.txt  # Backend Python deps (inside backend/)
├── package.json      # Frontend Node deps (inside frontend/)
├── .env.example      # Environment variable template
├── .env.local        # Local environment variables (!!! IMPORTANT !!!)
├── .gitignore        # Standard ignore list
├── cleanup.sh        # Utility script
├── backend_server.log # Backend logs
├── frontend_server.log # Frontend logs (dev server)
├── whitepaper.pdf    # 8.2MB of potential cringe
└── ... other config files
```

## Frontend (`frontend/`) - The Pretty Face

*   **Framework:** React 18 (`react`, `react-dom`)
*   **Build:** Create React App (`react-scripts`) - Boilerplate king, kinda cringe but predictable.
*   **Language:** TypeScript - Based choice, avoids some JS footguns.
*   **Routing:** `react-router-dom` v6
*   **Styling:** `styled-components`, `@tremor/react` (UI component library - could be nice)
*   **API Comms:** `axios` for HTTP, native `WebSocket` API for chat.
*   **Dev Server (`server.js`):** Node/Express server that:
    *   Serves the static React build (`build/`).
    *   Manually proxies `/api` HTTP requests to `http://backend:8000`.
    *   Manually proxies WebSocket connections for `/api/v1/ws/chat` to `ws://backend:8000/api/v1/ws/chat` (includes heartbeat logic).
*   **Production:** Served via Docker (`Dockerfile`, `nginx.conf`).

## Backend (`backend/`) - The Engine Room & AI Brain

*   **Framework:** FastAPI - Async, fast, Pydantic validation. Maximum basedness.
*   **Language:** Python
*   **Entry Point:** `src/api/app.py` - Defines FastAPI app, middleware, lifespan events, and API routes.
*   **Core Modules (`src/modules/`):** Logic is modularized:
    *   `api`: FastAPI app definition, routes, request/response models.
    *   `auth`: User models (SQLAlchemy), JWT authentication, password hashing (`bcrypt`), user/admin roles.
    *   `chat.history`: CRUD operations for chat sessions/messages in the DB.
    *   `config`: Pydantic models for loading `.env.local` settings. Centralized config management.
    *   `embedding`: Sentence Transformer embedding generation (`embed.py`, `service.py`). Uses a singleton service for efficiency.
    *   `generation`: LLM interaction (`generate.py`), presentation (`slides.py`). Handles Ollama/OpenAI adapters, prompt construction, streaming via WebSockets.
    *   `ingestion`: Document processing (`ingest.py`) using `unstructured`, chunking, embedding calls, storing in Milvus.
    *   `research`: Research report generation logic (`research.py`).
    *   `retrieval`: Semantic search (`search.py`), Milvus query construction, web search (Google CSE API).
    *   `vector_store`: Milvus operations (`milvus_ops.py`) - connection, schema, collection management (user/admin/global), storage, search execution, metadata updates.
*   **Database:** SQLAlchemy for relational data (Users, Chat History). Alembic mentioned in `requirements.txt` for migrations (good practice).
*   **Vector DB:** Milvus - Handles storing and searching text embeddings. Collections managed dynamically (`user_{id}`, `user_admin`, `global_kb`).
*   **Object Storage:** Minio (`minio` client in `requirements.txt`) - Likely used for storing original uploaded documents (`STORAGE_DIR = backend/storage/documents`).
*   **Deployment:** Containerized via `Dockerfile`.

## Core RAG Pipeline - How the Magic Happens

1.  **Ingestion (`ingest.py` -> `milvus_ops.py`):**
    *   User uploads document via `/upload` endpoint (`app.py`).
    *   `upload_document` determines target Milvus collection (user/admin/global).
    *   `unstructured` parses the document content.
    *   Text is chunked based on token count (`chunk_text`).
    *   `embed_chunks` uses the Sentence Transformer singleton service (`service.py`, `embed.py`) to generate embeddings for text chunks in batches.
    *   `store_with_metadata` inserts chunks (text, embedding, filename, metadata, tags, batch doc_id) into the appropriate Milvus collection.

2.  **Retrieval & Generation (WebSocket via `generate.py` -> `search.py` -> `milvus_ops.py`):**
    *   Frontend connects to `/ws/chat` with JWT token.
    *   `websocket_endpoint` authenticates user, handles session setup.
    *   Client sends `chat_message` with query.
    *   `chat_with_knowledge_core` orchestrates the RAG flow:
        *   **Determine Collections:** Identifies user, admin (if applicable), and global collections. Dynamically creates them via `init_collection` if they don't exist (`semantic_search`).
        *   **Embed Query:** Generates embedding for the user query (`get_embeddings`).
        *   **Vector Search:** Calls `search_collection` (`milvus_ops.py`) to perform vector search across relevant collections, potentially applying filters (`_build_milvus_filter_expression`). Results are aggregated and re-ranked.
        *   **Web Search (Optional):** Calls `google_search` if requested.
        *   **Filter Results:** Applies `min_score` threshold (`semantic_search`).
        *   **Construct Prompt:** Builds a prompt including system message, retrieved document chunks, web results (if any), chat history (from DB via `chat.history`), and the user query.
        *   **Generate Response:** Calls `generate_with_provider` which dispatches to either Ollama (`_call_ollama_chat_completion`) or OpenAI (`_call_openai_chat_completion`) adapter based on config/request.
        *   **Stream Response:** Streams LLM response chunks back to the frontend via WebSocket. Sends source document info after generation completes.
        *   **Save History:** Saves user query and assistant response to the DB (`add_chat_message`).

## Key Technologies & Libraries

*   **Frontend:** React, TypeScript, Styled Components, Tremor, Axios, WebSocket API
*   **Backend:** Python, FastAPI, Uvicorn
*   **AI/ML:**
    *   `sentence-transformers`: Text Embeddings
    *   `transformers` (Hugging Face): Underlying models?
    *   `torch`: ML framework
    *   `spacy`: NLP tasks (metadata extraction - currently unused in main flow)
    *   `unstructured`: Document parsing
    *   `ollama`: Local LLM interaction
    *   `openai`: OpenAI API interaction
    *   `crewai`: Agent framework? (Listed in reqs, usage not obvious from core files read)
    *   `langdetect`: Language detection (unused in main flow)
*   **Databases:**
    *   Milvus: Vector Database
    *   SQLAlchemy (+ Alembic): Relational DB (likely PostgreSQL or SQLite based on typical FastAPI setups)
*   **Storage:** Minio (S3 compatible)
*   **Auth:** PyJWT, bcrypt
*   **Deployment:** Docker, Docker Compose, Nginx (frontend proxy)

## Overall Vibe & Potential Next Steps

*   **Architecture:** Quite solid, modular backend design. FastAPI + Pydantic is a good stack. Use of singleton for embedding model is smart. Dynamic collection management in Milvus is robust.
*   **Complexity:** The AI/ML stack is non-trivial. Debugging RAG issues might require digging into prompts, retrieval results, and embedding quality.
*   **TODOs/Areas to Investigate:**
    *   How is `crewai` actually used?
    *   Confirm the specific relational DB (check `DATABASE_URL` in `.env.local` or `docker-compose.yml`).
    *   Dive into `slides.py` and `research.py` logic.
    *   Analyze frontend code (`frontend/src/`) to see UI implementation details and state management.
    *   Review error handling and logging coverage.
    *   Check if the commented-out metadata extraction (`ingest.py`) should be enabled.
    *   Assess test coverage (`backend/tests/`).

This codebase has potential. It's complex but seems well-structured in the backend core. Let's fucking go. 