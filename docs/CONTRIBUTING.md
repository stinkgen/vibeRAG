# VibeRAG - Contributing Guide

This guide provides information for engineers looking to contribute to, extend, or modify the VibeRAG codebase, based *only* on the analyzed source code.

## Engineering Overview

VibeRAG is a Dockerized application with four main parts:

1.  **Frontend Container (`vibe-frontend`):** Runs a Node.js Express server (`frontend/server.js`) that serves the static React build (`frontend/build`) and acts as a mandatory proxy for all browser communication to the backend.
2.  **Backend Container (`vibe-backend`):** Runs a Python FastAPI application (`backend/src/api/app.py`) containing the core application logic, API endpoints, authentication, user/session management, and interfaces to external services.
3.  **PostgreSQL Database:** Stores all user accounts, roles, JWT secrets, chat sessions, and messages. Managed via Docker Compose with persistent volumes.
4.  **Milvus Cluster:** Vector database services (`standalone`, `etcd`, `minio`) defined in `docker-compose.yml`. Used for per-user and global document vector storage.

Key technologies:

*   **Frontend:** React, TypeScript, Vite (build), Express (proxy), `ws` (WebSocket proxy), `axios` (HTTP proxy), CSS Modules.
*   **Backend:** Python, FastAPI, Uvicorn, Pydantic, `pymilvus`, `openai`, `requests`, `sentence-transformers`, `unstructured`, `spacy`, `langdetect`, `transformers`.
*   **Database:** PostgreSQL (users, auth, chat, sessions), Milvus (vectorized documents).
*   **Orchestration:** Docker Compose.

## Development Workflow

1.  **Prerequisites:** Docker, Docker Compose, Git.
2.  **Clone:** `git clone https://github.com/stinkgen/vibeRAG.git && cd vibeRAG`
3.  **Environment:** `cp .env.example .env.local` and edit `.env.local` with necessary API keys (OpenAI required, Google for web search), JWT secret (`JWT_SECRET_KEY`), and Postgres connection (`DATABASE_URL`). You can also set admin credentials for first boot.
4.  **Run:** `docker compose up --build -d`

**Backend Development:**

*   Source code (`backend/src`) is mounted into the container.
*   Uvicorn runs with `--reload`, so changes to `.py` files in `backend/src` trigger automatic server restarts.
*   Monitor logs: `docker compose logs -f backend`.
*   Access API docs: `http://localhost:8000/docs` (or custom port). Most endpoints require JWT authentication; use the `/api/v1/auth/login` endpoint to obtain a token.

**Frontend Development:**

*   **Hot-reloading is NOT configured in Docker.**
*   Changes to `frontend/src` require rebuilding the image: `docker compose up --build -d` (after `docker compose down`).
*   Consider running the React dev server locally (`npm start` or `vite dev` in `frontend/`) and proxying API requests to the backend container (`http://localhost:8000`) for a faster development loop (requires local Node.js/npm setup).

## Code Structure & Key Modules

*   **`docker-compose.yml`:** Defines services, dependencies, ports, volumes, environment variables.
*   **`frontend/`**
    *   `Dockerfile`: Multi-stage build; builds React app, then copies into Node.js image.
    *   `server.js`: **Critical proxy layer.** Handles serving static files and forwarding API/WS requests.
    *   `src/`: React application source.
        *   `App.tsx`: Main layout and routing.
        *   `components/`: UI components (Chat, DocumentManager, Config, AdminPanel, etc.).
        *   `config/api.ts`: API endpoint definitions.
*   **`backend/`**
    *   `Dockerfile`: Defines Python environment, installs dependencies, sets entry point.
    *   `requirements.txt`: Python dependencies.
    *   `src/api/app.py`: FastAPI app definition, routes, Pydantic models, lifespan events.
    *   `src/modules/`: Core logic.
        *   `config/config.py`: Dataclass-based configuration from environment variables.
        *   `auth/`: JWT token creation/validation, user CRUD, role checks.
        *   `users/`: User management endpoints, admin-only APIs.
        *   `sessions/`: Chat session and message APIs, per-user isolation.
        *   `embedding/`: `service.py` (singleton loader for SentenceTransformer), `embed.py` (batch encoding).
        *   `generation/`: `generate.py` (LLM routing, WebSocket RAG orchestration), `slides.py`/`research.py` (specific task generation), `exceptions.py`.
        *   `ingestion/ingest.py`: Document parsing (`unstructured`), metadata extraction (`spacy`, `langdetect`), chunking (`transformers`), main upload workflow.
        *   `retrieval/search.py`: Semantic search (`sentence-transformers` + Milvus call), Google Search API call.
        *   `vector_store/milvus_ops.py`: All `pymilvus` interactions (connect, init, store, search, delete, update). Per-user and global collections.

## Extending the Application

*   **Adding File Types:** Modify `ingestion/ingest.py` (`parse_document`) to handle new types, potentially adding new parsing libraries.
*   **Changing Embedding Model:** Update `CONFIG.embedding.model_name` in `config/config.py` (and potentially `CONFIG.milvus.dim` if dimensions change). Ensure the model is compatible with `sentence-transformers`.
*   **Adding LLM Providers:** Modify `generation/generate.py` (`generate_with_provider`), add relevant configuration in `config/config.py`, and update frontend components (`ModelSelector.tsx`, `Config.tsx`).
*   **Improving Retrieval:** Enhance `retrieval/search.py` and `vector_store/milvus_ops.py` (e.g., implement hybrid search fully, add re-ranking, support advanced Milvus filtering expressions passed from frontend).
*   **New Generation Agents:** Create new modules similar to `generation/slides.py` or `research/research.py`, define prompts, add API endpoints in `api/app.py`, and create corresponding frontend components.
*   **Persistent Chat History:** Chat history is now stored per-user in PostgreSQL via session/message APIs. See `sessions/` and `users/` modules.
*   **Authentication & RBAC:** All sensitive endpoints require JWT auth. User/role management is handled via admin-only APIs and the Admin Panel. Extend `auth/` and `users/` modules for new roles or permissions.
*   **Frontend Hot-Reloading:** Reconfigure `frontend/Dockerfile` and `docker-compose.yml` to use Vite's development server and HMR within the container.

## Coding Conventions (Observed)

*   **Python:** Type hints used extensively. Logging is present. Comments vary from explanatory to informal. Follows general PEP 8 principles.
*   **TypeScript/React:** Functional components with hooks. CSS Modules for styling. Standard React/TypeScript patterns.

## Known Limitations (from Code/README)

*   **Frontend Hot-Reloading:** Not configured in Docker.
*   **PDF Generation (`jspdf`):** Mentioned as potentially flaky in README.
*   **Milvus Metadata Updates:** Uses a complex fetch-delete-insert pattern (`update_metadata_in_vector_store`).
*   **Hybrid Search:** Implementation in `retrieval/search.py` appears incomplete.
*   **Error Handling:** Varies in robustness across modules.
*   **RBAC Edge Cases:** If you add new endpoints, ensure proper role checks and JWT validation.