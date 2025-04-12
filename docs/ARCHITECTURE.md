# VibeRAG - Architecture

This document details the architecture of the VibeRAG application, based on analysis of the source code.

## Component Breakdown

VibeRAG comprises four primary services orchestrated by Docker Compose: the Frontend (including a Node.js proxy), the Backend (FastAPI application), the PostgreSQL database (for user/auth/chat/session data), and the Milvus Vector Database cluster (for vectorized document storage).

```mermaid
graph LR
    User[User Browser] --> FEProxy{Frontend Container (Node.js Proxy + React UI)};
    FEProxy -- HTTP /api/v1/* --> Backend[Backend Container (FastAPI)];
    FEProxy -- WS /api/v1/ws/chat --> Backend;
    Backend -- TCP --> Milvus[(Milvus Cluster: standalone, etcd, minio)];
    Backend -- TCP --> Postgres[(PostgreSQL DB: users, sessions, chat, auth)];
    Backend -- HTTP --> OpenAI[OpenAI API];
    Backend -- HTTP --> Ollama[Ollama API];
    Backend -- HTTP --> GoogleSearch[Google Custom Search API];
```

**1. Frontend Container (`vibe-frontend`)**

*   **Build Process (`frontend/Dockerfile`):** Builds a production React application using Vite (`npm run build`).
*   **Runtime Process (`frontend/server.js`):** An Express (Node.js) server performs two key functions:
    1.  **Static File Serving:** Serves the built React application files (from `frontend/build`).
    2.  **Proxying:** Manually proxies requests from the user's browser to the backend container:
        *   **HTTP Proxy:** Forwards all requests starting with `/api` (using `axios`) to the backend service (`http://backend:8000`). This includes endpoints for document management, content generation, configuration, etc.
        *   **WebSocket Proxy:** Manually handles WebSocket upgrade requests for the specific path `/api/v1/ws/chat` (using the `ws` library). It establishes a connection to the backend WebSocket (`ws://backend:8000/api/v1/ws/chat`) and relays messages bidirectionally between the client and backend. Includes heartbeat handling (`ping`/`pong`) on both client and backend connections.
*   **React UI (`frontend/src/`)**
    *   **Technology:** React, TypeScript.
    *   **Core Components:**
        *   `App.tsx`: Root component, manages overall layout (sidebar, main content) and navigation between feature views.
        *   `Chat.tsx`: Implements the chat interface. Manages WebSocket connection state via `webSocketRef`, sends user queries and parameters, receives and renders streamed responses (text chunks, sources), handles chat history via persistent per-user sessions (PostgreSQL), manages knowledge filters (`KnowledgeFilter.tsx`), and allows model selection (`ModelSelector.tsx`). Requires JWT authentication for all chat/session APIs.
        *   `DocumentManager.tsx`: Provides UI for uploading files (PDF, TXT, MD), listing documents fetched from the `/list` endpoint, searching/sorting/filtering the list, deleting documents via the `/delete/{filename}` endpoint, and editing metadata via the `/documents/{filename}/metadata` endpoint.
        *   `PresentationViewer.tsx`: Allows users to input a prompt, optionally select a file context, choose model parameters (`ModelSelector.tsx`), and request presentation generation via the `/presentation` endpoint. Displays the resulting slides and sources. Includes `jspdf`-based PDF download functionality.
        *   `ResearchReport.tsx`: Allows users to input a query, toggle web search usage, and request a research report via the `/research` endpoint. Displays the structured report (title, summary, insights, analysis, sources).
        *   `Config.tsx`: Fetches current configuration from `/config`, allows modification of settings (chat defaults, provider details), displays provider status (`/providers/ollama/status`, `/providers/openai/models`), and allows triggering Ollama model loading (`/providers/ollama/load`).
        *   `KnowledgeFilter.tsx`: A reusable component (used in `Chat.tsx`) for selecting files/tags/collections to filter knowledge retrieval (fetches options from `/list`, `/tags`, `/collections` - though only `/list` seems fully implemented based on API).
        *   `ModelSelector.tsx`: A reusable component (used in `Chat.tsx`, `PresentationViewer.tsx`, `Config.tsx`) for selecting the LLM provider (OpenAI/Ollama) and specific model, fetching available models from the respective backend endpoints.
    *   **API Constants:** `src/config/api.ts` defines URLs for backend endpoints.

**2. Backend Container (`vibe-backend`)**

### Authentication & Multi-User Architecture

*   **User Model & Auth:** All API endpoints (except login) require JWT authentication. User accounts (username, hashed password, role, active status) are stored in PostgreSQL. The backend creates a default admin user on first startup if none exist.
*   **Role-Based Access Control:** Admin users can manage users via protected endpoints and the Admin Panel UI. Non-admins have access only to their own data.
*   **Session & Chat History:** Chat sessions and messages are stored per-user in PostgreSQL. All queries are filtered by user ID to enforce isolation.
*   **Per-User Data Isolation:** Each user has a private Milvus collection (`user_<user_id>`), with additional `admin` and `global` collections. File uploads and document management are routed to the correct collection and storage path based on the authenticated user.
*   **Admin Panel:** The frontend includes an Admin Panel for user CRUD, password resets, and role management.
*   **JWT Handling:** The frontend stores the JWT in localStorage and attaches it to all API requests via an Axios interceptor.

*   **Technology:** Python, FastAPI, Uvicorn, Pydantic.
*   **Runtime Process (`docker-compose.yml` command):** Runs the FastAPI application defined in `src.api.app:app` using `uvicorn` with `--reload` enabled.
*   **Core Application (`backend/src/api/app.py`)**
    *   Initializes the FastAPI app with lifespan management (`lifespan`) to ensure Milvus connection (`ensure_connection`) and collection initialization (`init_collection`) on startup.
    *   Configures CORS middleware to allow all origins.
    *   Defines Pydantic models for request validation and response serialization for all API endpoints.
    *   Mounts the `api_router` under the `/api/v1` prefix.
    *   Includes endpoints detailed in `REFERENCE.md`.
*   **Modules (`backend/src/modules/`)**
    *   `config/config.py`: Defines `dataclass`-based configuration objects (e.g., `ChatConfig`, `MilvusConfig`, `EmbeddingConfig`). Loads default values from environment variables (`os.getenv`). Provides a global `CONFIG` instance.
    *   `embedding/`: Handles text embedding.
        *   `service.py`: Implements `EmbeddingService` as a singleton to ensure the SentenceTransformer model (`CONFIG.embedding.model_name`) is loaded only once (`SentenceTransformer(model_name, device=device)`). Detects GPU availability (`torch.cuda.is_available`) to set the device (`cuda` or `cpu`).
        *   `embed.py`: Provides `embed_chunks`, which takes text chunks, retrieves the model via `get_embedding_model` (from `service.py`), encodes the chunks in batches using `model.encode`, and returns the chunks with added `embedding` fields.
    *   `generation/`: Handles LLM interactions.
        *   `generate.py`: Provides `generate_with_provider` which routes requests to either Ollama (via HTTP POST to `/api/generate`) or OpenAI (using the `openai` library's `AsyncOpenAI` client, handling both `chat.completions.create` and fallback to `completions.create`). Also contains `chat_with_knowledge_ws`, the core WebSocket handler that orchestrates retrieval (`semantic_search`), optional web search (`google_search`), prompt construction, calling `generate_with_provider`, and streaming results (`chunk`, `sources`, `error`, `end` messages) back via WebSocket.
        *   `slides.py`: Contains `create_presentation`, which performs semantic search for context, constructs a detailed JSON-focused prompt, calls `generate_with_provider`, parses the expected JSON output, and returns the slide structure.
        *   `exceptions.py`: Defines custom exceptions (`GenerationError`, etc.).
    *   `ingestion/ingest.py`: Handles document processing.
        *   Uses `unstructured` library (`partition`, `partition_pdf`) to parse various file types (PDF, HTML, MD, TXT).
        *   Uses `spacy` (`en_core_web_sm`) and `langdetect` for metadata extraction (language, entities, keywords) in `extract_metadata`.
        *   Uses `transformers.GPT2TokenizerFast` (`gpt2`) for token counting and text chunking (`chunk_text`) based on `CONFIG.ingestion.chunk_size` and `overlap`.
        *   `upload_document` orchestrates saving the file, calling `parse_document`, `embed_chunks`, and `store_with_metadata`.
    *   `research/research.py`: Contains `create_research_report`, which performs semantic search, optionally calls `google_search`, constructs a detailed JSON-focused prompt including context and web results, calls `generate_with_provider`, parses the JSON response, and returns the report structure.
    *   `retrieval/search.py`: Implements search functionalities.
        *   `semantic_search`: Takes a query, gets its embedding using `get_embeddings` (which calls `get_embedding_model`), and calls `search_collection` from `milvus_ops`.
        *   `keyword_search`: Performs basic keyword matching against text fetched via Milvus query (less sophisticated than semantic).
        *   `hybrid_search`: Combines scores from semantic and keyword results (implementation incomplete/potentially buggy based on code structure).
        *   `google_search`: Uses `requests` to call the Google Custom Search API (requires `GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`).
        *   `search_by_tag_list`, `search_by_metadata_field`: Wrappers around `milvus_ops` functions.
    *   `vector_store/milvus_ops.py`: Handles direct interaction with Milvus.
        *   Manages connection (`connect_milvus`, `ensure_connection`, `disconnect_milvus`).
        *   Initializes the collection (`init_collection`), defining the schema based on `CONFIG.milvus.field_params` (including fields like `chunk_id`, `doc_id`, `embedding`, `text`, `metadata` (JSON), `tags` (ARRAY), `filename`, `category`) and creating an HNSW index on the `embedding` field.
        *   `store_with_metadata`: Inserts batch data (embeddings, text, metadata, tags, etc.) into the collection.
        *   `delete_document`: Deletes entities based on a `filename` expression.
        *   `update_metadata_in_vector_store`: Fetches entities by filename, updates the `tags` list or `metadata` JSON field, and re-inserts (upserts) the modified entities (complex logic involving fetching, deleting, and re-inserting).
        *   `search_collection`: Performs the core vector search using `collection.search` with specified query vector, expression filters (`expr`), and search parameters.
        *   `search_by_tags`, `search_by_metadata`: Helper functions to construct Milvus expressions (`array_contains`, `json_contains`) and call `search_collection` or `collection.query`.

**3. Milvus Cluster (`docker-compose.yml` services)**

*   **Technology:** Milvus (`milvusdb/milvus:v2.3.3`), etcd (`quay.io/coreos/etcd:v3.5.5`), MinIO (`minio/minio:RELEASE.2023-03-20T20-16-18Z`).
*   **Role:** Provides the persistent vector storage and search capabilities.
*   **Services:**
    *   `standalone`: The main Milvus service.
    *   `etcd`: Metadata store for Milvus.
    *   `minio`: Object storage for Milvus data segments.
*   **Interaction:** The backend service (`vibe-backend`) connects to the `standalone` service on the host/port defined in `CONFIG.milvus` (defaults to `standalone:19530` within the Docker network).

## Data Flow Summary

*   **User Authentication:** Users must log in via the frontend login UI. Credentials are verified against PostgreSQL, and a JWT is issued on success.
*   **User Interaction:** Occurs via the React UI, proxied through the Node.js server in the frontend container. All API and WebSocket requests require a valid JWT.
*   **API Communication:** Standard HTTP request/response for most actions (upload, list, delete, config, generate presentation/research, user management, session management).
*   **Chat Communication:** Uses WebSocket connection, proxied by Node.js, allowing bidirectional streaming between `Chat.tsx` and `backend/src/modules/generation/generate.py:chat_with_knowledge_ws`. Chat history is persisted per-user in PostgreSQL.
*   **Knowledge Storage:** Documents are processed by the backend (`ingestion`), embeddings generated (`embedding`), and stored in per-user Milvus collections (`vector_store`). Files are stored in per-user directories on disk.
*   **Knowledge Retrieval:** Queries trigger semantic search (`retrieval`/`vector_store`) against the user's private and global Milvus collections.
*   **LLM Interaction:** Backend (`generation`) sends prompts (often including retrieved context) to external (OpenAI) or local (Ollama) LLM APIs.

## Key Design Points (Inferred from Code)

*   **Single Backend Service:** Despite the `modules` structure, the backend functions as a single FastAPI service rather than separate microservices.
*   **PostgreSQL as Core DB:** PostgreSQL is used for all user, authentication, role, and chat/session data. Milvus is used for vectorized document storage only.
*   **RBAC & Security:** All sensitive endpoints are protected by FastAPI dependencies enforcing JWT auth and role checks. Admin-only endpoints are strictly enforced.
*   **Per-User Data Isolation:** All document, chat, and session data is isolated per user. No cross-user access is possible at the API or DB level.
*   **Reliance on Config:** Extensive use of `CONFIG` object (`config/config.py`) loaded from environment variables drives behavior (model names, limits, hosts, ports, field names, index types, etc.).
*   **Embedding Model Singleton:** `embedding/service.py` ensures the SentenceTransformer model is loaded only once for efficiency.
*   **JSON for Complex Data:** Milvus `metadata` field stores combined chunk/document metadata as a JSON string. Presentation and Research generation rely heavily on prompting the LLM to return specific JSON structures, which are then parsed.
*   **Manual Proxying:** The Node.js server uses `axios` and `ws` directly for proxying, rather than standard proxy middleware libraries.
*   **Milvus Metadata Update:** Updates involve a read-delete-insert pattern (`update_metadata_in_vector_store`) rather than an in-place update, which might have performance implications.