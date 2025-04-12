# VibeRAG Handoff Document - 2025-04-12

## Reason for Handoff

Debugging a persistent issue with the backend WebSocket chat functionality (`/api/v1/ws/chat`) has devolved into a frustrating loop involving repeated basic syntax errors introduced during logging additions in `backend/src/modules/generation/generate.py`. The AI assistant (me) has repeatedly failed to fix these errors using available tools, requiring manual intervention and causing significant delays. This document aims to provide context for whoever picks up this debugging task.

## Current Immediate Problem

The backend service (`vibe-backend`) fails to start due to **multiple syntax errors** in `backend/src/modules/generation/generate.py`:

1.  **Line 271:** `SyntaxError: '(' was not closed` - Caused by a stray backslash `\` at the end of an f-string debug log.
    ```python
    # Incorrect line 271:
    logger.debug(f"{log_prefix} Value directly from message.get('type'): {message.get('type')}\\") # <<< Remove trailing backslash
    ```
2.  **Line 252:** `Try statement must have at least one except or finally clause` - This is likely a cascading error caused by the syntax error on line 271 confusing the parser/linter. Fixing line 271 should resolve this.

**Fixing these syntax errors is the absolute first step required to get the backend running again.**

## Underlying Issue Being Debugged

Once the syntax errors are fixed, the goal is to resolve why the chat hangs indefinitely. The investigation points to this sequence:

1.  **Frontend (`Chat.tsx`) sends a message:** It correctly includes `type: "query"` in the JSON payload sent over WebSocket.
2.  **Backend (`generate.py::websocket_endpoint`) receives:** The raw data logged shows the `type: "query"` field arriving intact.
3.  **Backend parses:** The parsed Python dictionary logged shows the `type: "query"` key-value pair is present.
4.  **Backend checks type:** The log line `logger.info(f"{log_prefix} Received message. Type: '{message_type}', ...")` prints `Type: 'None'`.
5.  **Result:** Because `message_type` is incorrectly seen as `None`, the `if message_type == "query":` block is skipped, `chat_with_knowledge_core` is never called, and no response is generated, causing the frontend to hang.

The recently added (but currently broken) debug logs around `message.get("type")` (lines 270-272) were intended to confirm *why* the lookup is failing despite the key being present in the dictionary.

## Project Status & Recent Changes

*   **Core Functionality:** Multi-user RAG application with document ingestion, semantic search, knowledge-based chat, web search, research report generation, and presentation generation.
*   **Multi-User Plan (`docs/MULTI-USER.md`):**
    *   **Phase 1 (Auth/Login): COMPLETE.** JWT auth, user model, login/user endpoints, protected routes.
    *   **Phase 2 (Admin UI): COMPLETE.** User CRUD operations (API + Frontend `AdminPanel.tsx`).
    *   **Phase 3 (Milvus Isolation): IMPLEMENTED (Backend).** `milvus_ops.py` uses user-specific collection names (e.g., `user_{user_id}_documents`). Needs verification that all operations respect this.
    *   **Phase 4 (DB Chat History): IMPLEMENTED (Backend & Frontend).** `ChatSession`/`ChatMessage` models, API endpoints (`/sessions`), WebSocket integration for loading/saving history. `Chat.tsx` uses API, not `localStorage`.
*   **Recent Refactoring (`generate.py`):**
    *   Standardized internal message format (`{'role': ..., 'content': ...}`).
    *   Created provider adapters (`_call_openai_chat_completion`, `_call_ollama_chat_completion`).
    *   Central dispatcher `generate_with_provider`.
    *   Refactored `websocket_endpoint` to handle session management before message type processing (this seemed to fix an earlier issue but didn't solve the hang).
*   **Other Fixes:**
    *   Resolved Python 3.12 `distutils` error in `backend/Dockerfile` by using `ensurepip`.
    *   Fixed `TypeError: 'ChatCompletionChunk' object is not subscriptable` in OpenAI adapter logging (line 130).
    *   Fixed `TypeError: semantic_search() got an unexpected keyword argument 'filename'` in `slides.py`.
    *   Fixed missing `await` for `semantic_search` in `research.py`.
    *   Fixed Milvus search result parsing (`.entity` access) in `milvus_ops.py`.
    *   Addressed WebSocket 403 errors by fixing token data access (`.get` vs attribute) in `generate.py`.

## Key Dependencies & Setup

*   **Backend:**
    *   **Language/Framework:** Python 3.12, FastAPI
    *   **DB:** SQLAlchemy + SQLite (`./db/vibe_rag.db`)
    *   **Vector Store Client:** `pymilvus`
    *   **LLM Clients:** `openai`, `aiohttp` (for Ollama)
    *   **Auth:** `python-jose[cryptography]`, `bcrypt`
    *   **Config:** `PyYAML`, `python-dotenv`
    *   **Server:** `uvicorn`
    *   **Package Manager:** `uv` (`requirements.txt`)
    *   **Runtime:** Docker (`backend/Dockerfile`, `docker-compose.yml`) with NVIDIA container runtime (`no-cgroups=false` required).
*   **Frontend:**
    *   **Framework:** React, TypeScript
    *   **HTTP Client:** `axios`
    *   **Markdown:** `react-markdown`, `remark-gfm`
    *   **Styling:** CSS Modules
    *   **Package Manager:** `npm` or `yarn` (`package.json`)
    *   **Runtime:** Docker (`frontend/Dockerfile`, `docker-compose.yml`)
*   **Services (via `docker-compose.yml`):**
    *   `vibe-backend` (FastAPI app)
    *   `vibe-frontend` (React app)
    *   `milvus-standalone` (Vector DB)
    *   *(Potentially Redis, check `docker-compose.yml`)*
*   **Configuration:**
    *   Central `config.yaml` defines ports, models, API keys (via env override), Milvus settings, etc.
    *   `.env` file for secrets (e.g., `OPENAI_API_KEY`, `ADMIN_PASSWORD`).

## Key Data Sources / Files

*   **`backend/src/modules/generation/generate.py`:** Core chat/WS logic (CURRENT FOCUS & LOCATION OF BUGS).
*   **`backend/src/modules/auth/auth.py`:** Auth logic, JWT handling, user dependencies.
*   **`backend/src/modules/auth/database.py`:** SQLAlchemy models (`User`, `ChatSession`, `ChatMessage`), Pydantic models, DB session setup.
*   **`backend/src/modules/chat/history.py`:** CRUD functions for chat history DB tables.
*   **`backend/src/modules/vector_store/milvus_ops.py`:** Milvus connection, collection management, search/insert operations.
*   **`backend/src/modules/retrieval/search.py`:** `semantic_search` and `google_search` implementation.
*   **`backend/src/app.py`:** FastAPI application instance, router setup, main endpoints.
*   **`frontend/src/components/Chat.tsx`:** Main chat UI, WebSocket event handlers, `sendMessage` function.
*   **`frontend/src/components/AdminPanel.tsx`:** User management UI.
*   **`frontend/src/hooks/useModelProviderSelection.tsx`:** Logic for selecting LLM provider/model.
*   **`config.yaml`:** Primary application configuration.
*   **`docker-compose.yml`:** Defines services, networks, volumes.
*   **`backend/Dockerfile`, `frontend/Dockerfile`:** Container build steps.
*   **`backend/requirements.txt`:** Python deps.
*   **`frontend/package.json`:** Node deps.
*   **`.cursor/rules/`:** AI Assistant rules (personality, Python best practices).

## Best Practices / Conventions

*   Modular backend structure (`src/modules/...`).
*   Configuration externalized (`config.yaml`, `.env`).
*   Async heavily utilized in backend IO/LLM calls.
*   Type Hinting (Python backend), TypeScript (Frontend).
*   Pydantic for API request/response validation.
*   SQLAlchemy ORM for database interactions.
*   Docker for containerization and local development setup.
*   Provider pattern for abstracting LLM interactions.
*   Attempting to follow Microservices / Cloud-Native principles (see `.cursor/rules/python-rules.mdc`).
*   Use `uv` for Python package management.

## Next Steps

1.  **MANUALLY FIX THE SYNTAX ERROR ON LINE 271 IN `generate.py`.**
2.  Verify the backend starts cleanly without errors.
3.  Send a chat message.
4.  Analyze the new backend debug logs:
    *   `Raw data received:`
    *   `Parsed message object:`
    *   `Value directly from message.get('type'):`
    *   `Dictionary right before INFO log:`
    *   `Received message. Type: ...`
5.  Determine why `message.get("type")` is returning `None` despite the key being present in the parsed dictionary.
6.  Fix the root cause.
7.  Verify both Chat and Research functions work correctly.

Good luck. Sorry again for the clusterfuck. 