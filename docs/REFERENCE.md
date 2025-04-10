# VibeRAG - Technical Reference

This document provides a technical reference for VibeRAG, detailing backend API endpoints, data models, and key module functionalities based *only* on the analyzed source code.

## Backend API (FastAPI)

*   **Location:** `backend/src/api/app.py`
*   **Framework:** FastAPI
*   **Base URL:** `/api/v1` (relative to backend service URL)
*   **Authentication:** None.
*   **API Docs:** Auto-generated Swagger UI at `/docs`.

### Endpoints (Defined in `backend/src/api/app.py`)

**Health & Configuration**

*   `GET /health`: Returns `{"status": "healthy", "message": "..."}`.
*   `GET /config`: Returns current configuration values derived from `CONFIG` object (see `backend/src/modules/config/config.py`). Includes `chat`, `openai`, `ollama`, `milvus` sections.
*   `POST /config`: Updates the runtime `CONFIG` object based on the request body (`ConfigUpdate` model). Updates `os.environ["OPENAI_API_KEY"]` if provided.

**Document Management**

*   `POST /upload`: Handles document ingestion.
    *   Accepts `multipart/form-data` with `file`, `tags` (JSON string list), `metadata` (JSON string dict).
    *   Saves file to `storage/documents` volume.
    *   Calls `upload_document` from `ingestion/ingest.py` for processing (parsing, chunking, embedding, storing).
    *   Returns `UploadResponse` (filename, num_chunks, tags, metadata, status).
*   `GET /list`: Lists unique ingested documents.
    *   Queries Milvus for distinct filenames and associated metadata/tags.
    *   Returns `List[DocInfo]` (doc_id, filename, tags, metadata).
*   `DELETE /delete/{filename}`: Deletes a document.
    *   Calls `delete_document` from `vector_store/milvus_ops.py` to remove entries from Milvus.
    *   Deletes the original file from `storage/documents`.
    *   Returns `DeleteResponse` (success, message).
*   `PUT /documents/{filename}/metadata`: Updates metadata for a document.
    *   Accepts `MetadataUpdateRequest` body (optional `tags`, optional `metadata`).
    *   Calls `update_metadata_in_vector_store` from `vector_store/milvus_ops.py`.
    *   Returns `{"message": "..."}`.
*   `GET /get_pdf/{filename}`: Serves an original PDF file.
    *   Returns `FileResponse` from `storage/documents/{filename}`.

**Content Generation & Chat**

*   `WS /ws/chat`: Handles real-time RAG chat via WebSocket.
    *   Accepts initial JSON message with parameters (`query`, `filename`, `knowledge_only`, `use_web`, `model`, `provider`, `temperature`, `filters`, `chat_history_id`).
    *   Orchestrated by `chat_with_knowledge_ws` in `generation/generate.py`.
    *   Performs retrieval (`semantic_search`), optional web search (`google_search`), prompt construction, and calls `generate_with_provider`.
    *   Streams back JSON messages: `{ "type": "chunk", "data": "..." }`, `{ "type": "sources", "data": [...] }`, `{ "type": "error", "data": "..." }`, `{ "type": "end" }`.
*   `POST /presentation`: Generates a presentation outline.
    *   Accepts `PresentationRequest` body (prompt, filename, n_slides, model, provider).
    *   Calls `create_presentation` from `generation/slides.py`.
    *   Returns `PresentationResponse` (slides, sources).
*   `POST /research`: Generates a research report.
    *   Accepts `ResearchRequest` body (query, use_web).
    *   Calls `create_research_report` from `research/research.py`.
    *   Returns `ResearchResponse` (report object).

**LLM Provider Interaction**

*   `GET /providers/ollama/status`: Checks Ollama server status and lists models via Ollama API (`/api/tags`).
*   `POST /providers/ollama/load`: Triggers Ollama model pull/load via Ollama API (`/api/pull`). Accepts `model_name` form data.
*   `GET /providers/openai/models`: Lists available OpenAI models via OpenAI API (`client.models.list()`). Filters results for likely text generation models. Returns `all_models` list and `suggested_default` model ID.

### Core Data Models (Pydantic)

Defined in `backend/src/api/app.py` for request/response validation.

*   **Requests:**
    *   `ChatRequest` (For potential non-WS endpoint, not primary path)
    *   `PresentationRequest` (`prompt`, `filename?`, `n_slides?`, `model?`, `provider?`)
    *   `ResearchRequest` (`query`, `use_web?`)
    *   `MetadataUpdateRequest` (`tags?`, `metadata?`)
    *   `ConfigUpdate` (Mirrors `GET /config` structure)
*   **Responses:**
    *   `NonStreamChatResponse` (Not primary path)
    *   `Slide` (`title`, `content`: List[str])
    *   `PresentationResponse` (`slides`: List[`Slide`], `sources`: List[str])
    *   `ResearchReport` (`title`, `summary`, `insights`, `analysis`, `sources`)
    *   `ResearchResponse` (`report`: `ResearchReport`)
    *   `UploadResponse` (`filename`, `num_chunks`, `tags`, `metadata`, `status`)
    *   `DeleteResponse` (`success`, `message`)
    *   `DocInfo` (`doc_id`, `filename`, `tags`, `metadata`)
    *   Provider status/model responses (various structures, see endpoints).

## Key Backend Modules & Functionality

*   **`config/config.py`:** Defines dataclasses for configuration sections (e.g., `MilvusConfig`, `EmbeddingConfig`, `ChatConfig`). Loads values primarily from environment variables. Provides global `CONFIG` instance.
*   **`embedding/service.py` & `embed.py`:** Manages loading a single instance (`EmbeddingService` singleton) of a `SentenceTransformer` model (`all-MiniLM-L6-v2` default) onto the appropriate device (GPU/CPU). `embed_chunks` uses this service to encode text batches.
*   **`ingestion/ingest.py`:**
    *   `parse_document`: Uses `unstructured` library to parse files (PDF, HTML, MD, TXT) into elements. Extracts page numbers for PDFs.
    *   `extract_metadata`: Uses `spacy` and `langdetect` to find language, named entities, and keywords.
    *   `chunk_text`: Splits text based on token count using `transformers` tokenizer (`gpt2`), respecting configured `chunk_size` and `overlap`.
    *   `upload_document`: Orchestrates file saving, parsing, embedding, and storage.
*   **`vector_store/milvus_ops.py`:**
    *   Handles all `pymilvus` interactions.
    *   `init_collection`: Creates/loads the Milvus collection with a schema defined by `CONFIG.milvus.field_params` (includes `chunk_id` (PK), `doc_id`, `embedding`, `text`, `metadata` (JSON), `tags` (ARRAY), `filename`, `category`) and creates an HNSW index.
    *   `store_with_metadata`: Inserts chunk data, ensuring data lists match the schema order.
    *   `search_collection`: Performs vector similarity search using `collection.search`.
    *   `delete_document`: Deletes by filename using `collection.delete` with an expression.
    *   `update_metadata_in_vector_store`: Implements update via fetch-delete-insert (upsert) logic.
*   **`retrieval/search.py`:**
    *   `semantic_search`: Orchestrates vector search by getting query embedding and calling `search_collection`.
    *   `google_search`: Executes web search via Google Custom Search API.
*   **`generation/generate.py`:**
    *   `get_openai_client`: Lazily initializes `AsyncOpenAI` client.
    *   `generate_with_provider`: Routes generation to Ollama (HTTP) or OpenAI (library call, with completion fallback).
    *   `chat_with_knowledge_ws`: Core WebSocket handler, combining retrieval, prompt engineering, generation, and streaming.
*   **`generation/slides.py` & `research/research.py`:** Implement specific generation tasks (`create_presentation`, `create_research_report`) by performing retrieval, constructing detailed JSON-focused prompts, calling `generate_with_provider`, and parsing the output.

## Frontend Key Aspects

*   **`frontend/server.js`:** Crucial Node.js proxy layer. All FE-BE communication passes through it.
*   **`frontend/src/config/api.ts`:** Defines endpoint paths used by `axios` calls.
*   **State Management:** Primarily component-local state (`useState`, `useEffect`). Chat history uses `localStorage`.
*   **Component Communication:** Mostly props-based. `ModelSelector` and `KnowledgeFilter` are reusable components.

*(This reference is derived solely from the analyzed source code files.)* 