# Backend Documentation (vibeRAG)

This document provides an overview of the vibeRAG backend application, its architecture, modules, API endpoints, and configuration.

## 1. Overview

The backend is a Python application built with the **FastAPI** framework. It serves as the API layer, handling requests from the frontend, orchestrating interactions with the knowledge base (Milvus), performing document ingestion, and leveraging Large Language Models (LLMs) for generation tasks (chat, presentations, research).

- **Framework:** FastAPI
- **Language:** Python
- **Server:** Uvicorn (ASGI)
- **Key Dependencies:**
    - `fastapi`, `uvicorn`, `pydantic`: Core web framework & data validation.
    - `pymilvus`: Interacting with Milvus vector database.
    - `sentence-transformers`, `transformers`, `torch`, `spacy`: Text embedding and NLP tasks.
    - `unstructured`: Document parsing and loading.
    - `ollama`, `openai`, `anthropic`: LLM integrations.
    - `crewai`: Potential framework for agent-based research tasks.
    - `minio`: Object storage for documents.
    - `python-dotenv`: Environment variable management.
- **Database:** Milvus (Vector Database)
- **Storage:** Minio (or local filesystem, path seems commented out in `app.py`)

## 2. Project Structure (`backend/`)

```
backend/
├── Dockerfile           # Docker configuration for backend container
├── requirements.txt     # Python dependencies
├── storage/             # Default directory for storing uploaded documents/data (used by Minio?)
├── tests/               # Unit and integration tests
└── src/                 # Application source code
    ├── __init__.py
    ├── api/             # FastAPI application and endpoints
    │   ├── __init__.py
    │   ├── app.py       # Main FastAPI app, defines routes and middleware
    │   ├── services/    # Supporting services (currently seems empty/unused)
    │   └── docs/        # Potentially for API documentation generation?
    └── modules/         # Core logic modules
        ├── __init__.py
        ├── config/      # Configuration loading and management
        ├── embedding/   # Text embedding generation
        ├── generation/  # Text generation (chat, slides)
        ├── ingestion/   # Document ingestion pipeline (loading, splitting, embedding)
        ├── research/    # Research report generation logic
        ├── retrieval/   # Information retrieval from vector store
        └── vector_store/ # Interaction with Milvus
```

## 3. Architecture & Modules (`src/modules/`)

The backend follows a modular architecture:

- **`api/app.py`**: The entry point. Initializes FastAPI, sets up middleware (CORS), defines Pydantic models for request/response validation, and registers API routers/endpoints.
- **`modules/config`**: Loads configuration from environment variables (`.env.local`) and potentially YAML files. Provides a centralized `CONFIG` object.
- **`modules/ingestion` (`ingest.py`)**: Handles the document processing pipeline:
    - Receives uploaded files.
    - Uses libraries like `unstructured` to load document content.
    - Splits documents into manageable chunks.
    - Calls the `embedding` module to generate vector embeddings for each chunk.
    - Calls the `vector_store` module to store chunks and embeddings in Milvus.
    - Potentially stores the original file in Minio/storage.
- **`modules/embedding`**: Contains logic to generate vector embeddings for text chunks using models like `sentence-transformers`.
- **`modules/vector_store` (`milvus_ops.py`)**: Manages interactions with the Milvus vector database:
    - Establishes and manages connections.
    - Creates and initializes collections.
    - Inserts document chunks and their embeddings.
    - Performs vector similarity searches.
    - Deletes documents/chunks.
    - Lists stored documents.
- **`modules/retrieval`**: Implements the logic to query the `vector_store` based on user queries. Takes a query, generates its embedding (using `modules/embedding`), and performs a similarity search in Milvus to find relevant document chunks.
- **`modules/generation` (`generate.py`, `slides.py`)**: Uses LLMs to generate text:
    - **Chat (`generate.py:chat_with_knowledge`)**: Takes a user query and retrieved context (from `modules/retrieval`). Formats a prompt for an LLM (Ollama, OpenAI). Sends the prompt to the selected LLM provider and streams the response back. Can optionally incorporate web search results.
    - **Slides (`slides.py:create_presentation`)**: Takes a prompt and generates presentation content (titles, bullet points) using retrieval and an LLM.
- **`modules/research` (`research.py`)**: Orchestrates the generation of research reports:
    - May involve multiple steps: planning, information gathering (retrieval + web search), synthesis, formatting.
    - Potentially uses `crewai` for defining research agents and tasks.
    - Uses LLMs for analysis and generation.

## 4. API Endpoints (`src/api/app.py`)

The following RESTful endpoints are provided:

- **`GET /health`**: Returns a status message indicating the backend is operational.
- **`POST /chat`**: Accepts a query, streams back an LLM response based on knowledge base + optional web search.
    - *Request:* `ChatRequest` (query, filename?, knowledge_only?, use_web?, stream?, model?, provider?)
    - *Response:* `StreamingResponse` (SSE) or `NonStreamChatResponse`
- **`GET /chat`**: Alternative GET endpoint for chat (useful for simple clients/testing).
- **`POST /presentation`**: Generates presentation slides based on a prompt.
    - *Request:* `PresentationRequest` (prompt, filename?, n_slides?, model?, provider?)
    - *Response:* `PresentationResponse` (slides, sources)
- **`POST /research`**: Generates a structured research report.
    - *Request:* `ResearchRequest` (query, use_web?)
    - *Response:* `ResearchResponse` (report)
- **`POST /upload`**: Uploads a document, processes it, and adds it to the knowledge base.
    - *Request:* `File`, `tags` (form data), `metadata` (form data)
    - *Response:* `UploadResponse` (filename, num_chunks, status, etc.)
- **`DELETE /delete/{filename}`**: Deletes a document from the knowledge base (Milvus and potentially storage).
    - *Request:* Path parameter `filename`
    - *Response:* `DeleteResponse` (success, message)
- **`GET /list`**: Returns a list of documents currently in the knowledge base.
    - *Request:* None
    - *Response:* `List[DocInfo]` (doc_id, filename, tags, metadata)
- **`GET /config`**: Retrieves the current backend configuration.
    - *Request:* None
    - *Response:* Dictionary containing configuration sections (chat, llm, milvus, etc.)
- **`POST /config`**: Updates parts of the backend configuration.
    - *Request:* `ConfigUpdate` (dict with sections to update)
    - *Response:* Success/failure message.
- **`GET /providers/ollama/status`**: Checks the status of the Ollama service.
- **`POST /providers/ollama/load`**: Attempts to load a specific model into Ollama.
- **`GET /providers/openai/models`**: Lists available OpenAI models (requires API key).
- **`GET /get_pdf/{filename}`**: Serves a PDF file from storage (path seems currently incorrect/commented out).

## 5. Configuration

- Configuration is primarily managed through environment variables loaded from `.env.local` in the project root.
- The `modules/config/config.py` file likely defines default values and loads/parses environment variables into a structured `CONFIG` object (possibly using Pydantic).
- Key configurable areas include:
    - Milvus connection details (host, port)
    - LLM settings (default models, providers, API keys for OpenAI/Anthropic)
    - Chat behavior (default temperature)
    - Storage paths (e.g., for Minio or local files)
- The `/config` endpoints allow runtime viewing and potential modification of some settings.

## 6. Running the Backend

1.  **Installation:** Navigate to the `backend/` directory and run `pip install -r requirements.txt` (preferably in a virtual environment).
2.  **Environment:** Ensure a `.env.local` file exists in the project root with necessary configurations (Milvus host, LLM keys, etc.). See `.env.example`.
3.  **Dependencies:** Requires external services like Milvus and potentially Ollama to be running.
4.  **Development Server:** Run `uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000` from the `backend/` directory.
5.  **Docker:** Use `docker-compose up --build backend` (or similar, based on `docker-compose.yml`) to build and run the backend container. 