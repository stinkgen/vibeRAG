# VibeRAG - Overview

## Project Purpose

VibeRAG is a Retrieval-Augmented Generation (RAG) application designed to allow users to interact with their own document-based knowledge base through an AI-powered chat interface and generate new content derived from that knowledge.

The system ingests user-uploaded documents (PDF, Markdown, TXT, HTML), processes them by chunking the text, generates vector embeddings using a SentenceTransformer model (`all-MiniLM-L6-v2` by default), and stores these embeddings along with metadata in a Milvus vector database. Users can then query this knowledge base via a chat interface, where the system retrieves relevant document chunks based on semantic similarity to the query and feeds them as context to a Large Language Model (LLM) (supporting OpenAI and Ollama providers) to generate an informed response.

Key functionalities include:

*   **Document Ingestion:** Processing and vectorizing uploaded documents.
*   **RAG Chat:** Conversational interaction with the knowledge base, featuring streaming responses and source attribution.
*   **Document Management:** Listing, deleting, and updating metadata (tags, custom key-value pairs) for ingested documents.
*   **Content Generation:** Creating presentation outlines (`slides.py`) and structured research reports (`research.py`) using LLMs informed by document context and optional web search results (via Google Custom Search).
*   **Configuration Management:** A UI (`Config.tsx`) and API endpoints for viewing and modifying LLM provider settings (API keys, hostnames, model selection), Milvus connection details, and other operational parameters (e.g., chunk size, embedding model).
*   **Provider Status Checks:** Functionality to check the status of Ollama and list available models from both Ollama and OpenAI.

The project emphasizes a containerized deployment using Docker Compose for managing the frontend, backend, and Milvus database services.

## High-Level Design

VibeRAG utilizes a multi-container architecture orchestrated by Docker Compose:

*   **Frontend (`frontend/`):** A React/TypeScript Single Page Application (SPA) responsible for the user interface. It includes components for chat (`Chat.tsx`), document management (`DocumentManager.tsx`), presentation generation/viewing (`PresentationViewer.tsx`), research report generation/viewing (`ResearchReport.tsx`), and configuration (`Config.tsx`). It communicates with the backend exclusively through a Node.js proxy server.
*   **Node.js Proxy (`frontend/server.js`):** An Express server running *within* the frontend container. It serves the static built React application files and acts as a mandatory intermediary, proxying all HTTP API calls and WebSocket connections from the browser to the backend service. This handles potential CORS issues and centralizes backend communication logic.
*   **Backend (`backend/`):** A Python application built with FastAPI, serving as the core logic engine. It exposes a RESTful API and a WebSocket endpoint. Responsibilities include:
    *   Handling API requests for document upload, deletion, listing, metadata updates, content generation (presentations, research), configuration management, and provider status checks.
    *   Managing the WebSocket connection for real-time chat, including receiving user queries, orchestrating the RAG pipeline, streaming LLM responses, and sending source information.
    *   Interfacing with the Milvus database (`vector_store/milvus_ops.py`) for storing and retrieving document vectors.
    *   Performing document parsing, chunking, and metadata extraction (`ingestion/ingest.py`).
    *   Generating text embeddings (`embedding/embed.py` using `embedding/service.py` singleton).
    *   Interacting with LLM providers (OpenAI, Ollama) (`generation/generate.py`).
    *   Performing semantic search (`retrieval/search.py`) and optional web searches.
*   **Milvus Cluster (`docker-compose.yml` services: `standalone`, `etcd`, `minio`):** The vector database system used to store and index document embeddings for efficient similarity search. It relies on `etcd` for metadata and `minio` for object storage.

**Core Architectural Principles:**

*   **Containerization (Docker):** Ensures consistent deployment and manages service dependencies.
*   **API-Driven:** Communication between frontend and backend relies on defined API endpoints and WebSocket messages.
*   **RAG Pipeline:** The central mechanism involves retrieving relevant document chunks (Retrieval) to provide context for LLM text generation (Augmented Generation).
*   **Modularity (Backend):** Logic is organized into distinct Python modules within `backend/src/modules/` (e.g., `ingestion`, `embedding`, `retrieval`, `generation`, `vector_store`).
*   **Chat History Persistence:** Chat history is persisted per-user in the PostgreSQL database via backend session/message APIs. The `Chat.tsx` component loads and manages chat sessions and messages for the authenticated user, ensuring multi-user isolation and server-side persistence.