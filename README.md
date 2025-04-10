# ⚡ VibeRAG ⚡

A Retrieval-Augmented Generation (RAG) system packing a FastAPI backend, React/TypeScript frontend, and Milvus vector storage – all wrapped in a cyberpunk UI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Docker](https://img.shields.io/badge/Docker-Powered-blue?logo=docker)](https://www.docker.com/)

## Features

- **Document Ingestion:** Handles various formats, chunks them, generates embeddings.
- **Vector Storage:** Utilizes Milvus for efficient similarity search.
- **Hybrid Search:** Combines semantic and keyword search (where applicable).
- **AI Chat:** Context-aware chat powered by LLMs (OpenAI/Ollama) with RAG.
- **Source Linking:** Cites sources used in RAG responses.
- **Streaming:** Real-time response streaming via WebSockets.
- **Chat History:** Persistent, context-aware chat sessions.
- **Content Generation:**
    - Presentation slide generation.
    - Research report generation.
- **Web Search Integration:** Augments knowledge with real-time web results (Google Custom Search).
- **Configuration UI:** Manage LLM settings, API keys, and provider status.

## Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** TypeScript, React, Vite, `react-markdown`
- **Vector DB:** Milvus
- **LLM Providers:** OpenAI, Ollama
- **Containerization:** Docker, Docker Compose
- **Proxying:** Node.js/Express (Frontend container), `ws` library for WebSockets

## Running with Docker Compose (Recommended)

Fire up the entire stack – backend, frontend, Milvus cluster – with Docker.

**Prerequisites:**

- Docker & Docker Compose
- Git

**Setup:**

1.  **Clone:**
    ```bash
    git clone https://github.com/yourusername/vibeRAG.git # <- Replace with your repo URL
    cd vibeRAG
    ```

2.  **Configure Environment (`.env.local`):**
    Copy the template and **edit it**. You *must* add your `OPENAI_API_KEY` if using OpenAI. Configure `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` for web search. Adjust ports if defaults clash.
    ```bash
    cp .env.example .env.local
    nano .env.local # Or your editor of choice
    ```
    *   `OLLAMA_HOST` defaults to `http://host.docker.internal:11434` to connect to Ollama running on your *host* machine.

3.  **Build & Launch:**
    This builds the images and starts all services in the background.
    ```bash
    docker compose up --build -d
    ```
    *   `--build`: Use after code changes to rebuild images.
    *   `-d`: Detached mode.

4.  **Access:**
    *   **Frontend:** `http://localhost:<FRONTEND_PORT>` (Default: `http://localhost:3000`)
    *   **Backend API (Direct):** `http://localhost:<BACKEND_PORT>` (Check `.env.local` or `docker-compose.yml`, often 8000 or similar)

5.  **Shutdown:**
    ```bash
    docker compose down
    ```
    Add `-v` to nuke data volumes (Milvus data, etc.).

6.  **Logs:**
    ```bash
    docker compose logs -f          # Tail all logs
    docker compose logs -f backend  # Tail backend logs
    docker compose logs -f frontend # Tail frontend (Node.js proxy) logs
    ```

## Development Hot-Reloading (Backend)

The `docker-compose.yml` mounts `./backend/src` into the container. The `uvicorn` command uses `--reload`, so backend code changes should trigger an automatic server restart. Check backend logs to confirm.

*(Note: Frontend hot-reloading isn't configured in the current production-focused Dockerfile setup.)*

## Project Structure

```
vibeRAG/
├── backend/             # FastAPI Microservice
│   ├── src/             # Core source code
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/            # React Frontend & Node.js Proxy Server
│   ├── public/
│   ├── src/             # React source code
│   ├── Dockerfile       # Multi-stage build (React build + Node server)
│   ├── package.json
│   └── server.js        # Node.js/Express server (serves static files, proxies API/WS)
├── .env.example         # Environment variable template
├── .env.local           # YOUR local environment vars (Gitignored)
├── .gitignore
├── docker-compose.yml   # Service definitions & orchestration
├── README.md
└── ...
```

## License

MIT - Go wild.