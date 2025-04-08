# VibeRAG

A Retrieval-Augmented Generation (RAG) system with a cyberpunk UI for knowledge management, document analysis, and AI-powered chat.

![VibeRAG](https://img.shields.io/badge/VibeRAG-Cyberpunk%20RAG-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

VibeRAG combines the power of vector databases, LLMs, and semantic search to provide a unified interface for managing documents and interacting with your knowledge base. The system includes:

- Document ingestion and chunking pipeline
- Vector storage with Milvus
- Semantic, keyword, and hybrid search capabilities
- AI chat with sources/citations
- Knowledge filtering by documents, collections, and tags
- Web search integration
- Streaming responses for a responsive UI
- Chat history management
- Presentation generation

## Screenshots

(Add screenshots of your UI here)

## Installation & Running (Docker Compose - Recommended)

This method runs the entire application stack (backend, frontend, Milvus, MinIO, etcd) in Docker containers.

### Prerequisites

- Docker and Docker Compose
- Git (for cloning)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/vibeRAG.git # Replace with actual repo URL
    cd vibeRAG
    ```

2.  **Configure Environment Variables:**
    Copy the example environment file. You **must** edit this file to add your API keys (like `OPENAI_API_KEY` if needed) and can adjust ports if necessary.
    ```bash
    cp .env.example .env.local
    nano .env.local # Or your preferred editor
    ```
    *   The `BACKEND_PORT` and `FRONTEND_PORT` variables control which ports the application is exposed on your host machine.
    *   Other variables (API keys, Milvus/MinIO settings) are passed into the respective containers.

3.  **Build and Run with Docker Compose:**
    This command will build the backend and frontend images (if not already built) and start all services defined in `docker-compose.yml`.
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Forces Docker to rebuild the images (use this after code changes).
    *   `-d`: Runs the containers in detached mode (in the background).

4.  **Accessing the Application:**
    *   **Frontend:** Open your browser to `http://localhost:<FRONTEND_PORT>` (e.g., `http://localhost:3000` if you used the default port).
    *   **Backend API:** The API is accessible at `http://localhost:<BACKEND_PORT>` (e.g., `http://localhost:8000`).

5.  **Stopping the Application:**
    ```bash
    docker-compose down
    ```
    Use `docker-compose down -v` to also remove the data volumes (Milvus data, etc.).

6.  **Viewing Logs:**
    ```bash
    docker-compose logs -f # View logs for all services
    docker-compose logs -f backend # View logs for backend only
    docker-compose logs -f frontend # View logs for frontend only
    ```

## Development (Using Docker Compose)

For development, you can modify the `docker-compose.yml` file to mount your local source code into the containers. This allows for hot-reloading (changes reflect without rebuilding the image).

1.  **Uncomment Volume Mounts:** In `docker-compose.yml`, uncomment the volume mount lines in the `backend` service:
    ```yaml
    # backend service...
    volumes:
      # Optional: Mount source code for development with hot-reloading
      # Remove this for production builds
      - ./backend/src:/app/src # <--- Uncomment this line
      # Map the main volumes dir for potential file access/storage if needed
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes:/app/volumes
    ```
    *(Note: Frontend hot-reloading requires more complex Docker setup, often involving running `npm start` inside the container instead of serving static files. The current frontend Dockerfile is optimized for production builds.)*

2.  **Start with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
3.  **Backend Hot-Reloading:** Changes made to files in your local `./backend/src` directory should now trigger the `uvicorn` server inside the container to reload automatically (check `docker-compose logs -f backend`).

## Project Structure (Dockerized)

```
vibeRAG/
├── backend/
│   ├── src/              # Backend FastAPI source code
│   ├── Dockerfile        # Instructions to build backend image
│   └── requirements.txt  # Backend Python dependencies
├── frontend/
│   ├── public/
│   ├── src/              # Frontend React source code
│   ├── Dockerfile        # Instructions to build frontend image
│   └── package.json      # Frontend dependencies
├── .env.example          # Environment variables template
├── .env.local            # Local environment variables (used by docker-compose)
├── .gitignore
├── docker-compose.yml    # Defines all services (backend, frontend, db, etc.)
├── README.md
└── ...                   # Other config/script files if any
```

## License

MIT

## Acknowledgments

- This project uses [Milvus](https://milvus.io/) for vector storage
- UI inspired by cyberpunk aesthetics
- Built with FastAPI and React