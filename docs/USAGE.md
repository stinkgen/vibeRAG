# VibeRAG - Usage Guide

This guide describes how to install, configure, and use the VibeRAG application from a user's perspective, based on the provided code and `README.md`.

## Installation & Setup (Docker Compose)

The primary method for running VibeRAG is using Docker and Docker Compose.

**Prerequisites:**

*   Docker & Docker Compose installed.
*   Git installed.

**Steps:**

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/stinkgen/vibeRAG.git
    cd vibeRAG
    ```

2.  **Configure Environment (`.env.local`):**
    *   Copy the template: `cp .env.example .env.local`
    *   Edit `.env.local`:
        *   **REQUIRED:** Provide `OPENAI_API_KEY`.
        *   **REQUIRED (for Web Search):** Provide `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID`.
        *   **Verify `OLLAMA_HOST` (if using Ollama):** Ensure it points to your Ollama instance (e.g., `http://host.docker.internal:11434` for host machine access from Docker).
        *   (Optional) Adjust `FRONTEND_PORT` (default: 3000) / `BACKEND_PORT` (default: 8000).
        *   (Optional) Configure Milvus connection (`MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION`) if deviating from defaults used in `docker-compose.yml`.
        *   (Optional) Set default chat models/providers (`CHAT_MODEL`, `CHAT_PROVIDER`), temperature (`CHAT_TEMPERATURE`), context limits (`CHAT_CHUNKS_LIMIT`).

3.  **Build and Run:**
    ```bash
    docker compose up --build -d
    ```
    *   Use `--build` initially and after code/dependency changes.
    *   Use `-d` for background execution.
    *   Allow time for services (especially Milvus) to initialize.

4.  **Access Application:**
    *   **UI:** `http://localhost:3000` (or custom `FRONTEND_PORT`).
    *   **API Docs:** `http://localhost:8000/docs` (or custom `BACKEND_PORT`/docs).

5.  **Stopping:**
    ```bash
    docker compose down
    ```
    *   Add `-v` to remove Milvus data volumes: `docker compose down -v`

6.  **Logs:**
    ```bash
    docker compose logs -f # All services
    docker compose logs -f backend # Backend logs
    docker compose logs -f frontend # Frontend Node.js proxy logs
    ```

## Application Features (UI)

Access the UI via `http://localhost:3000`. Navigation is via the sidebar.

**1. Chat (`💬 Chat`)**

*   **Functionality:** Send queries to an LLM, optionally augmented with context from ingested documents (RAG) and/or web search results.
*   **Interface (`Chat.tsx`):**
    *   **Input:** Type query in the text area.
    *   **Model Selection (`ModelSelector.tsx`):** Choose LLM Provider (OpenAI/Ollama) and Model from dropdowns.
    *   **Knowledge Filters (`KnowledgeFilter.tsx`):** Click the filter icon (+) to select specific Files, Collections (if implemented), or Tags to constrain the RAG context for the *next* message. Selected filters appear as badges.
    *   **Web Search Toggle:** Enable/disable augmentation with Google Search results (requires API keys).
    *   **Send:** Click the send button or press Enter.
    *   **Output:** Assistant responses stream into the chat pane. Responses using document context display numbered sources `[#]` below the message.
    *   **Sources:** Clicking a source number highlights the source information (filename, page number).
    *   **History:** Toggle the history panel (clock icon). Select previous chats to load them. Start new chats (+). Delete chats (trash icon). History is saved in browser `localStorage`.

**2. Documents (`📚 Documents`)**

*   **Functionality:** Manage the documents forming the knowledge base.
*   **Interface (`DocumentManager.tsx`):**
    *   **View:** Displays documents fetched from the backend `/list` endpoint. Toggle between Card and List views.
    *   **Search/Sort:** Filter documents using the search bar (matches filename, tags, metadata). Sort by columns in List view.
    *   **Upload:** Drag-and-drop or browse to select files (PDF, MD, TXT). Optionally add comma-separated tags and a single metadata key-value pair. Click "Upload Document" to send to the `/upload` endpoint.
    *   **Edit Metadata:** Click the pencil icon (List view) to open a modal. Modify tags (comma-separated) and add/remove/edit key-value metadata pairs. Saves changes via the `/documents/{filename}/metadata` PUT endpoint.
    *   **Delete:** Click the trashcan icon to remove the document via the `/delete/{filename}` endpoint.
    *   **Download:** Click the download icon (for PDFs) to fetch the original file via the `/get_pdf/{filename}` endpoint.

**3. Presentations (`🎨 Presentations`)**

*   **Functionality:** Generate presentation outlines based on a prompt and knowledge base context.
*   **Interface (`PresentationViewer.tsx`):**
    *   **Input:** Enter a prompt describing the desired presentation.
    *   **(Optional) Context:** Select a specific PDF file to provide focused context.
    *   **Parameters:** Set the desired number of slides. Select LLM provider/model (`ModelSelector.tsx`).
    *   **Generate:** Click "Generate Presentation" to send request to the `/presentation` endpoint.
    *   **Output:** Displays generated slides (title, bullet points). Lists source documents used.
    *   **Download:** Click "Download PDF" to attempt generation of a PDF using `jspdf` (may be unreliable).

**4. Research (`🔬 Research`)**

*   **Functionality:** Generate a structured research report based on a query, knowledge base context, and optional web search.
*   **Interface (`ResearchReport.tsx`):**
    *   **Input:** Enter the research query.
    *   **Web Search Toggle:** Enable/disable inclusion of Google Search results.
    *   **Generate:** Click "Start Research" to send request to the `/research` endpoint.
    *   **Output:** Displays the generated report with Title, Summary, Key Insights, Detailed Analysis, and Sources sections.
    *   **Sources:** Source filenames are clickable, attempting to open the corresponding PDF via the `/get_pdf/{filename}` endpoint.

**5. Config (`⚙️ Config`)**

*   **Functionality:** View and modify system configuration; check external provider status.
*   **Interface (`Config.tsx`):**
    *   **View Configuration:** Displays current settings fetched from `/config` (Chat defaults, OpenAI details, Ollama host/model, Milvus connection info).
    *   **Update Configuration:** Modify settings in the form and click "Save Configuration" to POST changes to `/config`. (API key updates require entering the key).
    *   **Provider Status:** Displays Ollama status (online/offline) and lists available models fetched from `/providers/ollama/status`. Fetches and lists available OpenAI models from `/providers/openai/models`.
    *   **Load Ollama Model:** Allows entering an Ollama model name and clicking "Load Model" to trigger a request to the `/providers/ollama/load` endpoint.

## Runtime Behavior

*   **Chat:** Uses WebSockets for real-time streaming responses.
*   **Document Processing:** Upload triggers backend processing (parsing, chunking, embedding, storage in Milvus).
*   **Generation Tasks:** Presentation and Research requests trigger backend workflows involving semantic search, LLM calls, and JSON parsing.
*   **State Persistence:** Chat history and selected models persist across browser sessions via `localStorage`. Document data persists in Milvus volumes (unless `docker compose down -v` is used). 