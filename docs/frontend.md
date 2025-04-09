# Frontend Documentation (vibeRAG)

This document provides an overview of the vibeRAG frontend application, its structure, components, and a user guide.

## 1. Overview

The frontend is a single-page application (SPA) built with **React** and **TypeScript**. It provides the user interface for interacting with the vibeRAG backend services.

- **Framework:** React (via `create-react-app`)
- **Language:** TypeScript
- **Styling:** CSS Modules (`*.module.css`), `styled-components`, global CSS (`src/styles/global.css`)
- **UI Components:** `@tremor/react`
- **State Management:** React Hooks (`useState`)
- **API Communication:** `axios`
- **Routing:** Internal tab-based navigation (managed in `App.tsx`)
- **Testing:** Jest, React Testing Library

## 2. Project Structure (`frontend/`)

```
frontend/
├── Dockerfile           # Docker configuration for frontend container
├── nginx.conf           # Nginx configuration for serving the app
├── package.json         # Project metadata and dependencies
├── yarn.lock            # Dependency lock file
├── tsconfig.json        # TypeScript compiler options
├── .gitignore           # Files ignored by Git
├── public/              # Static assets (index.html, favicons, etc.)
└── src/                 # Application source code
    ├── App.module.css   # Styles specific to App component
    ├── App.css          # Legacy/global styles for App (potentially)
    ├── App.tsx          # Main application component (layout, navigation)
    ├── App.test.tsx     # Tests for App component
    ├── index.css        # Global base styles
    ├── index.tsx        # Entry point of the React application
    ├── components/      # Reusable UI components
    │   ├── Chat.tsx
    │   ├── Chat.module.css
    │   ├── DocumentManager.tsx
    │   ├── DocumentManager.module.css
    │   ├── PresentationViewer.tsx
    │   ├── PresentationViewer.module.css
    │   ├── ResearchReport.tsx
    │   ├── ResearchReport.module.css
    │   ├── Config.tsx
    │   ├── Config.module.css
    │   ├── ModelSelector.tsx
    │   ├── ModelSelector.module.css
    │   ├── KnowledgeFilter.tsx
    │   └── KnowledgeFilter.module.css
    ├── config/          # Configuration files (if any) - currently seems empty/unused
    ├── styles/          # Global styles directory
    │   └── global.css
    ├── react-app-env.d.ts # Type declarations for create-react-app
    ├── reportWebVitals.ts # Web performance monitoring
    └── setupTests.ts    # Jest setup file
```

## 3. Core Components (`src/components/`)

The main application functionality is divided into these core components, managed by `App.tsx`:

-   **`Chat.tsx`**: Provides the chat interface for interacting with the RAG model. Allows users to ask questions and receive answers based on ingested documents or web search.
-   **`DocumentManager.tsx`**: Allows users to upload new documents, view existing documents, manage metadata/tags, and delete documents from the knowledge base.
-   **`PresentationViewer.tsx`**: Interface for generating slide-based presentations based on a user prompt and the knowledge base.
-   **`ResearchReport.tsx`**: Interface for generating structured research reports based on a query, potentially using web search.
-   **`Config.tsx`**: Allows users to view and potentially modify backend configurations (e.g., LLM settings, Milvus connection details).

Supporting components include:

-   **`ModelSelector.tsx`**: Likely used within other components to select the LLM model/provider.
-   **`KnowledgeFilter.tsx`**: Potentially used for filtering documents or search results based on tags or metadata.

## 4. Running the Frontend

**Prerequisites:**
- Node.js and Yarn (or npm)

**Setup:**
1.  **Navigate:** Change directory to `frontend/`.
2.  **Install Dependencies:** Run `yarn install` (or `npm install`). This includes installing `jspdf` which is required for presentation downloads.
3.  **Backend:** Ensure the backend server is running.
4.  **Development Server:** Run `yarn start` (or `npm run dev` or `npm start`) to launch the development server (usually on `http://localhost:3000`).
5.  **Build:** Run `yarn build` (or `npm run build`) to create a production-ready build in the `build/` directory.

## 5. User Guide

The vibeRAG frontend provides a unified interface for leveraging your knowledge base and AI capabilities. The main navigation is handled via the sidebar on the left.

### 5.1 Sidebar Navigation

-   **Logo & Title:** Displays the "vibeRAG" name and logo.
-   **Navigation Buttons:**
    -   **Chat:** Switches to the main chat interface.
    -   **Documents:** Opens the document management view.
    -   **Presentations:** Opens the presentation generation tool.
    -   **Research:** Opens the research report generation tool.
    -   **Config:** Shows the system configuration panel.
-   **Footer:** Displays application credits.

### 5.2 Chat (`Chat` Tab)

This is the primary interaction point for asking questions and getting answers from your knowledge.

-   **Input Area:** Type your question or query here.
-   **Send Button:** Submits your query to the backend.
-   **Chat History:** Displays the conversation history, including your prompts and the AI's responses.
-   **Sources:** Responses generated from your documents will often cite the source document chunks used.
-   **Options (Potential):**
    -   **Model Selection:** May allow choosing different LLM models or providers (e.g., Ollama, OpenAI).
    -   **Knowledge Only:** Toggle to restrict answers strictly to the ingested documents.
    -   **Use Web:** Toggle to allow the backend to incorporate web search results.
    -   **File Filter:** May allow scoping the chat to a specific uploaded document.

**How it Works:** When you send a query, the frontend sends it to the backend's `/chat` endpoint. The backend retrieves relevant information from the Milvus vector store (based on your documents), potentially performs a web search, and then uses an LLM to generate a response, streaming it back to the frontend.

### 5.3 Document Management (`Documents` Tab)

This section allows you to manage the knowledge base.

-   **Upload:**
    -   Click an "Upload" or "Add Document" button.
    -   Select a file (e.g., PDF, TXT) from your computer.
    -   Optionally add tags or metadata to help organize and filter the document later.
    -   Confirm the upload.
-   **Document List:** Displays all ingested documents. Information shown may include:
    -   Filename
    -   Upload date
    -   Tags/Metadata
    -   Status (e.g., "Processed", "Processing")
-   **Actions:** For each document, you might have options to:
    -   **View:** Open or preview the document (functionality might vary).
    -   **Delete:** Remove the document and its associated data from the system.
    -   **Edit Metadata:** Modify tags or metadata.

**How it Works:** Uploading sends the file and metadata to the `/upload` backend endpoint. The backend processes the document (splits into chunks, generates embeddings) and stores it in Minio and Milvus. Listing documents calls `/list`, and deleting calls `/delete/{filename}`.

### 5.4 Presentations (`Presentations` Tab)

Generate slide decks based on your knowledge.

-   **Prompt Input:** Enter a topic or instruction for the presentation (e.g., "Create a 5-slide presentation about the key findings in document X").
-   **Options (Potential):**
    -   **Number of Slides:** Specify the desired length.
    -   **Source Document:** Limit generation to a specific document.
    -   **Model Selection:** Choose the LLM model/provider.
-   **Generate Button:** Starts the presentation creation process.
-   **Viewer:** Displays the generated slides (title, bullet points). May include source citations.

**How it Works:** The frontend sends the prompt and options to the `/presentation` backend endpoint. The backend uses retrieval and an LLM to generate slide content based on the prompt and knowledge base.

### 5.5 Research (`Research` Tab)

Generate detailed reports on specific topics.

-   **Query Input:** Enter the research topic or question.
-   **Options (Potential):**
    -   **Use Web:** Toggle to enable/disable web search integration for broader context.
-   **Generate Button:** Initiates the report generation.
-   **Report Viewer:** Displays the structured report, which might include:
    -   Title
    -   Summary
    -   Key Insights
    -   Detailed Analysis
    -   Sources (documents and/or web links)

**How it Works:** The frontend sends the query and options to the `/research` backend endpoint. The backend likely uses a combination of knowledge base retrieval, web search (if enabled), and LLM generation (possibly using an agent framework like CrewAI) to compile the report.

### 5.6 Configuration (`Config` Tab)

View and potentially modify system settings.

-   **Display:** Shows current configuration values loaded by the backend (e.g., connected Milvus instance, default LLM models/providers, API keys status).
-   **Editing (Potential):** May allow modifying certain settings directly through the UI, which would update the backend configuration.

**How it Works:** This tab likely fetches data from the `/config` backend endpoint. If editing is enabled, it sends updates via a POST request to `/config`. It might also interact with provider-specific endpoints like `/providers/ollama/status`.

## 6. Styling and UI

-   Uses **CSS Modules** for component-scoped styles (e.g., `Chat.module.css`).
-   Uses **`@tremor/react`** for pre-built UI components (charts, cards, inputs, etc.), providing a consistent look and feel.
-   Global styles are defined in `src/index.css` and `src/styles/global.css`.
-   The overall theme aims for a modern, slightly "cyberpunk" aesthetic (as suggested by icon styling in `App.tsx`).

## 7. API Interaction

-   Uses `axios` to make asynchronous requests to the backend FastAPI server.
-   Endpoints targeted correspond to the features: `/chat`, `/upload`, `/list`, `/delete`, `/presentation`, `/research`, `/config`.
-   Handles streaming responses from the `/chat` endpoint for real-time updates. 