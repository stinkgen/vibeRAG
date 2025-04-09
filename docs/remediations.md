# VibeRAG Remediation Plan

This document outlines recommended fixes, feature implementations, and improvements based on the implementation review conducted on [Date - please fill in]. It addresses discrepancies between expected functionality (per documentation and component intent) and the actual code implementation.

## 1. Critical Fixes

These issues significantly impact core functionality or represent fundamental inconsistencies.

### 1.1. Resolve Storage Logic Conflict

*   **Issue:** Backend API endpoints (`/upload`, `/delete`, `/get_pdf` in `app.py`) use local filesystem logic tied to the commented-out `STORAGE_DIR` variable, while the project likely intends to use Minio/S3 (dependency exists, local path is commented out).
*   **Impact:** `/get_pdf` is non-functional. `/upload` and `/delete` perform potentially redundant/incorrect local file operations alongside Milvus/ingestion logic which presumably interacts with object storage.
*   **Location:** `backend/src/api/app.py` (endpoints: `upload_file`, `delete_file`, `get_pdf`), potentially `backend/src/modules/ingestion/ingest.py` (needs check for Minio interaction).
*   **Recommendation:**
    1.  **Confirm Minio Usage:** Verify that `upload_document` (in `ingestion` module) correctly uploads the original file content to Minio/S3 based on `.env.local` configuration.
    2.  **Refactor `/upload`:** Remove the local file save (`temp_file_path.open("wb")`) from the `/upload` endpoint in `app.py`. Ensure the file content (`BytesIO`) is passed correctly to `upload_document`.
    3.  **Refactor `/delete`:** Remove the local file deletion (`file_path.unlink()`) from the `/delete` endpoint in `app.py`. Add logic to delete the corresponding object from Minio/S3 when a document is deleted (potentially within `delete_document` function in `vector_store` or `ingestion` module).
    4.  **Implement `/get_pdf` for Minio:** Rewrite the `get_pdf` endpoint in `app.py` to fetch the specified file from Minio/S3 and return it, possibly using a streaming response or generating a pre-signed URL (depending on security requirements and Minio client library capabilities).
    5.  **Remove `STORAGE_DIR`:** Completely remove the commented-out `STORAGE_DIR` variable and associated logic from `app.py`.

## 2. Missing Features

These are features described in the user guide or implied by component structure that are not implemented.

### 2.1. Document Content View

*   **Issue:** The `DocumentManager.tsx` component lacks functionality to view or preview the content of uploaded documents.
*   **Impact:** Users cannot easily inspect the documents they have uploaded.
*   **Location:** `frontend/src/components/DocumentManager.tsx`.
*   **Recommendation:** Implement a "View" action for documents. This likely requires:
    *   Adding a view button/link to the document card/list item.
    *   When clicked, fetch the document from the backend (using the corrected `/get_pdf/{filename}` endpoint).
    *   Display the PDF content, potentially using a library like `react-pdf` or by opening it in a new tab/modal iframe.

### 2.2. Document Metadata Editing

*   **Issue:** The `DocumentManager.tsx` component does not allow editing tags or metadata after a document is uploaded.
*   **Impact:** Users cannot correct or update organizational information for existing documents.
*   **Location:** `frontend/src/components/DocumentManager.tsx`, `backend/src/api/app.py`.
*   **Recommendation:**
    1.  **Backend:** Create a new backend endpoint (e.g., `PUT /documents/{filename}/metadata`) that accepts updated tags and metadata and updates the corresponding entries in Milvus.
    2.  **Frontend:** Add an "Edit" action to `DocumentManager.tsx`. This could open a modal displaying the current filename, tags, and metadata, allowing modification and submission to the new backend endpoint. Refresh the document list upon successful update.

### 2.3. Presentation - Specify Number of Slides

*   **Issue:** The `PresentationViewer.tsx` component does not have a UI element to specify the desired number of slides, despite the backend API (`/presentation`) supporting the `n_slides` parameter.
*   **Impact:** Users cannot control the length of the generated presentation.
*   **Location:** `frontend/src/components/PresentationViewer.tsx`.
*   **Recommendation:** Add a number input field to the presentation generation form in `PresentationViewer.tsx`. Pass the value from this input as the `n_slides` parameter in the API call within the `handleSubmit` function.

### 2.4. Presentation - Download as PDF

*   **Issue:** The `PresentationViewer.tsx` component has an incomplete/non-functional `downloadPDF` function.
*   **Impact:** Users cannot download the generated presentation slides.
*   **Location:** `frontend/src/components/PresentationViewer.tsx`, potentially requires a new backend endpoint or library.
*   **Recommendation:** Choose an implementation strategy:
    *   **Option A (Backend Generation):** Create a new backend endpoint (e.g., `POST /presentation/download`) that takes the generated slide data (or the original prompt/parameters) and uses a library (like `reportlab`, `python-pptx` then convert, or a headless browser) to generate a PDF/PPTX file and return it.
    *   **Option B (Frontend Generation):** Use a frontend library (like `jspdf`, `pptxgenjs`) within `PresentationViewer.tsx`'s `downloadPDF` function to generate the PDF/PPTX directly from the displayed slide data (`presentation.slides`).
    *   Implement the chosen strategy, including adding a functional "Download" button.

## 3. Backend API Gaps

These are missing endpoints required by existing frontend components.

### 3.1. Missing `/collections` and `/tags` Endpoints

*   **Issue:** The `KnowledgeFilter.tsx` component attempts to fetch filter options from `/collections` and `/tags` endpoints, which are not defined in `backend/src/api/app.py`.
*   **Impact:** Filtering by collection or tag in the `KnowledgeFilter` component is non-functional.
*   **Location:** `backend/src/api/app.py`, `frontend/src/components/KnowledgeFilter.tsx`.
*   **Recommendation:**
    1.  **Decide on Scope:** Determine if filtering chat/search by arbitrary tags or predefined collections is a required feature.
    2.  **If Required:**
        *   Implement backend logic to efficiently retrieve unique tag names and collection names associated with documents stored in Milvus.
        *   Create `/tags` and `/collections` GET endpoints in `app.py` that return these lists.
        *   Modify backend search/chat logic (`semantic_search`, `chat_with_knowledge`) to accept and apply tag/collection filters during Milvus queries.
        *   Update `KnowledgeFilter.tsx` API calls if needed.
    3.  **If Not Required:**
        *   Remove the 'Collections' and 'Tags' options from the `filterTypeSelector` in `KnowledgeFilter.tsx`.
        *   Remove the corresponding API fetching logic from `KnowledgeFilter.tsx`.

## 4. Code Cleanup & Refactoring

### 4.1. Remove Unused `backend/src/api/services/` Directory

*   **Issue:** The `backend/src/api/services/` directory exists but is empty and unused.
*   **Impact:** Minor project clutter.
*   **Location:** `backend/src/api/services/`.
*   **Recommendation:** Delete the directory.

### 4.2. Unify Model Selection Logic (Optional)

*   **Issue:** `Chat.tsx` duplicates logic found in `ModelSelector.tsx` and `Config.tsx` for fetching provider status and listing models.
*   **Impact:** Code duplication.
*   **Location:** `frontend/src/components/Chat.tsx`, `frontend/src/components/ModelSelector.tsx`, `frontend/src/components/Config.tsx`.
*   **Recommendation:** Refactor `Chat.tsx` to utilize the `ModelSelector.tsx` component for consistency, removing the duplicated API calls and state management for model/provider selection within `Chat.tsx` itself.

### 4.3. Review Frontend API Endpoint Definitions

*   **Issue:** Frontend components reference API endpoints via `../config/api`. The structure/existence of this config file wasn't explicitly reviewed.
*   **Impact:** Potential for inconsistencies if endpoints change.
*   **Location:** `frontend/src/config/api.ts` (presumably), all components making API calls.
*   **Recommendation:** Ensure `frontend/src/config/api.ts` exists and accurately defines constants for all backend API endpoints used by the frontend components. Centralizing these makes future updates easier. 