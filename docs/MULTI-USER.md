# Multi-User & Authentication Implementation for VibeRAG

**Status: Completed (Initial Implementation)**

This document outlines the plan and execution for adding multi-user support, authentication, role-based access, per-user data isolation, and persistent storage to VibeRAG.

The initial plan envisioned using SQLite for simplicity, but during implementation, a decision was made to **upgrade to PostgreSQL** for better scalability, robustness, and handling of concurrent operations, especially regarding chat history and user management.

## Phase 1: Basic Authentication

- **Status:** Completed.
- **User Account Model & Storage:** A **User** model (username, hashed password, role, active status) was implemented using SQLAlchemy. **PostgreSQL** is now used as the database backend, managed via a dedicated service in `docker-compose.yml` with a persistent named volume (`postgres_data`). The default admin user (`admin`/`admin`) is created on first startup if no users exist. Passwords use Bcrypt hashing.
- **JWT Token Authentication:** Stateless JWT authentication is implemented. The `/api/v1/auth/login` endpoint verifies credentials against the Postgres DB and returns a signed JWT containing user ID and role.
- **API Endpoint Protection:** FastAPI dependencies (`Depends(get_current_user)`, `Depends(get_current_active_admin_user)`) enforce authentication and authorization on relevant API routes using the JWT.
- **Frontend Landing & Login UI:** A `Login.tsx` component handles user login. It stores the JWT in `localStorage` and uses an Axios interceptor (`frontend/src/config/axios.ts`) to attach the `Authorization: Bearer <token>` header to subsequent requests.
- **Configuration:** JWT secret (`JWT_SECRET_KEY`) and database connection (`DATABASE_URL`) are configured via `.env.local` and `docker-compose.yml`.

## Phase 2: Admin Role and User Management Interface

- **Status:** Completed.
- **Admin Role & Authorization:** Backend endpoints for user management are protected, requiring an authenticated admin user.
- **User Management Endpoints:** Admin APIs for listing (`GET /users`), creating (`POST /users`), updating (`PUT /users/{id}`), and deleting (`DELETE /users/{id}`) users are implemented.
- **Admin Web Interface:** An `AdminPanel.tsx` component provides a UI for admins to list, create, activate/deactivate, reset passwords for, and delete users. It also allows the admin to change their own password.
- **Feedback and Validation:** Basic success/error messages are displayed in the Admin Panel.

## Phase 3: Per-User Data Isolation in Milvus (Multi-Tenancy)

- **Status:** Completed.
- **Milvus Collections per User:** The system now uses separate Milvus collections: one for the admin (`admin`), one shared globally (`global`), and one private collection per user (`user_<user_id>`).
- **Lifecycle of Collections:** The `admin` and `global` collections are initialized on startup. User-specific collections (`user_<user_id>`) are created automatically when a user is created via the Admin Panel.
- **Document Ingestion Changes:** The `POST /upload` endpoint now directs documents to the correct collection based on the authenticated user. Admins currently upload to their personal `admin` collection (future enhancement: allow admin to upload to `global`).
- **File Storage:** Files are stored locally within the backend container's mapped volume (`/app/data/uploads/<username>/<filename>`). *Note: While this provides basic separation, a more robust solution might use dedicated object storage like MinIO, especially for larger scale.*
- **Document Listing and Management:** `GET /list` returns documents from the user's private collection and the `global` collection. Non-admins have read-only access to global documents. Admins can manage their own and global documents. Deletion (`DELETE /delete`) and metadata updates (`PUT /documents/.../metadata`) respect user boundaries and collection context.
- **Retrieval (Semantic Search) Adjustments:** Chat (`/api/v1/chat`) and Research (`/api/v1/research`) now perform semantic search across both the user's private collection *and* the `global` collection, combining results to provide comprehensive answers.

## Phase 4: Persistent Chat History per User (Postgres-backed)

- **Status:** Completed.
- **Database Schema:** The **PostgreSQL** database includes `chat_sessions` and `chat_messages` tables, linked to the `users` table.
- **Session Management Endpoints:** APIs (`POST /sessions`, `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}`) allow the frontend to manage chat sessions.
- **WebSocket Chat Integration:** The `/api/v1/ws/chat` endpoint now loads history from the specified `session_id`, uses it for conversational context with the LLM, and stores new user/assistant messages in the Postgres DB.
- **Frontend Adjustments:** `Chat.tsx` uses the session APIs to load history, associates messages with the active session ID, and handles session creation/selection via the history panel.
- **Multi-User Safety:** All database queries for sessions/messages are filtered by the authenticated `user_id`, ensuring users can only access their own chat history.
- **Cleanup:** Deleting a user via the Admin Panel now cascades to delete their associated chat sessions and messages from the Postgres database.

## Key Deviations from Initial Plan

1.  **Database Backend:** Switched from **SQLite** to **PostgreSQL**. This was done to provide better support for concurrent access (expected with multiple users and background tasks), improved scalability, and more robust data management features compared to file-based SQLite within a containerized environment.
2.  **File Storage:** Currently uses basic local volume mapping (`/app/data/uploads/<username>`) instead of potentially integrating with the MinIO service used by Milvus. This is simpler for now but could be revisited.
3.  **Admin Upload Target:** Admin uploads currently default to the admin's private collection (`admin`). The planned feature to allow admin to choose between uploading to personal vs. global collection was deferred for simplicity in this iteration.

*(Outcome:* VibeRAG now supports multiple users with secure authentication, role-based access control, isolated user data in Milvus, and persistent chat history stored in PostgreSQL. The core requirements for multi-user support have been implemented.)*