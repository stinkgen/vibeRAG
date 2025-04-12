# Frontend Chat Component Implementation Guide

This document outlines the best practices and patterns for implementing the React Chat component (`Chat.tsx`) in the VibeRAG frontend, based on debugging sessions and established fixes. The goal is to ensure stable WebSocket connections, reliable state management, and correct handling of API interactions for a seamless user experience.

## 1. Key Challenges Identified

Initial implementations faced several issues:

*   **WebSocket Instability:** Connections were closing prematurely, often triggered by component re-renders caused by state updates related to session or message fetching. The primary cause was often complex `useEffect` dependency arrays or automatic actions (like creating a new session) within WebSocket event handlers (`onopen`).
*   **State Management Complexity:** Intertwined state updates (e.g., setting active session ID triggering message fetching, which might trigger other effects) led to unpredictable re-renders and potential race conditions.
*   **API Call Authorization:** Initial setup sometimes failed to consistently attach the JWT `Authorization` header to standard HTTP requests (like fetching sessions or provider status) immediately after login, leading to 401 errors.
*   **Incorrect Validation:** Message sending logic sometimes blocked valid attempts or failed silently due to incorrect validation order or premature function returns.

## 2. Core State Management (`useState`)

*   **Messages:** `useState<ChatMessageData[]>([])` - Holds the messages for the *currently active* chat session. Cleared when the active session changes.
*   **Sessions:** `useState<ChatSessionData[]>([])` - Holds the list of available chat sessions for the user. Fetched on mount and potentially refreshed after creating/deleting sessions.
*   **Active Session ID:** `useState<number | null>(null)` - Stores the ID of the currently selected chat session. Crucial for fetching messages and sending new ones. Should only be updated by explicit user action (selecting a session, creating a new one).
*   **Input:** `useState<string>(\'\')` - The user's current message input.
*   **Loading States:** Use specific loading states (e.g., `isSessionLoading`, `isMessageLoading`, `isSending`) rather than a single generic `isLoading` to provide more granular feedback and avoid unnecessary loading indicators (e.g., don't show loading just because the WebSocket is connecting).
*   **Error State:** `useState<string | null>(null)` - Stores user-facing error messages (API errors, WS errors, validation errors). Clear the error when a subsequent action is successful.

## 3. API Call Management (`useEffect`, `useCallback`, Axios)

*   **Axios Interceptor:** The primary mechanism for adding the `Authorization: Bearer <token>` header should be configured *once* in a higher-level component (like `App.tsx`) using Axios interceptors. Use the `eject` method to remove the specific interceptor on logout or when the token becomes invalid, rather than `clear()`. Ensure the interceptor logic correctly adds the header.
*   **Fetching Sessions (`fetchSessions`):**
    *   Wrap the `axios.get(\'/api/v1/sessions\')` call in a `useCallback`.
    *   **Crucially, use an empty dependency array `[]` for this `useCallback`.** The function itself doesn't depend on component state like `activeSessionId`.
    *   Call this function once in a `useEffect` hook with an empty dependency array `[]` to fetch the initial list on component mount.
    *   Call it again *only* when necessary (e.g., after successfully creating or deleting a session via `handleNewSession` or `handleDeleteSession`).
*   **Fetching Messages (`fetchMessages`):**
    *   Wrap the `axios.get(\`/api/v1/sessions/\${sessionId}\`)` call in a `useCallback`.
    *   Use an empty dependency array `[]`. The function takes `sessionId` as an argument.
    *   Call this function from a `useEffect` hook that *depends only on `activeSessionId`*. This ensures messages are fetched *only* when the user selects a different session.
    ```typescript
    useEffect(() => {
        if (activeSessionId) {
            fetchMessages(activeSessionId);
        } else {
            setMessages([]); // Clear messages if no session is active
        }
    }, [activeSessionId, fetchMessages]); // fetchMessages is stable due to its empty dependency array
    ```
*   **Creating/Deleting Sessions:** These actions (e.g., `handleNewSession`, `handleDeleteSession`) should be triggered by user events (button clicks). After a successful API call (e.g., `axios.post('/api/v1/sessions')`), update the relevant state (`setActiveSessionId`) and potentially re-fetch the session list (`fetchSessions()`).

## 4. WebSocket Connection Management (`useEffect`, `useRef`, Event Handlers)

*   **Single Connection Lifecycle:** The WebSocket connection should ideally be established *once* when the `Chat` component mounts (assuming a valid auth token exists) and closed *only* when the component unmounts.
*   **`useRef` for Instance:** Store the WebSocket instance in a `useRef<WebSocket | null>(null)`. This provides a stable reference across re-renders without causing effects to re-run.
*   **Connection `useEffect`:**
    *   Use a `useEffect` hook with an **empty dependency array `[]`** to manage the connection lifecycle.
    *   Inside the effect, check for the auth token. If present, and if `webSocketRef.current` is null or closed, initiate the connection.
    *   Define the `onopen`, `onmessage`, `onerror`, and `onclose` handlers *inside* this `useEffect` or ensure they are stable callbacks (e.g., defined outside or using `useCallback` with stable dependencies).
    *   **Return a cleanup function** from this `useEffect` that explicitly closes the WebSocket connection (`webSocketRef.current?.close(1000, "Component unmounting")`) and sets the ref to `null`. This is critical to prevent stale connections or errors on unmount.
    ```typescript
    useEffect(() => {
        const token = localStorage.getItem('vibeRAG_authToken');
        if (token && (!webSocketRef.current || webSocketRef.current.readyState === WebSocket.CLOSED)) {
            // ... connection logic ...
            // ... assign handlers (onopen, onmessage, onerror, onclose) ...
             webSocketRef.current = ws;
        }

        return () => {
            // Cleanup logic
            if (webSocketRef.current) {
                webSocketRef.current.close(1000, "Component unmounting");
                webSocketRef.current = null;
            }
        };
    }, []); // Empty dependency array ensures this runs only on mount/unmount
    ```
*   **`onopen`:** Log success, reset reconnect attempts. **Do NOT automatically trigger state updates** like creating a new session here, as this can cause re-renders that interfere with the connection stability.
*   **`onmessage`:** Parse the message. Use a `switch` statement for different message types (`chunk`, `sources`, `error`, `session_id`, `end`). Update relevant state (`setMessages`, `setError`, `setActiveSessionId`). Be careful if state updates here might trigger other effects unintentionally. When handling `session_id`, only update state if the received ID is different from the current `activeSessionId`.
*   **`onerror`:** Log the error, set an error message for the user.
*   **`onclose`:** Log the closure. Implement reconnect logic *only* for unexpected closures (e.g., `event.code !== 1000` and `event.reason !== "Component unmounting"`). Use a counter and delays (e.g., exponential backoff) to prevent infinite loops. Clear the `webSocketRef.current`.

## 5. Message Sending Logic (`sendMessage`)

*   **Validation First:** Perform checks in this order *before* attempting to send:
    1.  Check if input is empty (`!input.trim()`). If so, set error, return.
    2.  Check WebSocket readiness (`!webSocketRef.current || webSocketRef.current.readyState !== WebSocket.OPEN`). If not ready, set error, return.
    3.  Check if `activeSessionId` is set (`!activeSessionId`). If not, set error, return.
*   **Construct Payload:** Create the message object including `query`, `session_id`, and any other relevant parameters (`knowledge_only`, `model`, etc.).
*   **Update UI Immediately:** Add the user's message to the `messages` state *before* sending over WebSocket for optimistic UI update.
*   **Send:** Call `webSocketRef.current.send(JSON.stringify(messageToSend))` within a `try...catch` block.
*   **Clear Input & Set Loading:** Clear the input field and set loading state for the assistant's response.

## 6. Session Management

*   **Loading:** Fetch the session list on mount.
*   **Selection:** Allow users to click on a session in the list. This should update `activeSessionId`, triggering the `useEffect` that calls `fetchMessages`.
*   **Creation:** A "New Chat" button should trigger `handleNewSession`. This function calls `POST /api/v1/sessions`, and on success, updates `activeSessionId` with the new ID and potentially calls `fetchSessions` to refresh the list.
*   **Deletion:** A delete button per session calls `handleDeleteSession`. This calls `DELETE /api/v1/sessions/{id}`, and on success, calls `fetchSessions` to refresh the list and potentially clears `activeSessionId` if the deleted session was active.

## 7. Error Handling

*   Provide clear feedback to the user via the `error` state for:
    *   API call failures (fetching sessions/messages, creating/deleting sessions).
    *   WebSocket connection errors/unexpected closures.
    *   Message sending failures.
    *   Input validation errors.
*   Log detailed errors to the console for debugging.

## 8. Key Takeaways / Best Practices

*   **Stable Callbacks:** Use `useCallback` for functions passed as props or used in `useEffect` dependency arrays. Ensure the dependency arrays for `useCallback` are minimal and correct.
*   **Minimal `useEffect` Dependencies:** Keep `useEffect` dependency arrays as small as possible to avoid unnecessary re-runs. Empty arrays `[]` are crucial for mount/unmount logic like establishing the initial WebSocket connection.
*   **Decouple WS Connection from State Changes:** The WebSocket connection lifecycle should primarily depend only on component mount/unmount and the initial auth token, *not* on frequent state changes like `activeSessionId`.
*   **Explicit User Actions:** Avoid triggering complex state updates or API calls automatically within WebSocket handlers (`onopen`). Rely on explicit user actions (button clicks) to manage sessions.
*   **Clear Validation:** Implement clear, ordered validation checks before performing critical actions like sending messages.
*   **Specific Loading States:** Use multiple boolean flags for different loading states instead of one generic flag. 