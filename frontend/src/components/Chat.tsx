import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import styles from './Chat.module.css';
import { ModelSelector } from './ModelSelector';
import KnowledgeFilter from './KnowledgeFilter';
import API_ENDPOINTS from '../config/api';
import { useModelProviderSelection } from '../hooks/useModelProviderSelection';

// Types from database.py Pydantic models
interface ChatMessageData {
    id?: number; // Optional for messages not yet saved
    sender: 'user' | 'assistant';
    content: string;
    timestamp?: string; // String representation
    sources?: string[]; // Keep sources client-side for now
}

interface ChatSessionData {
    id: number;
    title: string;
    user_id: number;
    created_at: string;
    last_updated_at: string;
}

// Define component props
interface ChatProps {
    isAuthReady: boolean;
}

function Chat({ isAuthReady }: ChatProps) {
    const [messages, setMessages] = useState<ChatMessageData[]>([]);
    const [input, setInput] = useState('');
    const {
        currentModel,
        currentProvider,
        loadingStatus: modelLoadingStatus,
        providerError: modelProviderError
    } = useModelProviderSelection(isAuthReady);
    const [knowledgeOnly, setKnowledgeOnly] = useState<boolean>(true);
    const [useWeb, setUseWeb] = useState<boolean>(false);
    const [streamMode, setStreamMode] = useState<boolean>(true);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [showSources, setShowSources] = useState<{ [key: number]: boolean }>({});
    const webSocketRef = useRef<WebSocket | null>(null);
    const messagesEndRef = useRef<HTMLDivElement | null>(null);
    const reconnectAttemptRef = useRef<number>(0);
    const maxReconnectAttempts = 5;
    const reconnectDelay = 3000; // 3 seconds
    
    // --- Session State ---
    const [sessions, setSessions] = useState<ChatSessionData[]>([]);
    const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
    const [isSessionLoading, setIsSessionLoading] = useState<boolean>(false);

    // State for history panel visibility
    const [isHistoryPanelOpen, setIsHistoryPanelOpen] = useState(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // --- Fetch User Sessions --- 
    const fetchSessions = useCallback(async () => {
        setIsSessionLoading(true);
        try {
            const response = await axios.get<ChatSessionData[]>('/api/v1/sessions');
            setSessions(response.data);
            // If no active session or list is empty, maybe create/select one?
            // For now, just load the list.
            if (response.data.length > 0 && !activeSessionId) {
                // Optionally auto-select the most recent session
                // setActiveSessionId(response.data[0].id); 
            }
            setError(null); // Clear general errors on successful session fetch
        } catch (err) {
            console.error("Failed to fetch sessions:", err);
            if (axios.isAxiosError(err) && err.response?.status === 401) {
                setError("Unauthorized to fetch sessions.");
            } else {
                 setError("Failed to load chat sessions.");
            }
        } finally {
            setIsSessionLoading(false);
        }
    }, []);

    // --- Fetch Messages for Active Session --- 
    const fetchMessages = useCallback(async (sessionId: number) => {
        setIsLoading(true); 
        setMessages([]);
        try {
            const response = await axios.get<{ messages: ChatMessageData[] }>(`/api/v1/sessions/${sessionId}`);
            setMessages(response.data.messages || []);
            setError(null);
        } catch (err) {
            console.error(`Failed to fetch messages for session ${sessionId}:`, err);
            if (axios.isAxiosError(err) && err.response?.status === 401) {
                setError("Unauthorized to fetch messages.");
            } else {
                setError("Failed to load messages for this session.");
            }
            setMessages([]);
        } finally {
            setIsLoading(false);
        }
    }, []);
  
    // --- Effect to fetch sessions, depends on isAuthReady --- 
    useEffect(() => {
        // Only run if auth is ready
        if (isAuthReady) {
            console.log("Chat component received isAuthReady=true, fetching initial sessions...");
            const loadAndSelectSession = async () => {
                 // No timeout needed now
                setIsSessionLoading(true);
                try {
                    const response = await axios.get<ChatSessionData[]>('/api/v1/sessions');
                    const fetchedSessions = response.data;
                    setSessions(fetchedSessions);
                    console.log(`Fetched ${fetchedSessions.length} sessions.`);

                    let currentActiveId: number | null = null;
                    setActiveSessionId(id => { currentActiveId = id; return id; });

                    if (currentActiveId === null) {
                        if (fetchedSessions.length > 0) {
                            console.log(`Auto-selecting most recent session: ${fetchedSessions[0].id}`);
                            setActiveSessionId(fetchedSessions[0].id);
                        } else {
                            console.log("No sessions found, creating a new one automatically...");
                            await handleNewSession(); 
                        }
                    } else {
                         console.log(`Session ${currentActiveId} is already active, no auto-selection needed.`);
                    }
                    setError(null);
                } catch (err) {
                    console.error("Failed to fetch initial sessions:", err);
                    if (axios.isAxiosError(err) && err.response?.status === 401) {
                         setError("Unauthorized to load initial sessions.");
                    } else {
                         setError("Failed to load chat sessions.");
                    }
                } finally {
                    setIsSessionLoading(false);
                }
            };
            loadAndSelectSession();
        } else {
             console.log("Chat component: isAuthReady is false, delaying session fetch.");
             // Optionally reset state if auth becomes not ready (e.g., logout)
             setSessions([]);
             setActiveSessionId(null);
             setMessages([]);
        }
        
    // Dependency array now includes isAuthReady
    }, [isAuthReady]); // Removed fetchSessions from deps as it's stable

    // --- Effect to fetch messages, depends on isAuthReady and activeSessionId ---
    useEffect(() => {
        if (isAuthReady && activeSessionId) {
            console.log(`Chat component: isAuthReady is true & activeSessionId is ${activeSessionId}, fetching messages...`);
            fetchMessages(activeSessionId);
        }
         else {
             // Clear messages if session changes or auth not ready
             setMessages([]); 
         }
    // Add isAuthReady dependency
    }, [isAuthReady, activeSessionId, fetchMessages]);

    // --- WebSocket Connection Effect, depends on isAuthReady ---
    useEffect(() => {
        if (isAuthReady) {
            console.log("Chat component: isAuthReady is true, initiating WebSocket connection...");
            // Get token status on mount
            const token = localStorage.getItem('vibeRAG_authToken');
            if (token) {
                if (!webSocketRef.current || webSocketRef.current.readyState === WebSocket.CLOSED) {
                    console.log("useEffect initiating WebSocket connection...");
                    // Define connection logic directly or call a stable setup function
                    // For simplicity, duplicating the core logic here:
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsPath = '/api/v1/ws/chat';
                    const wsUrl = `${wsProtocol}//${window.location.host}${wsPath}?token=${encodeURIComponent(token)}`; 
                    
                    const ws = new WebSocket(wsUrl);
                    webSocketRef.current = ws;
                    
                    ws.onopen = () => {
                        console.log("WebSocket Connected (from useEffect)");
                        setError(null);
                        reconnectAttemptRef.current = 0; 
                    };
                    
                    ws.onmessage = (event) => {
                        try {
                            const message = JSON.parse(event.data);
                            console.log("WebSocket Message Received:", message);
                            switch (message.type) {
                               case 'session_id':
                                   console.log(`Received new session ID: ${message.session_id}`);
                                   setActiveSessionId(message.session_id);
                                   // Optionally trigger a fetch for this new session's messages if needed
                                   // fetchMessagesForSession(message.session_id); 
                                   break;
                               case 'session_confirm':
                                   console.log(`WebSocket confirmed session ID: ${message.session_id}`);
                                   // Trust the backend confirmation and ensure frontend state matches.
                                   // No need to warn if they briefly differ during init.
                                   if (activeSessionId !== message.session_id) {
                                      setActiveSessionId(message.session_id);
                                   }
                                   break;
                               case 'chunk':
                                   setMessages(prev => {
                                       const lastMessage = prev[prev.length - 1];
                                       if (lastMessage && lastMessage.sender === 'assistant') {
                                           return [
                                               ...prev.slice(0, -1),
                                               { ...lastMessage, content: lastMessage.content + message.data }
                                           ];
                                       } else {
                                           return [
                                               ...prev,
                                               { sender: 'assistant', content: message.data, sources: [] }
                                           ];
                                       }
                                   });
                                   break;
                               case 'sources':
                                   setMessages(prev => {
                                       const lastMessage = prev[prev.length - 1];
                                       if (lastMessage && lastMessage.sender === 'assistant') {
                                           const currentSources = lastMessage.sources || [];
                                           const newSources = Array.isArray(message.data) ? message.data : [];
                                           return [
                                               ...prev.slice(0, -1),
                                               { ...lastMessage, sources: [...currentSources, ...newSources] }
                                           ];
                                       } 
                                       return prev;
                                   });
                                   break;
                               case 'web_results':
                                   console.log('Received web results:', message.data);
                                   // Consider adding a placeholder or modifying the system message display
                                   break;
                               case 'end':
                                   setIsLoading(false);
                                   console.log('Received end signal, processing complete.');
                                   break;
                               case 'error':
                                   setError(message.data || 'An unknown error occurred.');
                                   setIsLoading(false);
                                   break;
                               default:
                                   console.warn("Received unknown message type:", message.type);
                            }
                        } catch (e) {
                            console.error("Failed to parse WebSocket message or update state:", e);
                            setError("Error processing message from server.");
                        }
                    };
                    
                    ws.onerror = (event) => { /* ... existing onerror logic ... */
                        console.error("WebSocket Error:", event);
                        setError("WebSocket connection error."); 
                        setIsLoading(false);
                    };
                    
                    ws.onclose = (event) => { /* ... existing onclose logic ... */
                        console.log("WebSocket Closed:", event.code, event.reason);
                        setIsLoading(false);
                        const wasConnected = webSocketRef.current !== null; // Check before nulling
                        webSocketRef.current = null; 
                        if (event.code !== 1000 && event.reason !== "Component unmounting" && reconnectAttemptRef.current < maxReconnectAttempts) { 
                           reconnectAttemptRef.current++;
                           setError(`WebSocket closed unexpectedly. Reconnecting attempt ${reconnectAttemptRef.current}...`);
                           // Need a stable way to reconnect - maybe a separate function?
                           // For now, log that reconnect *should* happen
                            console.warn("Attempting reconnect via timeout (logic needs review)");
                           // setTimeout(connectWebSocket, reconnectDelay * reconnectAttemptRef.current); // Can't call useCallback version here
                        } else if (event.code !== 1000 && event.reason !== "Component unmounting") {
                           setError("WebSocket connection failed permanently. Please refresh or login again.");
                        }
                    };
                }
            }
        } else {
             console.log("Chat component: isAuthReady is false, delaying WebSocket connection.");
             // Ensure WS is closed if auth becomes not ready
             if (webSocketRef.current) {
                 console.log("Closing WebSocket due to isAuthReady becoming false.");
                 webSocketRef.current.close(1000, "Auth not ready");
                 webSocketRef.current = null;
             }
        }

        // Cleanup remains the same - runs when component unmounts OR isAuthReady changes
        return () => {
            reconnectAttemptRef.current = maxReconnectAttempts;
            if (webSocketRef.current) {
                console.log("Closing WebSocket connection due to component unmount/dependency change.");
                webSocketRef.current.close(1000, "Component unmounting or auth change");
                webSocketRef.current = null;
            }
        };
    // Add isAuthReady dependency
    }, [isAuthReady]); 

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = () => {
        // 1. Check for empty input
        if (!input.trim()) {
             setError("Input cannot be empty.");
             setInput(''); // Clear the whitespace input
             return; 
        }

        // 2. Check WebSocket state
        if (!webSocketRef.current || webSocketRef.current.readyState !== WebSocket.OPEN) {
            setError("WebSocket is not connected. Please wait or refresh.");
            // Don't try to auto-reconnect here, let the existing onclose logic handle it.
            return; 
        }
        
        // 3. Check for active session ID
        if (!activeSessionId) {
            setError("No active chat session selected. Cannot send message.");
            console.error("Attempted to send message without an active session ID.");
            // Potentially auto-create a session here if desired?
            // handleNewSession(); // Example: Auto-create if none selected
            return; 
        }

        // --- All checks passed, proceed --- 
        setError(null); // Clear any previous validation errors

        const messageToSend = {
            type: "query",
            query: input,
            session_id: activeSessionId,
            knowledge_only: knowledgeOnly,
            use_web: useWeb,
            provider: currentProvider,
            model: currentModel,
            // Add temperature if needed by backend
        };

        // Update UI immediately with the user's message
        setMessages(prev => [...prev, { sender: 'user', content: input }]);
        
        // Send the message object as a JSON string
        try {
             webSocketRef.current.send(JSON.stringify(messageToSend));
             console.log("Sent message object:", messageToSend);
        } catch (e) {
            console.error("Failed to send message:", e);
            setError("Failed to send message. WebSocket might be closed.");
             // Attempt to reconnect if sending fails, maybe?
            // connectWebSocket();
        }

        setInput(''); // Clear input after sending
        // setError(null); // Already cleared above
        setIsLoading(true); // Set loading state for assistant response
    };

    // Handler for Enter key press in input
    const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent newline
            if (!isLoading && input.trim()) {
                sendMessage();
            }
        }
    };

    const toggleSources = (index: number) => {
        setShowSources(prev => ({ ...prev, [index]: !prev[index] }));
    };

    // --- Session Management UI Handlers ---
    const handleNewSession = async () => {
        console.log("Creating new session...");
        setIsSessionLoading(true);
        setMessages([]);
        setActiveSessionId(null);
        try {
            const response = await axios.post<ChatSessionData>('/api/v1/sessions', { title: "New Chat" });
            setActiveSessionId(response.data.id);
            await fetchSessions(); // Await the refresh
            setError(null);
        } catch (err) {
             console.error("Failed to create new session:", err);
             setError("Failed to start a new chat session.");
        } finally {
             setIsSessionLoading(false);
        }
    };
    
    const handleSelectSession = (sessionId: number) => {
        if (sessionId !== activeSessionId) {
            setActiveSessionId(sessionId);
            // Messages will be fetched by the useEffect hook
        }
    };
    
    const handleDeleteSession = async (sessionId: number) => {
        if (!window.confirm("Are you sure you want to delete this chat session?")) return;
        try {
            await axios.delete(`/api/v1/sessions/${sessionId}`);
            const prevActiveId = activeSessionId;
            setSessions(prev => prev.filter(s => s.id !== sessionId));
            if (prevActiveId === sessionId) {
                setActiveSessionId(null);
                setMessages([]);
                // After deleting the active session, check if others remain and select the first
                 const remainingSessions = sessions.filter(s => s.id !== sessionId);
                 if (remainingSessions.length > 0) {
                     setActiveSessionId(remainingSessions[0].id);
                 } else {
                     // Or create a new one if none are left?
                     // await handleNewSession(); 
                 }
            }
        } catch (err) {
             console.error(`Failed to delete session ${sessionId}:`, err);
             setError("Failed to delete chat session.");
        }
    };

    // TODO: Add handleRenameSession if needed

    return (
        <div className={styles.container}> 
            {/* Header */} 
            <div className={styles.cyberHeader}> 
                <h1 className={styles.title}>AI Chat Interface</h1>
                 <div className={styles.headerActions}> 
                     <button onClick={() => handleNewSession()} className={styles.newChatButton} title="New Chat">+</button>
                     <button onClick={() => setIsHistoryPanelOpen(!isHistoryPanelOpen)} className={styles.historyButton} title="Chat History">H</button>
                 </div>
            </div>

             {/* History Panel */} 
             {isHistoryPanelOpen && (
                 <div className={styles.historyPanel}>
                    <div className={styles.historyHeader}> 
                         <h3>Chat Sessions</h3>
                         <button onClick={() => setIsHistoryPanelOpen(false)} className={styles.closeButton}>&times;</button>
                     </div>
                     <div className={styles.historyList}> 
                         {isSessionLoading ? (
                             <p>Loading...</p>
                         ) : sessions.length === 0 ? (
                             <p className={styles.noHistory}>No chat history yet.</p>
                         ) : (
                             sessions.map(session => (
                                 <div 
                                     key={session.id} 
                                     className={`${styles.historyItem} ${session.id === activeSessionId ? styles.active : ''}`} 
                                     onClick={() => { 
                                         handleSelectSession(session.id);
                                         setIsHistoryPanelOpen(false); // Close panel on selection
                                     }}
                                 >
                                     <div className={styles.historyTitle}>{session.title}</div>
                                     <div className={styles.historyMeta}>
                                         <span className={styles.historyDate}>{new Date(session.last_updated_at).toLocaleDateString()}</span>
                                         <button 
                                             onClick={(e) => { 
                                                 e.stopPropagation(); // Prevent selecting session
                                                 handleDeleteSession(session.id); 
                                             }} 
                                             className={styles.historyDeleteButton}
                                             title="Delete Session"
                                         >
                                             {/* Add Icon Later: <FaTrash /> */} X 
                                         </button>
                                     </div>
                                 </div>
                             ))
                         )}
                     </div>
                 </div>
             )}

            {/* System Panel (Controls) */} 
            <div className={styles.systemPanel}>
                <div className={styles.toggleGroup}>
                    <label className={styles.toggle}>
                        <input type="checkbox" checked={knowledgeOnly} onChange={(e) => setKnowledgeOnly(e.target.checked)} />
                        <span className={styles.toggleLabel}>Knowledge Only</span>
                    </label>
                    <label className={styles.toggle}>
                        <input type="checkbox" checked={useWeb} onChange={(e) => setUseWeb(e.target.checked)} />
                        <span className={styles.toggleLabel}>Web Search</span>
                    </label>
                    <label className={styles.toggle}>
                        <input type="checkbox" checked={streamMode} onChange={(e) => setStreamMode(e.target.checked)} disabled />
                        <span className={styles.toggleLabel}>Stream Mode</span>
                    </label>
                </div>
                <ModelSelector isAuthReady={isAuthReady} />
            </div>

            {/* Messages Container */} 
            <div className={styles.messagesContainer}>
                {messages.map((msg, index) => (
                    <div key={index} className={`${styles.messageRow} ${msg.sender === 'user' ? styles.user : styles.assistant}`}> 
                        <div className={`${styles.messageContent} ${msg.sender === 'user' ? styles.user : styles.assistant}`}> 
                             <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                            {msg.sender === 'assistant' && msg.sources && msg.sources.length > 0 && (
                                <div className={styles.sourcesContainer}>
                                    <button onClick={() => toggleSources(index)} className={styles.sourcesToggle}>
                                        {showSources[index] ? 'Hide Sources' : 'Show Sources'} ({msg.sources.length})
                                    </button>
                                    {showSources[index] && (
                                        <ul className={styles.sourcesList}>
                                            {msg.sources.map((source, srcIndex) => (
                                                <li key={srcIndex}>{source}</li>
                                            ))}
                                        </ul>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                 ))}
                 {isLoading && messages[messages.length - 1]?.sender === 'user' && (
                    <div className={`${styles.messageRow} ${styles.assistant} ${styles.pending}`}> 
                         <div className={`${styles.messageContent} ${styles.assistant}`}> 
                             <div className={styles.typingIndicator}> <span className={styles.typingDot}></span><span className={styles.typingDot}></span><span className={styles.typingDot}></span> </div>
                         </div>
                     </div>
                 )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */} 
             <div className={styles.inputArea}>
                 <textarea
                     className={styles.queryInput} 
                     value={input}
                     onChange={(e) => setInput(e.target.value)}
                     onKeyDown={handleKeyDown}
                     placeholder="Enter your query here..."
                     rows={1}
                     disabled={isLoading}
                 />
                 <button onClick={sendMessage} disabled={isLoading || !input.trim()} className={styles.sendButton}> 
                     Send
                 </button>
             </div>

             {/* Error Display */} 
             {error && <div className={styles.errorMessage}>{error}</div>}
        </div>
    );
}

export default Chat; 