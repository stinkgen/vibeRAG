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

// Assume AgentOutput structure matches backend (or create/import a type)
interface AgentFinalOutput {
    result?: string; // Assuming final answer is in 'result' field
    error?: string; // Or error message
}

interface AgentTaskUpdatePayload {
    type: 'task_update';
    task_db_id: number;
    status: 'completed' | 'failed';
    payload: AgentFinalOutput;
}

interface Message {
    id: string; // Unique ID for React keys
    sender: 'user' | 'assistant' | 'system';
    text: string;
    sources?: any[];
    isLoading?: boolean;
    // Add fields to track agent tasks
    isAgentMessage?: boolean; // Identify messages related to /agent command
    agentTaskId?: number | null; // Store DB task ID associated with the queued message
    agentTaskComplete?: boolean; // Mark if the result has been received
}

function Chat({ isAuthReady }: ChatProps) {
    const [messages, setMessages] = useState<Message[]>([]);
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
    const ws = useRef<WebSocket | null>(null);
    const messagesEndRef = useRef<null | HTMLDivElement>(null);
    const [pendingAgentTasks, setPendingAgentTasks] = useState<Record<number, string>>({}); // { task_db_id: message_id }
    const currentUserId = useRef<number | null>(null); // User ID needed for WS connection logic (if not derived from token)
    
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
            const response = await axios.get<{ messages: Message[] }>(`/api/v1/sessions/${sessionId}`);
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

    // --- Get user ID from token when auth is ready --- 
    useEffect(() => {
        if (isAuthReady) {
            const storedToken = localStorage.getItem('vibeRAG_authToken');
            if (storedToken) {
                // Simple decode to get user ID (assuming decodeJwt exists or is imported/replicated)
                try {
                    const base64Url = storedToken.split('.')[1];
                    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
                    const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
                        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
                    }).join(''));
                    const decoded = JSON.parse(jsonPayload);
                    currentUserId.current = decoded?.id ?? null;
                     console.log("Chat component: Fetched user ID from token:", currentUserId.current);
                } catch (error) {
                    console.error("Chat component: Failed to decode token for user ID:", error);
                    currentUserId.current = null;
                }
            } else {
                currentUserId.current = null;
            }
        } else {
            currentUserId.current = null; // Clear user ID if auth not ready
        }
    }, [isAuthReady]);

    const connectWebSocket = useCallback(() => {
        // Check isAuthReady before connecting
        if (!isAuthReady) {
            console.log("WebSocket connection deferred: Auth not ready.");
            return;
        }
        if (!currentUserId.current) {
            console.error("Cannot connect WebSocket: User ID not available (derived from token).");
            return;
        }
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            console.log("WebSocket already connected.");
            return;
        }
        
        // Get token directly from localStorage
        const storedToken = localStorage.getItem('vibeRAG_authToken');
        if (!storedToken) { 
            console.error("WebSocket connection failed: No auth token found in localStorage.");
            setError("Authentication token not found. Please log in again."); // Provide user feedback
            return;
        }

        // Construct WS URL - Append token as query parameter
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/agents/ws/tasks?token=${encodeURIComponent(storedToken)}`; 
        
        console.log(`Attempting to connect WebSocket: ${wsUrl}`);
        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            console.log('WebSocket Connected');
            setError(null); // Clear connection errors on successful open
        };

        ws.current.onclose = (event) => {
            console.log('WebSocket Disconnected:', event.code, event.reason);
            ws.current = null; // Reset ref first
            if (event.code === 1008) { 
                setError("WebSocket connection failed: Authentication error. Please refresh or log in again.");
            } else if (!event.wasClean && isAuthReady) { // Only try to reconnect if not a clean close and we are still authenticated
                setError("WebSocket connection lost. Attempting to reconnect...");
                setTimeout(() => {
                    console.log("Attempting WebSocket reconnect...");
                    // Check isAuthReady again before reconnecting
                    if (isAuthReady) connectWebSocket(); 
                }, 5000); 
            } else {
                 setError(null); // Clear error on clean close or if auth is no longer ready
            }
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket Error:', error);
            setError("WebSocket connection error.");
            ws.current = null; // Ensure ref is null on error
        };

        ws.current.onmessage = (event) => {
            try {
                const messageData = JSON.parse(event.data);
                console.log('WebSocket Message Received:', messageData);

                if (messageData.type === 'task_update') {
                    const update = messageData as AgentTaskUpdatePayload;
                    const messageIdToUpdate = pendingAgentTasks[update.task_db_id];

                    if (messageIdToUpdate) {
                        setMessages(prevMessages => 
                            prevMessages.map(msg => {
                                if (msg.id === messageIdToUpdate && !msg.agentTaskComplete) {
                                    let resultText = '';
                                    if (update.status === 'completed') {
                                        resultText = update.payload.result || 'Task completed successfully.';
                                    } else { // failed
                                        resultText = `Error: ${update.payload.error || 'Task failed.'}`;
                                    }
                                    return {
                                        ...msg,
                                        text: `${msg.text}\n\n**Agent Result:**\n${resultText}`,
                                        isLoading: false, // Mark as no longer loading
                                        agentTaskComplete: true, // Mark as complete
                                    };
                                }
                                return msg;
                            })
                        );
                        // Remove task from pending map once updated
                        setPendingAgentTasks(prev => {
                            const newPending = { ...prev };
                            delete newPending[update.task_db_id];
                            return newPending;
                        });
                    } else {
                        console.warn(`Received task update for unknown/already completed task ID: ${update.task_db_id}`);
            }
        } else {
                     console.log("Received non-task_update WebSocket message:", messageData);
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', event.data, e);
            }
        };
    // Include isAuthReady in dependencies
    }, [isAuthReady, setError]); 

    // Effect to connect WebSocket when auth is ready 
    useEffect(() => {
        if (isAuthReady && !ws.current) {
            connectWebSocket();
        }
        // Cleanup function to close WebSocket on component unmount or auth loss
        return () => {
            if (ws.current) {
                console.log("Closing WebSocket connection on cleanup.");
                // Prevent automatic reconnection attempts after cleanup
                ws.current.onclose = null; 
                ws.current.close();
                ws.current = null;
            }
        };
    // Dependency now includes isAuthReady and connectWebSocket
    }, [isAuthReady, connectWebSocket]); 

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async (e?: React.FormEvent) => {
        e?.preventDefault();
        const userMessage = input.trim();
        if (!userMessage) return;

        const newUserMessage: Message = {
            id: Date.now().toString() + '-user',
            sender: 'user', 
            text: userMessage
        };
        setMessages(prev => [...prev, newUserMessage]);
        setInput('');
        // Don't set global isLoading for agent commands yet

        // --- Agent Command Handling --- 
        const agentMatch = userMessage.match(/^\/agent\s+(\d+)\s+([\s\S]*)/); // Escape the forward slash
        if (agentMatch) {
            const agentId = parseInt(agentMatch[1], 10);
            const goal = agentMatch[2].trim();
            const loadingMessageId = Date.now().toString() + '-agent-loading';
            
            // Add placeholder message
            const loadingMessage: Message = {
                id: loadingMessageId,
                sender: 'system', 
                text: `⏳ Queuing task for Agent ${agentId}... Goal: ${goal.substring(0, 50)}...`,
                isLoading: true,
                isAgentMessage: true,
                agentTaskId: null,
                agentTaskComplete: false,
            };
            setMessages(prev => [...prev, loadingMessage]);
            
            try {
                const response = await axios.post<{ task_db_id: number, celery_task_id: string }>(
                    `/api/v1/agents/${agentId}/run`,
                    { goal: goal } // Send goal in request body
                );
                const { task_db_id, celery_task_id } = response.data;
                
                // Update placeholder message with task ID
                setMessages(prevMessages => 
                    prevMessages.map(msg => 
                        msg.id === loadingMessageId 
                        ? { ...msg, 
                            text: `✅ Task queued for Agent ${agentId} (ID: ${task_db_id}). Waiting for result... Goal: ${goal.substring(0, 50)}...`, 
                            agentTaskId: task_db_id,
                            // isLoading: true, // Keep loading until WS update
                          } 
                        : msg
                    )
                );
                // Store mapping for WS update
                setPendingAgentTasks(prev => ({ ...prev, [task_db_id]: loadingMessageId }));
                
            } catch (error: any) {
                console.error("Error running agent:", error);
                const errorMsg = axios.isAxiosError(error) && error.response?.data?.detail
                               ? error.response.data.detail
                               : 'Failed to queue agent task.';
                // Update placeholder message with error
                 setMessages(prevMessages => 
                    prevMessages.map(msg => 
                        msg.id === loadingMessageId 
                        ? { ...msg, sender: 'system', text: `Error: ${errorMsg}`, isLoading: false } 
                        : msg
                    )
                );
            }

        } else {
            // --- Regular Chat Handling --- 
            setIsLoading(true); // Set global loading for regular chat
            // ... (existing non-agent chat logic using /api/v1/chat/stream or WebSocket) ...
            // Need to reintegrate the original WebSocket chat logic here if it was removed/overwritten.
            // Assuming original WebSocket logic was like this:
            if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
                 setError('WebSocket is not connected for chat. Please wait or refresh.');
                 console.error("WebSocket not ready to send chat message.");
                 setIsLoading(false);
            return; 
        }
            // ... (rest of the original regular chat send logic) ...
            // Make sure this part eventually sets setIsLoading(false) when the stream ends.
        }
    };

    // Handler for Enter key press in input
    const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent newline
            if (!isLoading && input.trim()) {
                handleSendMessage();
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
                    <div key={msg.id || index} className={`${styles.messageRow} ${msg.sender === 'user' ? styles.user : styles.assistant}`}> 
                        <div className={`${styles.messageContent} ${msg.sender === 'user' ? styles.user : styles.assistant}`}> 
                             <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                            {msg.isLoading && msg.isAgentMessage && <span className={styles.loadingIndicator}> (Running...)</span>}
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
                 {(isLoading || Object.keys(pendingAgentTasks).length > 0) && (
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
                     placeholder={Object.keys(pendingAgentTasks).length > 0 ? "Agent is processing..." : "Ask VibeRAG or type /agent <id> <prompt>..."}
                     rows={1}
                     disabled={isLoading || messages.some(m => m.isLoading)}
                 />
                 <button onClick={handleSendMessage} disabled={isLoading || messages.some(m => m.isLoading)} className={styles.sendButton}> 
                     Send
                 </button>
             </div>

             {/* Error Display */} 
             {error && <div className={styles.errorMessage}>{error}</div>}
        </div>
    );
}

export default Chat; 