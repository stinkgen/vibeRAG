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
    const agentWs = useRef<WebSocket | null>(null);
    const chatWs = useRef<WebSocket | null>(null);
    const messagesEndRef = useRef<null | HTMLDivElement>(null);
    const [pendingAgentTasks, setPendingAgentTasks] = useState<Record<number, string>>({}); // { task_db_id: message_id }
    const currentUserId = useRef<number | null>(null);
    
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

    const connectAgentWebSocket = useCallback(() => {
        // Check isAuthReady before connecting
        if (!isAuthReady) {
            console.log("Agent WS connection deferred: Auth not ready.");
            return;
        }
        if (!currentUserId.current) {
            console.error("Cannot connect Agent WebSocket: User ID not available (derived from token).");
            return;
        }
        if (agentWs.current && agentWs.current.readyState === WebSocket.OPEN) {
            console.log("Agent WebSocket already connected.");
            return;
        }
        
        const storedToken = localStorage.getItem('vibeRAG_authToken');
        if (!storedToken) { 
            console.error("Agent WS connection failed: No auth token found.");
            setError("Auth token missing. Please log in.");
            return;
        }

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const agentWsUrl = `${wsProtocol}//${window.location.host}/api/v1/agents/ws/tasks?token=${encodeURIComponent(storedToken)}`; 
        
        console.log(`Attempting to connect Agent WebSocket: ${agentWsUrl}`);
        agentWs.current = new WebSocket(agentWsUrl);

        agentWs.current.onopen = () => {
            console.log('Agent WebSocket Connected');
            // setError(null); // Don't clear general error here
        };

        agentWs.current.onclose = (event) => {
            console.log('Agent WebSocket Disconnected:', event.code, event.reason);
            agentWs.current = null; 
            if (event.code === 1008) { 
                setError("Agent WebSocket failed: Auth error.");
            } else if (!event.wasClean && isAuthReady) {
                setError("Agent WebSocket connection lost. Reconnecting...");
                setTimeout(() => {
                    if (isAuthReady) connectAgentWebSocket(); 
                }, 5000); 
            }
        };

        agentWs.current.onerror = (error) => {
            console.error('Agent WebSocket Error:', error);
            setError("Agent WebSocket connection error.");
            agentWs.current = null; 
        };

        // Specific message handler for Agent WS
        agentWs.current.onmessage = (event) => {
            try {
                const messageData = JSON.parse(event.data);
                console.log('Agent WS Message Received:', messageData);

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
                                        isLoading: false, 
                                        agentTaskComplete: true, 
                                    };
                                }
                                return msg;
                            })
                        );
                        setPendingAgentTasks(prev => {
                            const newPending = { ...prev };
                            delete newPending[update.task_db_id];
                            return newPending;
                        });
                    } else {
                        console.warn(`Received task update for unknown/completed task: ${update.task_db_id}`);
                    }
                } else {
                     console.log("Received unexpected Agent WS message type:", messageData);
                }
            } catch (e) {
                console.error('Failed to parse Agent WS message:', event.data, e);
            }
        };
    }, [isAuthReady, setError, pendingAgentTasks]);

    // --- New function for Chat WebSocket --- 
    const connectChatWebSocket = useCallback(() => {
        if (!isAuthReady) {
            console.log("Chat WS connection deferred: Auth not ready.");
            return;
        }
        if (chatWs.current && chatWs.current.readyState === WebSocket.OPEN) {
            console.log("Chat WebSocket already connected.");
            return;
        }

        // Chat WS uses cookie auth handled by backend, no token needed in URL
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const chatWsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/chat`; 
        
        console.log(`Attempting to connect Chat WebSocket: ${chatWsUrl}`);
        chatWs.current = new WebSocket(chatWsUrl);

        chatWs.current.onopen = () => {
            console.log('Chat WebSocket Connected');
            setError(null); // Clear errors when chat connects
        };

        chatWs.current.onclose = (event) => {
            console.log('Chat WebSocket Disconnected:', event.code, event.reason);
            chatWs.current = null; 
            // Don't necessarily set error or reconnect automatically for chat?
            // Let user see the banner.
            if (event.code !== 1000) { // 1000 is normal closure
                 setError("Chat connection lost. Please refresh.");
            }
        };

        chatWs.current.onerror = (error) => {
            console.error('Chat WebSocket Error:', error);
            setError("Chat connection error.");
            chatWs.current = null; 
        };

        // Specific message handler for Chat WS
        chatWs.current.onmessage = (event) => {
            // TODO: Implement message handling for incoming chat messages/streams
            // This will likely involve updating the `messages` state
            console.log('Chat WS Message Received:', event.data);
            // Example: If streaming, append data; if complete, add new message
            // Need to coordinate with backend message format
        };
    }, [isAuthReady, setError]);

    // Effect to connect WebSockets when auth is ready 
    useEffect(() => {
        if (isAuthReady) {
            console.log("Auth is ready, connecting WebSockets...");
            connectAgentWebSocket();
            connectChatWebSocket();
        }
        // Cleanup function to close WebSockets
        return () => {
            if (agentWs.current) {
                console.log("Closing Agent WebSocket connection.");
                agentWs.current.onclose = null; 
                agentWs.current.close();
                agentWs.current = null;
            }
            if (chatWs.current) {
                console.log("Closing Chat WebSocket connection.");
                chatWs.current.onclose = null; 
                chatWs.current.close();
                chatWs.current = null;
            }
        };
    // Rerun if auth status changes OR if the connect functions themselves change (due to dependencies)
    }, [isAuthReady, connectAgentWebSocket, connectChatWebSocket]); 

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

        // --- Agent Command Handling --- 
        const agentMatch = userMessage.match(/^\/agent\s+(\d+)\s+([\s\S]*)/);
        if (agentMatch) {
            const agentId = parseInt(agentMatch[1], 10);
            const goal = agentMatch[2].trim();
            const loadingMessageId = Date.now().toString() + '-agent-loading';
            
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
                    { goal: goal }
                );
                const { task_db_id } = response.data;
                
                setMessages(prevMessages => 
                    prevMessages.map(msg => 
                        msg.id === loadingMessageId 
                        ? { ...msg, 
                            text: `✅ Task queued for Agent ${agentId} (ID: ${task_db_id}). Waiting for result via Agent WS... Goal: ${goal.substring(0, 50)}...`, 
                            agentTaskId: task_db_id,
                          } 
                        : msg
                    )
                );
                setPendingAgentTasks(prev => ({ ...prev, [task_db_id]: loadingMessageId }));
                
            } catch (error: any) {
                console.error("Error running agent:", error);
                const errorMsg = axios.isAxiosError(error) && error.response?.data?.detail
                               ? error.response.data.detail
                               : 'Failed to queue agent task.';
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
            
            // Use chatWs for sending regular messages
            if (!chatWs.current || chatWs.current.readyState !== WebSocket.OPEN) {
                 setError('WebSocket is not connected for chat. Please wait or refresh.');
                 console.error("Chat WebSocket not ready to send chat message.");
                 setIsLoading(false); // Stop loading indicator
                 return; 
            }
            
            try {
                // Send the message via chatWs
                // The backend chat_websocket_handler needs to know how to process this
                // Sending as a simple JSON for now
                const messagePayload = JSON.stringify({ query: userMessage, stream: streamMode });
                chatWs.current.send(messagePayload);
                console.log("Sent message via Chat WS:", messagePayload);
                
                // NOTE: We are NOT setting isLoading=false here immediately.
                // The isLoading state should be managed by the chatWs.onmessage handler 
                // when the streaming response starts and ends.
                // The current onmessage handler for chatWs is just a placeholder.
                
            } catch (sendError) {
                 console.error("Failed to send message via Chat WebSocket:", sendError);
                 setError("Failed to send message. Please try again.");
                 setIsLoading(false); // Stop loading on send error
            }
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