import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import styles from './Chat.module.css';
import KnowledgeFilter from './KnowledgeFilter';
import API_ENDPOINTS from '../config/api';
import { ModelSelector } from './ModelSelector';

// Brain SVG component - keeping it raw and minimal
const BrainIcon = () => (
    <svg 
        className={styles.brainIcon}
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 24 24"
    >
        <path d="M12 2c5.5 0 10 4.5 10 10s-4.5 10-10 10S2 17.5 2 12 6.5 2 12 2zm0 2c-4.4 0-8 3.6-8 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 2c3.3 0 6 2.7 6 6s-2.7 6-6 6-6-2.7-6-6 2.7-6 6-6zm2 2.4l-2.7 2.7-1.3-1.3-1.4 1.4 2.7 2.7 4.1-4.1-1.4-1.4z" 
            fill="currentColor" 
            stroke="none"
        />
    </svg>
);

// Interface for chat history entry
interface ChatHistory {
  id: string;
  title: string;
  date: string;
  messages: {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    sources?: string[];
    pending?: boolean;
  }[];
  modelProvider?: string;
  modelName?: string;
}

interface ChatResponse {
  response: string;
  sources: string[];
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: string[];
  pending?: boolean;
}

interface ConfigState {
  chat: {
    model: string;
    provider: string;
    temperature: number;
  };
}

// Add interface for filter option
interface FilterOption {
  id: string;
  name: string;
  type: 'file' | 'collection' | 'tag';
}

// Add interface for the OpenAI Models API Response to use in Chat.tsx
interface OpenAIModelData {
  id: string;
  created: number;
}
interface OpenAIModelsApiResponse {
  all_models: OpenAIModelData[];
  suggested_default: string;
  error?: string;
}

// System prompt with more in-depth response instruction
const SYSTEM_PROMPT = `You are an AI assistant providing in-depth, comprehensive, and thoughtful responses. 
When answering questions, include relevant context, examples, and explanations. 
Be thorough in your analysis while maintaining clarity and structure.
If using knowledge from documents, synthesize and integrate this information to provide a complete picture.
When appropriate, consider different perspectives and potential implications of the information.`;

const Chat: React.FC = () => {
  const [query, setQuery] = useState('');
  const [filename, setFilename] = useState('');
  const [knowledgeOnly, setKnowledgeOnly] = useState(true);
  const [useWeb, setUseWeb] = useState(false);
  const [streamResponse, setStreamResponse] = useState(true);
  
  // Conversation history
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [chatHistories, setChatHistories] = useState<ChatHistory[]>([]);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  
  // Sources management
  const [expandedSourceIds, setExpandedSourceIds] = useState<Set<string>>(new Set());
  
  // Errors and loading
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const webSocketRef = useRef<WebSocket | null>(null);
  const saveTimeoutRef = useRef<number | null>(null);
  const lastSavedMessagesRef = useRef<string>('');
  
  // ADD NEW modelConfig state
  const [modelConfig, setModelConfig] = useState<{ provider: string; model: string }>(() => {
    const savedConfig = localStorage.getItem('vibeRAG_modelConfig');
    if (savedConfig) {
      try {
        const parsedConfig = JSON.parse(savedConfig);
        // Basic validation
        if (parsedConfig && typeof parsedConfig.provider === 'string' && typeof parsedConfig.model === 'string') {
            console.log("Loaded model config from localStorage:", parsedConfig);
            return parsedConfig;
        }
      } catch (e) {
        console.error("Failed to parse saved model config from localStorage:", e);
      }
    }
    // Return a temporary default if nothing valid is found in localStorage
    console.log("No valid model config in localStorage, using temporary default.");
    return { provider: 'openai', model: 'gpt-4' }; // Temporary placeholder
  });

  // Add state for knowledge filters
  const [knowledgeFilters, setKnowledgeFilters] = useState<FilterOption[]>([]);
  
  // Add this state near the other state declarations in the Chat component
  const [showKnowledgeFilterDialog, setShowKnowledgeFilterDialog] = useState(false);
  
  // Effect to load chat history from localStorage on mount
  useEffect(() => {
    console.log("Loading chat histories from localStorage...");
    // Check for saved histories
    const savedHistories = localStorage.getItem('vibeRAG_chatHistories');
    if (savedHistories) {
      try {
        // Parse the JSON data
        const parsedHistories = JSON.parse(savedHistories) as ChatHistory[];
        
        // Sort histories by date (newest first)
        const historiesWithDates = parsedHistories.sort((a, b) => {
          return new Date(b.date).getTime() - new Date(a.date).getTime();
        });
        
        setChatHistories(historiesWithDates);
        console.log("Loaded chat histories:", historiesWithDates.length);
      } catch (e) {
        console.error("Error parsing saved chat histories:", e);
      }
    }
    
    // Check for active chat
    const activeChat = localStorage.getItem('vibeRAG_activeChat');
    if (activeChat) {
      try {
        const historyId = JSON.parse(activeChat) as string;
        setActiveHistoryId(historyId);
        console.log("Loading active chat history:", historyId);
      } catch (e) {
        console.error("Error parsing active chat ID:", e);
      }
    }
  }, []);
  
  // Effect to load active chat messages when activeHistoryId changes
  useEffect(() => {
    if (activeHistoryId) {
      const history = chatHistories.find(h => h.id === activeHistoryId);
      if (history) {
        console.log("Found active history, loading messages...");
        // Convert timestamp strings back to Date objects for each message
        const messagesWithDates = history.messages.map(msg => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        
        // Set the messages from the selected history
        setMessages(messagesWithDates);
        // Store the initial messages in a ref to detect changes
        lastSavedMessagesRef.current = JSON.stringify(
          messagesWithDates.map(m => ({ ...m, timestamp: m.timestamp.toISOString() }))
        );
        
        console.log("Loaded active chat messages:", messagesWithDates.length);
        
        // Also restore the model settings if they exist
        if (history.modelProvider || history.modelName) {
          setModelConfig(prev => ({
            ...prev,
            provider: history.modelProvider || prev.provider,
            model: history.modelName || prev.model
          }));
          console.log(`Restored model settings:`, {
            provider: history.modelProvider || 'default',
            model: history.modelName || 'default'
          });
        }
      } else {
        console.warn(`Active history (${activeHistoryId}) not found in loaded histories`);
      }
    }
  }, [activeHistoryId, chatHistories]);
  
  // Save chat histories to localStorage whenever they change
  useEffect(() => {
    if (chatHistories.length > 0) {
      localStorage.setItem('vibeRAG_chatHistories', JSON.stringify(chatHistories));
      console.log("Saved chat histories to localStorage:", chatHistories.length);
    }
  }, [chatHistories]);
  
  // Save active chat ID to localStorage whenever it changes
  useEffect(() => {
    if (activeHistoryId) {
      localStorage.setItem('vibeRAG_activeChat', JSON.stringify(activeHistoryId));
      console.log("Saved active chat ID to localStorage:", activeHistoryId);
    }
  }, [activeHistoryId]);
  
  // Add effect to automatically save messages when they change (debounced)
  useEffect(() => {
    if (messages.length > 0) {
      // Convert messages to string to detect actual content changes
      const messagesJson = JSON.stringify(
        messages.map(m => ({ ...m, timestamp: m.timestamp.toISOString() }))
      );
      
      // Only save if messages have actually changed
      if (messagesJson !== lastSavedMessagesRef.current) {
        console.log("Messages changed, scheduling save...");
        
        // Clear any existing timeout
        if (saveTimeoutRef.current !== null) {
          window.clearTimeout(saveTimeoutRef.current);
        }
        
        // Set a new timeout to save the chat
        saveTimeoutRef.current = window.setTimeout(() => {
          console.log("Executing debounced save of chat messages");
          saveCurrentChat();
          saveTimeoutRef.current = null;
          // Update last saved state
          lastSavedMessagesRef.current = messagesJson;
        }, 500);
      }
    }
    
    // Cleanup on unmount
    return () => {
      if (saveTimeoutRef.current !== null) {
        window.clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [messages]);
  
  // Function to save the current chat to history
  const saveCurrentChat = useCallback(() => {
    if (messages.length === 0) return;
    
    console.log("Saving current chat to history...");
    
    // Create a title based on the first user message
    const firstUserMessage = messages.find(m => m.role === 'user');
    const title = firstUserMessage 
      ? firstUserMessage.content.substring(0, 40) + (firstUserMessage.content.length > 40 ? '...' : '')
      : 'New Chat';
    
    // Remove any pending status before saving
    const messagesForStorage = messages.map(msg => ({
      ...msg,
      timestamp: msg.timestamp.toISOString(),
      pending: false // Ensure nothing is saved as pending
    }));
    
    if (activeHistoryId) {
      // Update existing history
      setChatHistories(prev => prev.map(history => 
        history.id === activeHistoryId 
          ? { 
              ...history, 
              messages: messagesForStorage, 
              date: new Date().toISOString(),
              modelProvider: modelConfig.provider,
              modelName: modelConfig.model
            }
          : history
      ));
      console.log(`Updated existing chat history: ${activeHistoryId}`);
    } else {
      // Create new history
      const newHistoryId = Math.random().toString(36).substring(2, 9);
      const newHistory: ChatHistory = {
        id: newHistoryId,
        title,
        date: new Date().toISOString(),
        messages: messagesForStorage,
        modelProvider: modelConfig.provider,
        modelName: modelConfig.model
      };
      
      setChatHistories(prev => [...prev, newHistory]);
      setActiveHistoryId(newHistoryId);
      console.log(`Created new chat history: ${newHistoryId}`);
    }
  }, [messages, activeHistoryId, modelConfig]);
  
  // Function to start a new chat
  const startNewChat = () => {
    console.log("Starting a new chat session...");
    closeWebSocket(); // Close existing WebSocket connection
    setMessages([]);
    setActiveHistoryId(null);
    setQuery('');
    setError('');
    // Reset last saved message state
    lastSavedMessagesRef.current = '';
    
    console.log("Started new chat");
  };
  
  // Function to select a chat from history
  const selectChatHistory = (historyId: string) => {
    // Only change if it's different
    if (historyId === activeHistoryId) {
      console.log("Already on this chat history, nothing to do");
      setShowHistoryPanel(false);
      return;
    }
    
    // Close any existing connection
    if (eventSourceRef.current) {
      console.log("Closing EventSource in selectChatHistory");
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    
    // Clear the save timeout
    if (saveTimeoutRef.current !== null) {
      window.clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    
    setActiveHistoryId(historyId);
    setShowHistoryPanel(false);
    console.log(`Selected chat history: ${historyId}`);
  };
  
  // Function to delete a chat history
  const deleteChatHistory = (historyId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering selectChatHistory
    
    console.log(`Attempting to delete chat history: ${historyId}`);
    
    // Special handling for the problematic chat ID
    if (historyId === "5shvv01") {
      console.log("Detected problematic chat ID, using force removal");
      
      // Filter out the problematic chat from histories
      const filteredHistories = chatHistories.filter(h => h.id !== historyId);
      setChatHistories(filteredHistories);
      
      // If this was the active chat, clear it
      if (activeHistoryId === historyId) {
        // Close any existing connection
        if (eventSourceRef.current) {
          console.log("Closing EventSource during force removal");
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
        
        setMessages([]);
        setActiveHistoryId(null);
        
        // Reset last saved message state
        lastSavedMessagesRef.current = '';
      }
      
      // Force update localStorage directly
      try {
        localStorage.setItem('vibeRAG_chatHistories', JSON.stringify(filteredHistories));
        
        // Also clear activeChat reference if needed
        if (activeHistoryId === historyId) {
          localStorage.removeItem('vibeRAG_activeChat');
        }
        
        console.log(`Successfully force-removed problematic chat history: ${historyId}`);
      } catch (e) {
        console.error("Error during force removal of chat history:", e);
      }
      
      return;
    }
    
    // Normal deletion path for other chat histories
    try {
      setChatHistories(prev => prev.filter(h => h.id !== historyId));
      console.log(`Deleted chat history: ${historyId}`);
      
      if (activeHistoryId === historyId) {
        // Close any existing connection
        if (eventSourceRef.current) {
          console.log("Closing EventSource in deleteChatHistory");
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
        
        setMessages([]);
        setActiveHistoryId(null);
        
        // Reset last saved message state
        lastSavedMessagesRef.current = '';
      }
    } catch (error) {
      console.error(`Error deleting chat history ${historyId}:`, error);
    }
  };

  // Toggle source expansion
  const toggleSourceExpansion = (messageId: string) => {
    setExpandedSourceIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Effect to scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch config on mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await axios.get<ConfigState>(API_ENDPOINTS.CONFIG); // Fetch general config if needed
        // You might use parts of this config, but modelConfig state handles the model selection
        console.log("Fetched general config:", response.data);
      } catch (error) {
        console.error('Failed to load general config:', error);
      }
    };
    fetchConfig();
  }, []);
  
  // Cleanup WebSocket connection on component unmount or when starting a new chat
  useEffect(() => {
    return () => {
      if (webSocketRef.current) {
        console.log('[WebSocket] Closing WebSocket connection on cleanup.');
        webSocketRef.current.close();
        webSocketRef.current = null;
      }
    };
  }, []); // Run only on mount and unmount

  const closeWebSocket = () => {
      if (webSocketRef.current) {
        console.log('[WebSocket] Closing WebSocket connection manually.');
        webSocketRef.current.close();
        webSocketRef.current = null;
      }
  };

  // Handle knowledge filter changes
  const handleFilterChange = (selectedFilters: FilterOption[]) => {
    setKnowledgeFilters(selectedFilters);
    
    // Save filters to localStorage for persistence
    localStorage.setItem('vibeRAG_knowledgeFilters', JSON.stringify(selectedFilters));
  };

  // Load knowledge filters from localStorage
  useEffect(() => {
    try {
      const savedFilters = localStorage.getItem('vibeRAG_knowledgeFilters');
      if (savedFilters) {
        setKnowledgeFilters(JSON.parse(savedFilters));
      }
    } catch (err) {
      console.error("Error loading knowledge filters:", err);
    }
  }, []);

  // --- MODIFICATION START: Effect to set default model if not loaded from localStorage ---
  useEffect(() => {
    const savedConfig = localStorage.getItem('vibeRAG_modelConfig');
    if (!savedConfig) {
      console.log("No model config in localStorage, fetching suggested default...");
      const fetchSuggestedDefault = async () => {
        try {
          const response = await axios.get<OpenAIModelsApiResponse>(API_ENDPOINTS.OPENAI_MODELS);
          if (response.data && response.data.suggested_default) {
            const suggestedConfig = { provider: 'openai', model: response.data.suggested_default };
            console.log("Setting model config to suggested default:", suggestedConfig);
            setModelConfig(suggestedConfig);
            localStorage.setItem('vibeRAG_modelConfig', JSON.stringify(suggestedConfig));
          } else {
            console.warn("API did not return a suggested default model.");
            // Keep the temporary default set in useState if API fails
          }
        } catch (error) {
          console.error('Failed to fetch suggested default OpenAI model:', error);
          // Keep the temporary default set in useState if API fails
        }
      };
      fetchSuggestedDefault();
    }
  }, []); // Run only once on mount
  // --- MODIFICATION END ---

  // Restore the full handleSubmit function
  // Make event optional to allow calling from onKeyDown without an event
  const handleSubmit = (e?: React.FormEvent<HTMLFormElement>) => {
    if (e) e.preventDefault(); // Prevent default only if event is provided
    if (loading || query.trim() === '') return;

    setLoading(true);
    setError('');
    const currentQuery = query; // Store the query
    setQuery(''); // Clear the input field immediately

    // Close existing WebSocket if any before starting
    closeWebSocket();

    // Add user message and initial assistant message (pending)
    const assistantMessageId = Date.now().toString() + '-assistant'; // Use ID for tracking
    setMessages((prevMessages) => [
      ...prevMessages,
      { 
        id: Date.now().toString() + '-user', 
        role: 'user', 
        content: currentQuery, 
        timestamp: new Date(), 
        sources: [] 
      },
      { 
        id: assistantMessageId, 
        role: 'assistant', 
        content: '', 
        sources: [], 
        timestamp: new Date(),
        pending: true // Mark as pending
      }, 
    ]);

    // Construct WebSocket URL (relative path to trigger proxy)
    // Ensure the protocol is ws or wss based on window.location.protocol
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'; // Original line
    const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/chat`; // Original line

    // --- TEMPORARY CHANGE: Connect directly to backend --- 
    // const wsUrl = `ws://localhost:8000/api/v1/ws/chat`; // Commented out
    // --------------------------------------------------

    // console.log(`[WebSocket] Connecting directly to ${wsUrl}`); // Comment out direct log
    console.log(`[WebSocket] Connecting to proxy at ${wsUrl}`); // Log proxy connection

    try {
      const socket = new WebSocket(wsUrl);
      webSocketRef.current = socket;

      socket.onopen = () => {
        console.log('[WebSocket] Connection opened.');
        // Prepare the initial message payload based on backend expectations
        const initialPayload = {
          query: currentQuery,
          knowledge_only: knowledgeOnly,
          use_web: useWeb,
          provider: modelConfig.provider,
          model: modelConfig.model,
          filters: knowledgeFilters, // Send selected filters
          chat_history_id: activeHistoryId, // Send active chat ID if available
          // Add any other parameters expected by the backend WebSocket endpoint
        };
        console.log('[WebSocket] Sending initial payload:', initialPayload);
        socket.send(JSON.stringify(initialPayload));
      };

      socket.onmessage = (event) => {
        // console.log('[WebSocket] Raw message received:', event.data);
        try {
          const data = JSON.parse(event.data);
          // console.log('[WebSocket] Parsed message data:', data);

          setMessages((prevMessages) => {
            const updatedMessages = [...prevMessages];
            const assistantMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);

            if (assistantMsgIndex === -1) {
                console.warn('[WebSocket] Could not find assistant message to update.');
                return prevMessages; // Should not happen ideally
            }
            
            const assistantMsg = { ...updatedMessages[assistantMsgIndex] }; // Clone to modify

            // Make assistant message non-pending as soon as we get *any* data
            if (assistantMsg.pending) {
                assistantMsg.pending = false;
            }

            switch (data.type) {
                case 'sources':
                    assistantMsg.sources = data.data; // Assuming data structure { type: 'sources', data: [...] }
                    // console.log('[WebSocket] Updated sources:', assistantMsg.sources);
                    break;
                case 'response':
                    assistantMsg.content += data.data; // Assuming data structure { type: 'response', data: "..." }
                    // console.log('[WebSocket] Appended response content chunk.');
                    break;
                case 'error':
                    console.error('[WebSocket] Error message received:', data.data);
                    const errorText = data.data?.startsWith('LLM generation failed:') 
                                        ? data.data 
                                        : `**Error:** ${data.data || 'Unknown error from backend.'}`;
                    assistantMsg.content = errorText;
                    setError(data.data || 'Unknown error from backend'); // Set top-level error state
                    setLoading(false);
                    closeWebSocket(); // Close socket on backend error signal
                    break;
                case 'end': // Backend signals end of stream
                    console.log('[WebSocket] Received end signal from backend.');
                    setLoading(false);
                    closeWebSocket(); // Close socket explicitly
                    break;
                // Handle other message types if the backend sends them
                default:
                    console.warn('[WebSocket] Unknown message type received:', data.type, data);
            }
            
            updatedMessages[assistantMsgIndex] = assistantMsg;
            return updatedMessages;
          });

        } catch (parseError) {
          console.error('[WebSocket] Failed to parse message JSON:', event.data, parseError);
          setError('Received malformed data from server.');
          setMessages((prevMessages) => { // Update UI to show error state
            const updatedMessages = [...prevMessages];
            const assistantMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
            if (assistantMsgIndex !== -1) {
                updatedMessages[assistantMsgIndex].content = "**Error:** Failed to process server message.";
                updatedMessages[assistantMsgIndex].pending = false;
            }
            return updatedMessages;
          });
          setLoading(false);
          closeWebSocket(); // Close socket on parsing error
        }
      };

      socket.onerror = (error) => {
        console.error('[WebSocket] Connection error:', error);
        setError('WebSocket connection error. Please check the server and network.');
        setLoading(false);
        setMessages((prevMessages) => { // Update UI to show error state
            const updatedMessages = [...prevMessages];
            const assistantMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
            if (assistantMsgIndex !== -1 && updatedMessages[assistantMsgIndex].pending) { 
                updatedMessages[assistantMsgIndex].content = "**Error:** Connection failed.";
                updatedMessages[assistantMsgIndex].pending = false;
            }
            return updatedMessages;
        });
        webSocketRef.current = null; // Clear ref on error
      };

      socket.onclose = (event) => {
        console.log(`[WebSocket] Connection closed. Code: ${event.code}, Reason: ${event.reason || 'N/A'}`);
        setLoading(false);
        // Ensure the last message isn't stuck in pending state if closed unexpectedly
        setMessages((prevMessages) =>
            prevMessages.map((msg) =>
              msg.id === assistantMessageId && msg.pending ? { ...msg, pending: false, content: msg.content || "[Connection Closed]" } : msg
            )
        );
        webSocketRef.current = null; // Clear the ref
      };

    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to initialize WebSocket.');
      setLoading(false);
       setMessages((prevMessages) => { // Update UI to show error state
            const updatedMessages = [...prevMessages];
            const assistantMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
             // Correctly check if the message exists and is pending before modifying
            if (assistantMsgIndex !== -1 && updatedMessages[assistantMsgIndex].pending) {
                updatedMessages[assistantMsgIndex].content = "**Error:** Could not initiate connection.";
                updatedMessages[assistantMsgIndex].pending = false;
            }
            return updatedMessages;
        });
    }
  };

  const handleSourceClick = (source: string) => {
    // Extract the filename from the source string
    const pdfFilename = source.split('/').pop() || '';
    
    // Open the PDF in a new tab
    window.open(`${API_ENDPOINTS.GET_DOCUMENT(pdfFilename)}`, '_blank');
  };
  
  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Format date for chat history
  const formatDate = (date: Date) => {
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    
    if (isToday) {
      return `Today at ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const isYesterday = date.toDateString() === yesterday.toDateString();
    
    if (isYesterday) {
      return `Yesterday at ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    
    return date.toLocaleDateString(undefined, { 
      month: 'short', 
      day: 'numeric',
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
    });
  };
  
  // Auto-resize textarea
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(e.target.value);
    const textarea = e.target;
    
    // Reset height to auto to get the correct scrollHeight
    textarea.style.height = '24px';
    
    // Set to scrollHeight + 2px border
    const scrollHeight = textarea.scrollHeight;
    textarea.style.height = `${Math.min(scrollHeight, 100)}px`;
  };

  // --- MODIFICATION START: Update handlers passed to ModelSelector ---
  const handleProviderChange = (newProvider: string) => {
    // Determine a sensible default model for the new provider
    // We could fetch defaults per provider, but hardcoding is simpler for now.
    // Let ModelSelector handle fetching its own list based on the new provider.
    const defaultModel = newProvider === 'openai' ? 'gpt-4o' : 'llama3'; // TODO: Improve default selection?
    const newConfig = { provider: newProvider, model: defaultModel }; // Reset model when provider changes
    console.log(`Provider changed to ${newProvider}, resetting model to ${defaultModel}`);
    setModelConfig(newConfig);
    localStorage.setItem('vibeRAG_modelConfig', JSON.stringify(newConfig));
  };

  const handleModelUpdate = (newModel: string) => {
    // Provider remains the same, just update the model
    const newConfig = { ...modelConfig, model: newModel };
    console.log(`Model updated to ${newModel} for provider ${modelConfig.provider}`);
    setModelConfig(newConfig);
    localStorage.setItem('vibeRAG_modelConfig', JSON.stringify(newConfig));
  };
  // --- MODIFICATION END ---

  return (
    <div className={styles.container}>
      <div className={styles.cyberHeader}>
        <div className={styles.headerActions}>
          <button 
            className={styles.historyButton} 
            onClick={() => setShowHistoryPanel(!showHistoryPanel)}
            title="Chat History"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="8" y1="6" x2="21" y2="6"></line>
              <line x1="8" y1="12" x2="21" y2="12"></line>
              <line x1="8" y1="18" x2="21" y2="18"></line>
              <line x1="3" y1="6" x2="3.01" y2="6"></line>
              <line x1="3" y1="12" x2="3.01" y2="12"></line>
              <line x1="3" y1="18" x2="3.01" y2="18"></line>
            </svg>
          </button>
          <button 
            className={styles.newChatButton} 
            onClick={startNewChat}
            title="New Chat"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="16"></line>
              <line x1="8" y1="12" x2="16" y2="12"></line>
            </svg>
          </button>
        </div>
        <h2 className={styles.title}>
          <BrainIcon />
          AI Chat Interface
        </h2>
        <div className={styles.gridBackground}></div>
      </div>
      
      {showHistoryPanel && (
        <div className={styles.historyPanel}>
          <div className={styles.historyHeader}>
            <h3>Chat History</h3>
            <button 
              className={styles.closeButton}
              onClick={() => setShowHistoryPanel(false)}
            >
              <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
          <div className={styles.historyList}>
            {chatHistories.length === 0 ? (
              <div className={styles.noHistory}>No chat history yet</div>
            ) : (
              chatHistories
                .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
                .map(history => (
                  <div 
                    key={history.id} 
                    className={`${styles.historyItem} ${activeHistoryId === history.id ? styles.active : ''}`}
                    onClick={() => selectChatHistory(history.id)}
                  >
                    <div className={styles.historyTitle}>{history.title}</div>
                    <div className={styles.historyMeta}>
                      <span className={styles.historyDate}>
                        <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '4px', verticalAlign: 'middle' }}>
                          <circle cx="12" cy="12" r="10"></circle>
                          <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                        {formatDate(new Date(history.date))}
                      </span>
                      <button 
                        className={styles.historyDeleteButton}
                        onClick={(e) => deleteChatHistory(history.id, e)}
                        title="Delete this chat"
                      >
                        <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6"></polyline>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                          <line x1="10" y1="11" x2="10" y2="17"></line>
                          <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                      </button>
                    </div>
                  </div>
                ))
            )}
          </div>
        </div>
      )}
      
      <div className={styles.messagesContainer} ref={messagesContainerRef}>
        {messages.length === 0 ? (
          <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column'}}>
            <div style={{
              width: '80px', 
              height: '80px', 
              borderRadius: '50%', 
              border: '2px solid rgba(0, 150, 255, 0.5)',
              boxShadow: '0 0 20px rgba(0, 150, 255, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '20px'
            }}>
              <BrainIcon />
            </div>
            <div style={{color: 'rgba(255,255,255,0.6)', fontSize: '0.9rem', textAlign: 'center'}}>
              ENTER QUERY TO INITIATE KNOWLEDGE TRANSFER
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div 
              key={message.id}
              className={`${styles.messageRow} ${styles[message.role]} ${message.pending ? styles.pending : ''}`}
            >
              <div className={`${styles.messageContent} ${styles[message.role]}`}>
                {message.content && message.content.split('\n').map((line, i) => (
                  <div key={i}>{line || ' '}</div>
                ))}
                
                {message.pending && (
                  <div className={styles.typingIndicator}>
                    <span className={styles.typingDot}></span>
                    <span className={styles.typingDot}></span>
                    <span className={styles.typingDot}></span>
                  </div>
                )}
                
                <div className={styles.messageTime}>
                  {formatTimestamp(message.timestamp)}
                </div>
                
                {message.sources && message.sources.length > 0 && (
                  <div className={styles.sourcesContainer}>
                    <button 
                      className={styles.sourcesToggle}
                      onClick={() => toggleSourceExpansion(message.id)}
                    >
                      {expandedSourceIds.has(message.id) ? 'HIDE SOURCES' : 'SHOW SOURCES'}
                      <span className={styles.sourcesBadge}>{message.sources.length}</span>
                    </button>
                    
                    <div className={`${styles.sourcesList} ${expandedSourceIds.has(message.id) ? styles.expanded : ''}`}>
                      {message.sources.map((source, index) => (
                        <div 
                          key={index} 
                          className={styles.sourceItem}
                          onClick={() => handleSourceClick(source)}
                        >
                          {source}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className={styles.systemPanel}>
        <div className={styles.inputGroup}>
          <input
            type="text"
            placeholder="TARGET FILE: [OPTIONAL]"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            className={styles.input}
          />
        </div>
        
        <div className={styles.toggleGroup}>
          <div className={styles.toggleRow}>
            <button 
              className={styles.filterButton}
              onClick={() => setShowKnowledgeFilterDialog(true)}
              title="Select knowledge sources"
              disabled={!knowledgeOnly}
            >
              <span className={styles.filterButtonLabel}>SELECT SOURCES</span>
              {knowledgeFilters.length > 0 && (
                <span className={styles.filterBadgeCount}>{knowledgeFilters.length}</span>
              )}
            </button>
            
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={knowledgeOnly}
                onChange={() => setKnowledgeOnly(!knowledgeOnly)}
              />
              <span className={styles.toggleLabel}>KNOWLEDGE ONLY</span>
            </label>
          </div>
          
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={useWeb}
              onChange={() => setUseWeb(!useWeb)}
            />
            <span className={styles.toggleLabel}>WEB SEARCH</span>
          </label>
          
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={streamResponse}
              onChange={() => setStreamResponse(!streamResponse)}
            />
            <span className={styles.toggleLabel}>STREAM MODE</span>
          </label>
        </div>
        
        <div className={styles.inputGroup}>
          <ModelSelector
              provider={modelConfig.provider}
              model={modelConfig.model}
              onProviderChange={handleProviderChange}
              onModelChange={handleModelUpdate}
          />
        </div>
      </div>
      
      {showKnowledgeFilterDialog && (
        <div className={styles.modalOverlay} onClick={() => setShowKnowledgeFilterDialog(false)}>
          <div className={styles.filterModal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Knowledge Filters</h3>
              <button 
                className={styles.closeButton}
                onClick={() => setShowKnowledgeFilterDialog(false)}
              >
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            <div className={styles.modalContent}>
              <KnowledgeFilter 
                initialFilters={knowledgeFilters} 
                onFilterChange={(filters) => {
                  handleFilterChange(filters);
                  if (filters.length > 0) {
                    setShowKnowledgeFilterDialog(false);
                  }
                }}
              />
            </div>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className={styles.inputArea}>
        <textarea
          ref={textareaRef}
          value={query}
          onChange={handleTextareaChange}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              if (streamResponse) {
                handleSubmit();
              } else {
                handleSubmit();
              }
            }
          }}
          placeholder="Enter your query here..."
          className={styles.queryInput}
          disabled={loading}
        />
        <button 
          type="submit" 
          className={`${styles.sendButton} ${loading ? styles.loading : ''}`}
          disabled={loading || !query.trim()}
        >
          {loading ? '...' : 'Send'}
        </button>
      </form>
      
      {error && <div className={styles.errorMessage}>{error}</div>}
      
      {/* Add debug button to force clear problematic chat outside normal UI flow */}
      <div style={{ position: 'fixed', bottom: '5px', right: '5px', zIndex: 999, opacity: 0.3 }}>
        <button 
          onClick={() => {
            // Find and remove problematic chat with ID "5shvv01"
            const filteredHistories = chatHistories.filter(h => h.id !== "5shvv01");
            setChatHistories(filteredHistories);
            localStorage.setItem('vibeRAG_chatHistories', JSON.stringify(filteredHistories));
            
            // Check if we need to clear active chat
            if (activeHistoryId === "5shvv01") {
              localStorage.removeItem('vibeRAG_activeChat');
              setActiveHistoryId(null);
              setMessages([]);
            }
            
            console.log("Force cleared problematic chat");
          }}
          style={{
            background: 'rgba(0,0,0,0.5)', 
            border: '1px solid rgba(255,50,50,0.3)', 
            color: 'rgba(255,50,50,0.8)',
            fontSize: '10px',
            padding: '2px 5px',
            cursor: 'pointer'
          }}
        >
          Clear stuck chat
        </button>
      </div>
    </div>
  );
};

export default Chat; 