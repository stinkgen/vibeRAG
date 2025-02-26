import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import styles from './Chat.module.css';
import KnowledgeFilter from './KnowledgeFilter';
import API_ENDPOINTS from '../config/api';

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

// System prompt with more in-depth response instruction
const SYSTEM_PROMPT = `You are an AI assistant providing in-depth, comprehensive, and thoughtful responses. 
When answering questions, include relevant context, examples, and explanations. 
Be thorough in your analysis while maintaining clarity and structure.
If using knowledge from documents, synthesize and integrate this information to provide a complete picture.
When appropriate, consider different perspectives and potential implications of the information.`;

// Add interface for filter option
interface FilterOption {
  id: string;
  name: string;
  type: 'file' | 'collection' | 'tag';
}

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
  const saveTimeoutRef = useRef<number | null>(null);
  const lastSavedMessagesRef = useRef<string>('');
  
  // Configuration
  const [config, setConfig] = useState<ConfigState>({
    chat: {
      model: '',
      provider: '',
      temperature: 0.7
    }
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
          setConfig(prev => ({
            ...prev,
            chat: {
              ...prev.chat,
              provider: history.modelProvider || prev.chat.provider,
              model: history.modelName || prev.chat.model
            }
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
              modelProvider: config.chat.provider,
              modelName: config.chat.model
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
        modelProvider: config.chat.provider,
        modelName: config.chat.model
      };
      
      setChatHistories(prev => [...prev, newHistory]);
      setActiveHistoryId(newHistoryId);
      console.log(`Created new chat history: ${newHistoryId}`);
    }
  }, [messages, activeHistoryId, config]);
  
  // Function to start a new chat
  const startNewChat = () => {
    // Close any existing connection
    if (eventSourceRef.current) {
      console.log("Closing EventSource in startNewChat");
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    
    // Clear the save timeout
    if (saveTimeoutRef.current !== null) {
      window.clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    
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
        const { data } = await axios.get(API_ENDPOINTS.CONFIG);
        // Only update if we don't have a saved config
        if (!config.chat.model || !config.chat.provider) {
          setConfig(data as ConfigState);
        }
      } catch (error) {
        console.error('Failed to fetch config:', error);
      }
    };
    
    fetchConfig();
  }, [config.chat.model, config.chat.provider]);
  
  // Cleanup function for EventSource
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        console.log("Closing EventSource on component unmount");
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      
      // Also clear any pending save timeout
      if (saveTimeoutRef.current !== null) {
        window.clearTimeout(saveTimeoutRef.current);
        saveTimeoutRef.current = null;
      }
    };
  }, []);

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

  // Handle streaming response with improved robustness and error recovery
  const handleStreamingResponse = (e?: React.FormEvent) => {
    // Prevent default form submission behavior if event is provided
    if (e) e.preventDefault();
    
    // Validate input
    if (!query.trim()) return;
    
    // Reset streaming state
    setError('');
    setLoading(true);
    
    // Close any existing EventSource before creating a new one
    if (eventSourceRef.current) {
      console.log("Closing existing EventSource before starting new request");
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    
    // Add user message to the conversation immediately
    const userMessageId = Math.random().toString(36).substring(2, 9);
    const userMessage: Message = {
      id: userMessageId,
      role: 'user',
      content: query,
      timestamp: new Date()
    };
    
    // Add a pending assistant message
    const assistantMessageId = Math.random().toString(36).substring(2, 9);
    const pendingMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      sources: [],
      pending: true
    };
    
    // Ensure messages are updated atomically
    const updatedMessages = [...messages, userMessage, pendingMessage];
    setMessages(updatedMessages);
    console.log("Added user message and pending assistant message");
    
    // Build URL with query params
    const params = new URLSearchParams({
      query: query,
      knowledge_only: knowledgeOnly.toString(),
      use_web: useWeb.toString(),
      stream: 'true',
      ...(config.chat.model ? { model: config.chat.model } : {}),
      ...(config.chat.provider ? { provider: config.chat.provider } : {}),
      ...(filename ? { filename: filename } : {})
    });
    
    // Include full conversation context for better continuity - include the current user message
    const context = [
      ...messages.filter(msg => !msg.pending).map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      { role: userMessage.role, content: userMessage.content }
    ];
    params.append('context', JSON.stringify(context));
    
    // Prepare filter data
    const filterData = {
      files: knowledgeFilters.filter(f => f.type === 'file').map(f => f.id),
      collections: knowledgeFilters.filter(f => f.type === 'collection').map(f => f.id),
      tags: knowledgeFilters.filter(f => f.type === 'tag').map(f => f.id),
    };
    
    // Add filter data to the URL
    params.append('filters', JSON.stringify(filterData));
    
    const url = `${API_ENDPOINTS.CHAT}?${params.toString()}`;
    console.log(`Sending request to: ${url}`);
    
    // Clear the input after sending
    setQuery('');
    
    // Create EventSource for SSE
    try {
      console.log("Creating new EventSource connection...");
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;
      
      let receivedData = false;
      let connectionOpened = false;
      let retryCount = 0;
      const MAX_RETRIES = 3;
      
      // Connection opened
      eventSource.onopen = () => {
        console.log("EventSource connection opened");
        connectionOpened = true;
      };
      
      // Track accumulated content to avoid race conditions
      let accumulatedContent = '';
      let accumulatedSources: string[] = [];
      
      eventSource.onmessage = (event) => {
        receivedData = true; // We've received at least one message
        console.log("Received SSE message:", event.data.substring(0, 50) + "...");
        
        try {
          // Parse the JSON message
          const data = JSON.parse(event.data);
          
          // Check for connection status message
          if (data.status === "connected") {
            console.log("SSE connection confirmed by server");
            return;
          }
          
          // Check for error
          if (data.error) {
            setError(data.error);
            console.error("Error from server:", data.error);
            eventSource.close();
            eventSourceRef.current = null;
            setLoading(false);
            
            // Update the pending message to show the error
            setMessages(prev => {
              // Find the message and update it
              const updatedMessages = [...prev];
              const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
              
              if (pendingMsgIndex !== -1) {
                updatedMessages[pendingMsgIndex] = {
                  ...updatedMessages[pendingMsgIndex],
                  content: `Error: ${data.error}`,
                  pending: false
                };
              }
              
              return updatedMessages;
            });
            return;
          }
          
          // Handle sources
          if (data.sources) {
            console.log("Received sources:", data.sources);
            accumulatedSources = data.sources;
            
            // Update the pending message with sources
            setMessages(prev => {
              // Find the message and update it
              const updatedMessages = [...prev];
              const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
              
              if (pendingMsgIndex !== -1) {
                updatedMessages[pendingMsgIndex] = {
                  ...updatedMessages[pendingMsgIndex],
                  sources: accumulatedSources
                };
              }
              
              return updatedMessages;
            });
            return;
          }
          
          // Handle response chunk
          if (data.response) {
            console.log("Received response chunk:", data.response.substring(0, 20) + "...");
            accumulatedContent += data.response;
            
            // Update the pending message content
            setMessages(prev => {
              // Find the message and update it
              const updatedMessages = [...prev];
              const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
              
              if (pendingMsgIndex !== -1) {
                updatedMessages[pendingMsgIndex] = {
                  ...updatedMessages[pendingMsgIndex],
                  content: accumulatedContent,
                  sources: accumulatedSources
                };
              }
              
              return updatedMessages;
            });
            return;
          }
          
          console.log("Received unhandled data format:", data);
        } catch (e) {
          // Fallback for non-JSON messages
          console.error('Error parsing SSE message:', e, "Raw data:", event.data);
          
          if (event.data.includes('[DONE]')) {
            console.log("Received [DONE] message");
            eventSource.close();
            eventSourceRef.current = null;
            setLoading(false);
            
            // Mark the assistant message as complete
            setMessages(prev => {
              // Find the message and update it
              const updatedMessages = [...prev];
              const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
              
              if (pendingMsgIndex !== -1) {
                updatedMessages[pendingMsgIndex] = {
                  ...updatedMessages[pendingMsgIndex],
                  pending: false
                };
              }
              
              return updatedMessages;
            });
            return;
          }
          
          // Don't set error for empty messages
          if (event.data.trim()) {
            setError(`Error parsing response: ${(e as Error).message || 'Unknown error'}. Check console for details.`);
          }
        }
      };
      
      // Handle specific SSE events
      eventSource.addEventListener('end', () => {
        console.log("EventSource 'end' event received");
        eventSource.close();
        eventSourceRef.current = null;
        setLoading(false);
        
        // Mark the assistant message as complete
        setMessages(prev => {
          // Find the message and update it atomically
          const updatedMessages = [...prev];
          const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
          
          if (pendingMsgIndex !== -1) {
            updatedMessages[pendingMsgIndex] = {
              ...updatedMessages[pendingMsgIndex],
              content: accumulatedContent,
              sources: accumulatedSources,
              pending: false
            };
          }
          
          return updatedMessages;
        });
      });
      
      eventSource.onerror = (err) => {
        console.error("EventSource error:", err);
        
        // If we never received any data and connection was opened, retry
        if (connectionOpened && !receivedData && retryCount < MAX_RETRIES) {
          retryCount++;
          console.log(`Retrying connection (${retryCount}/${MAX_RETRIES})...`);
          return; // EventSource will automatically try to reconnect
        }
        
        // If max retries reached or other conditions not met
        if (!receivedData) {
          console.log("Connection failed without receiving any data");
          const errorMsg = "Connection error. The server may be unavailable or not properly configured for event streaming. Please check your browser console for details.";
          setError(errorMsg);
          
          // Update the pending message with the error
          setMessages(prev => {
            // Find the message and update it
            const updatedMessages = [...prev];
            const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
            
            if (pendingMsgIndex !== -1) {
              updatedMessages[pendingMsgIndex] = {
                ...updatedMessages[pendingMsgIndex],
                content: errorMsg,
                pending: false
              };
            }
            
            return updatedMessages;
          });
        } else {
          // If we've received some data, it might just be end of stream
          console.log("Error after receiving some data, might be end of stream");
          
          // Mark the assistant message as complete with whatever content we have
          setMessages(prev => {
            // Find the message and update it
            const updatedMessages = [...prev];
            const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
            
            if (pendingMsgIndex !== -1) {
              updatedMessages[pendingMsgIndex] = {
                ...updatedMessages[pendingMsgIndex],
                content: accumulatedContent || "Response was interrupted",
                sources: accumulatedSources,
                pending: false
              };
            }
            
            return updatedMessages;
          });
        }
        
        eventSource.close();
        eventSourceRef.current = null;
        setLoading(false);
      };
      
      // Safety timeout to ensure the loading state doesn't get stuck
      setTimeout(() => {
        if (eventSourceRef.current === eventSource) {
          console.log("Timeout reached, closing EventSource");
          eventSource.close();
          eventSourceRef.current = null;
          setLoading(false);
          
          if (!receivedData) {
            const timeoutMsg = "Request timed out after 30 seconds. Please try again or check server logs.";
            setError(timeoutMsg);
            
            // Update the pending message with the timeout error
            setMessages(prev => {
              // Find the message and update it
              const updatedMessages = [...prev];
              const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
              
              if (pendingMsgIndex !== -1) {
                updatedMessages[pendingMsgIndex] = {
                  ...updatedMessages[pendingMsgIndex],
                  content: timeoutMsg,
                  pending: false
                };
              }
              
              return updatedMessages;
            });
          } else {
            // Mark the assistant message as complete even if timeout occurred
            setMessages(prev => {
              // Find the message and update it
              const updatedMessages = [...prev];
              const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
              
              if (pendingMsgIndex !== -1) {
                updatedMessages[pendingMsgIndex] = {
                  ...updatedMessages[pendingMsgIndex],
                  content: accumulatedContent,
                  sources: accumulatedSources,
                  pending: false
                };
              }
              
              return updatedMessages;
            });
          }
        }
      }, 30000); // 30 second timeout
      
    } catch (err) {
      console.error("Error creating EventSource:", err);
      const connectionError = `Failed to connect to server: ${(err as Error).message || 'Unknown error'}. Please try again.`;
      setError(connectionError);
      setLoading(false);
      
      // Update the pending message with the connection error
      setMessages(prev => {
        // Find the message and update it
        const updatedMessages = [...prev];
        const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId && msg.pending);
        
        if (pendingMsgIndex !== -1) {
          updatedMessages[pendingMsgIndex] = {
            ...updatedMessages[pendingMsgIndex],
            content: connectionError,
            pending: false
          };
        }
        
        return updatedMessages;
      });
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) return;
    
    // Close any existing stream
    if (eventSourceRef.current) {
      console.log("Closing existing EventSource before starting new request");
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    
    if (streamResponse) {
      handleStreamingResponse(e);
      return;
    }
    
    // Regular JSON response flow
    setLoading(true);
    setError('');
    
    // Add user message to the conversation immediately
    const userMessageId = Math.random().toString(36).substring(2, 9);
    const userMessage: Message = {
      id: userMessageId,
      role: 'user',
      content: query,
      timestamp: new Date()
    };
    
    // Add a pending assistant message
    const assistantMessageId = Math.random().toString(36).substring(2, 9);
    const pendingMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: 'Loading...',
      timestamp: new Date(),
      pending: true
    };
    
    const updatedMessages = [...messages, userMessage, pendingMessage];
    setMessages(updatedMessages);
    console.log("Added user message and pending assistant message (non-streaming)");
    
    // Clear the input after sending
    setQuery('');
    
    try {
      const response = await axios.post<ChatResponse>(API_ENDPOINTS.CHAT, {
        query: query,
        knowledge_only: knowledgeOnly,
        use_web: useWeb,
        stream: false,
        ...(config.chat.model ? { model: config.chat.model } : {}),
        ...(config.chat.provider ? { provider: config.chat.provider } : {}),
        ...(filename ? { filename: filename } : {})
      });
      
      const data = response.data;
      console.log("Received non-streaming response:", data);
      
      // Update the pending message with the response
      setMessages(prev => {
        // Find the message and update it
        const updatedMessages = [...prev];
        const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
        
        if (pendingMsgIndex !== -1) {
          updatedMessages[pendingMsgIndex] = {
            ...updatedMessages[pendingMsgIndex],
            content: data.response || "No response received", 
            sources: data.sources || [],
            pending: false
          };
        }
        
        return updatedMessages;
      });
      
    } catch (error) {
      console.error('Chat request failed:', error);
      const errorMsg = 'Failed to get response. Please try again.';
      setError(errorMsg);
      
      // Update the pending message with the error
      setMessages(prev => {
        // Find the message and update it
        const updatedMessages = [...prev];
        const pendingMsgIndex = updatedMessages.findIndex(msg => msg.id === assistantMessageId);
        
        if (pendingMsgIndex !== -1) {
          updatedMessages[pendingMsgIndex] = {
            ...updatedMessages[pendingMsgIndex],
            content: errorMsg,
            pending: false
          };
        }
        
        return updatedMessages;
      });
    } finally {
      setLoading(false);
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

  // Add this handler function with the other handlers
  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    const [provider, model] = value.split(' [');
    const cleanModel = model.replace(']', '');
    
    const newConfig = {
      ...config,
      chat: {
        ...config.chat,
        provider,
        model: cleanModel
      }
    };
    
    setConfig(newConfig);
    
    // Save to localStorage
    localStorage.setItem('vibeRAG_modelConfig', JSON.stringify(newConfig));
    
    console.log(`Model changed to: ${cleanModel} (${provider})`);
  };

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
          <select 
            className={styles.modelSelect}
            value={`${config.chat.provider} [${config.chat.model}]`}
            disabled={false}
            onChange={handleModelChange}
          >
            <option value={`${config.chat.provider} [${config.chat.model}]`}>
              {`${config.chat.provider.toUpperCase()} [${config.chat.model}]`}
            </option>
            <option value="openai [gpt-4]">OPENAI [GPT-4]</option>
            <option value="openai [gpt-3.5-turbo]">OPENAI [GPT-3.5-TURBO]</option>
            <option value="anthropic [claude-3-opus]">ANTHROPIC [CLAUDE-3-OPUS]</option>
            <option value="anthropic [claude-3-sonnet]">ANTHROPIC [CLAUDE-3-SONNET]</option>
            <option value="ollama [llama3]">OLLAMA [LLAMA3]</option>
          </select>
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
      
      <form className={styles.inputForm} onSubmit={(e) => streamResponse ? handleStreamingResponse(e) : handleSubmit(e)}>
        <textarea
          ref={textareaRef}
          value={query}
          onChange={handleTextareaChange}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              if (streamResponse) {
                handleStreamingResponse();
              } else {
                handleSubmit(e);
              }
            }
          }}
          placeholder="Enter your query here..."
          className={styles.queryInput}
          disabled={loading}
        />
        <button 
          type="submit" 
          className={styles.sendButton}
          disabled={loading || !query.trim()}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
          <span>SEND</span>
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