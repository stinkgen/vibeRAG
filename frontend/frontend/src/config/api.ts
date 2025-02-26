/**
 * API Configuration
 * Centralizes all API URLs and endpoints for the frontend
 */

// Get the API base URL from environment variables or use default
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API Endpoints
const API_ENDPOINTS = {
  // Chat endpoints
  CHAT: `${API_BASE_URL}/chat`,
  
  // Document management
  UPLOAD: `${API_BASE_URL}/upload`,
  LIST_DOCUMENTS: `${API_BASE_URL}/list`,
  DELETE_DOCUMENT: (docId: string) => `${API_BASE_URL}/delete/${docId}`,
  GET_DOCUMENT: (filename: string) => `${API_BASE_URL}/get_pdf/${filename}`,
  
  // Configuration
  CONFIG: `${API_BASE_URL}/api/config`,
  UPDATE_CONFIG: `${API_BASE_URL}/api/config`,
  
  // Provider status
  OLLAMA_STATUS: `${API_BASE_URL}/api/providers/ollama/status`,
  OLLAMA_LOAD: `${API_BASE_URL}/api/providers/ollama/load`,
  OPENAI_MODELS: `${API_BASE_URL}/api/providers/openai/models`,
  
  // Presentation and research
  PRESENTATION: `${API_BASE_URL}/api/presentation`,
  RESEARCH: `${API_BASE_URL}/research`,
  
  // Knowledge filters
  DOCUMENTS: `${API_BASE_URL}/api/documents`,
  COLLECTIONS: `${API_BASE_URL}/api/collections`,
  TAGS: `${API_BASE_URL}/api/tags`,
};

export default API_ENDPOINTS; 