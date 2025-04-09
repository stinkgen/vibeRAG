/**
 * API Configuration
 * Centralizes all API URLs and endpoints for the frontend
 */

// Get the base URL from environment variables, defaulting to a relative path
// Nginx will proxy requests starting with /api/
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

// API Endpoints
const API_ENDPOINTS = {
  // Chat endpoints
  CHAT: `${API_BASE_URL}/api/v1/chat`,
  
  // Document management
  UPLOAD: `${API_BASE_URL}/api/v1/upload`,
  DELETE_DOCUMENT_BY_FILENAME: (filename: string) => `${API_BASE_URL}/api/v1/delete/${filename}`,
  GET_DOCUMENT: (filename: string) => `${API_BASE_URL}/api/v1/get_pdf/${filename}`,
  UPDATE_DOCUMENT_METADATA: (filename: string) => `${API_BASE_URL}/api/v1/documents/${filename}/metadata`,
  
  // Configuration
  CONFIG: `${API_BASE_URL}/api/v1/config`,
  UPDATE_CONFIG: `${API_BASE_URL}/api/v1/config`,
  
  // Provider status
  OLLAMA_STATUS: `${API_BASE_URL}/api/v1/providers/ollama/status`,
  OLLAMA_LOAD: `${API_BASE_URL}/api/v1/providers/ollama/load`,
  OPENAI_MODELS: `${API_BASE_URL}/api/v1/providers/openai/models`,
  
  // Presentation and research
  PRESENTATION: `${API_BASE_URL}/api/v1/presentation`,
  RESEARCH: `${API_BASE_URL}/api/v1/research`,
  
  // Knowledge filters & Doc Listing (using /list which is defined as DOCUMENTS)
  DOCUMENTS: `${API_BASE_URL}/api/v1/list`,
  COLLECTIONS: `${API_BASE_URL}/api/v1/collections`,
  TAGS: `${API_BASE_URL}/api/v1/tags`,
};

export default API_ENDPOINTS; 