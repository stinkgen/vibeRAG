import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './Config.module.css';
import API_ENDPOINTS from '../config/api';

interface ConfigState {
  chat: {
    model: string;
    provider: string;
    temperature: number;
    chunks_limit: number;
  };
  openai: {
    api_key: string;
    base_url: string;
  };
  ollama: {
    host: string;
    model: string;
  };
  milvus: {
    host: string;
    port: number;
    collection_name: string;
  };
}

interface OllamaStatus {
  online: boolean;
  error?: string;
  models: string[];
}

interface OpenAIModelsResponse {
  models: string[];
  error?: string;
}

interface OllamaLoadResponse {
  status: string;
  message: string;
}

const Config: React.FC = () => {
  const [config, setConfig] = useState<ConfigState>({
    chat: {
      model: 'llama3',
      provider: 'ollama',
      temperature: 0.7,
      chunks_limit: 5
    },
    openai: {
      api_key: '',
      base_url: 'https://api.openai.com/v1'
    },
    ollama: {
      host: '',
      model: 'llama3'
    },
    milvus: {
      host: 'localhost',
      port: 19530,
      collection_name: 'documents'
    }
  });
  
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus>({ online: false, models: [] });
  const [openaiModels, setOpenaiModels] = useState<string[]>([]);
  const [loadingOllama, setLoadingOllama] = useState(false);
  const [loadingOpenai, setLoadingOpenai] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelLoadMessage, setModelLoadMessage] = useState('');
  const [error, setError] = useState('');
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    const fetchConfig = async () => {
      setLoading(true);
      try {
        const response = await axios.get<ConfigState>(API_ENDPOINTS.CONFIG);
        setConfig(response.data);
      } catch (error) {
        console.error('Failed to load config:', error);
        setError('Failed to load configuration. Check server connection.');
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  useEffect(() => {
    const checkOllamaStatus = async () => {
      try {
        const response = await axios.get<OllamaStatus>(API_ENDPOINTS.OLLAMA_STATUS);
        setOllamaStatus(response.data);
      } catch (error) {
        console.error('Failed to check Ollama status:', error);
      }
    };

    if (config.ollama.host) {
      checkOllamaStatus();
    }
  }, [config.ollama.host]);

  const fetchOpenAIModels = async () => {
    try {
      const response = await axios.get<OpenAIModelsResponse>(API_ENDPOINTS.OPENAI_MODELS);
      setOpenaiModels(response.data.models);
    } catch (error) {
      console.error('Failed to fetch OpenAI models:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await axios.post(API_ENDPOINTS.UPDATE_CONFIG, config);
      setMessage('Configuration saved successfully!');
      
      // Re-check Ollama status after saving config
      const response = await axios.get<OllamaStatus>(API_ENDPOINTS.OLLAMA_STATUS);
      setOllamaStatus(response.data);
    } catch (error) {
      console.error('Failed to save config:', error);
      setError('Failed to save configuration.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (section: keyof ConfigState, field: string, value: any) => {
    setConfig({
      ...config,
      [section]: {
        ...config[section],
        [field]: value
      }
    });
  };

  const loadOllamaModel = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoadingOllama(true);
    setModelLoadMessage('');
    
    const formData = new FormData();
    formData.append('model_name', config.ollama.model);
    
    try {
      const response = await axios.post<OllamaLoadResponse>(API_ENDPOINTS.OLLAMA_LOAD, formData);
      setModelLoadMessage(`Model ${config.ollama.model} load initiated: ${response.data.status}`);
      
      // Check updated status after load
      setTimeout(async () => {
        const statusResponse = await axios.get<OllamaStatus>(API_ENDPOINTS.OLLAMA_STATUS);
        setOllamaStatus(statusResponse.data);
      }, 2000);
    } catch (error) {
      console.error('Failed to load Ollama model:', error);
      setModelLoadMessage('Failed to load model. Ensure Ollama is running and model name is correct.');
    } finally {
      setLoadingOllama(false);
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Configuration</h2>
      
      {message && (
        <div className={`${styles.message} ${message.includes('success') ? styles.success : styles.error}`}>
          {message}
        </div>
      )}
      
      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.configSection}>
          <h3>Chat Settings</h3>
          
          <div className={styles.inputGroup}>
            <label>Provider</label>
            <select 
              value={config.chat.provider}
              onChange={(e) => handleChange('chat', 'provider', e.target.value)}
            >
              <option value="ollama">Ollama</option>
              <option value="openai">OpenAI</option>
            </select>
          </div>
          
          <div className={styles.inputGroup}>
            <label>Model</label>
            {config.chat.provider === 'ollama' ? (
              <select
                value={config.chat.model}
                onChange={(e) => {
                  handleChange('chat', 'model', e.target.value);
                  handleChange('ollama', 'model', e.target.value);
                }}
              >
                {ollamaStatus.models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
                {ollamaStatus.models.length === 0 && (
                  <option value={config.chat.model}>{config.chat.model}</option>
                )}
              </select>
            ) : (
              <select
                value={config.chat.model}
                onChange={(e) => handleChange('chat', 'model', e.target.value)}
              >
                {openaiModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
                {openaiModels.length === 0 && (
                  <option value={config.chat.model}>{config.chat.model}</option>
                )}
              </select>
            )}
          </div>
          
          <div className={styles.inputGroup}>
            <label>Temperature</label>
            <input 
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.chat.temperature}
              onChange={(e) => handleChange('chat', 'temperature', parseFloat(e.target.value))}
            />
            <span>{config.chat.temperature}</span>
          </div>
          
          <div className={styles.inputGroup}>
            <label>Chunks Limit</label>
            <input 
              type="number"
              value={config.chat.chunks_limit}
              onChange={(e) => handleChange('chat', 'chunks_limit', parseInt(e.target.value))}
            />
          </div>
        </div>
        
        <div className={styles.configSection}>
          <h3>OpenAI Settings</h3>
          
          <div className={styles.inputGroup}>
            <label>API Key</label>
            <input 
              type="password"
              value={config.openai.api_key}
              onChange={(e) => handleChange('openai', 'api_key', e.target.value)}
              placeholder="sk-..."
            />
          </div>
          
          <div className={styles.inputGroup}>
            <label>Base URL</label>
            <input 
              type="text"
              value={config.openai.base_url}
              onChange={(e) => handleChange('openai', 'base_url', e.target.value)}
            />
          </div>
          
          <div className={styles.statusSection}>
            <div className={styles.statusTitle}>Available Models:</div>
            <div className={styles.modelsList}>
              {loadingOpenai ? (
                <div>Loading models...</div>
              ) : openaiModels.length > 0 ? (
                <ul>
                  {openaiModels.slice(0, 5).map(model => (
                    <li key={model}>{model}</li>
                  ))}
                  {openaiModels.length > 5 && <li>+ {openaiModels.length - 5} more</li>}
                </ul>
              ) : (
                <div>{config.openai.api_key ? 'No models available' : 'Enter API key to see available models'}</div>
              )}
            </div>
          </div>
        </div>
        
        <div className={styles.configSection}>
          <h3>Ollama Settings</h3>
          
          <div className={styles.inputGroup}>
            <label>Ollama Host URL</label>
            <input 
              type="text"
              value={config.ollama.host}
              onChange={(e) => handleChange('ollama', 'host', e.target.value)}
            />
          </div>
          
          <div className={styles.statusIndicator}>
            Ollama Status: 
            {ollamaStatus.online ? 
              <span className={styles.online}> Online</span> : 
              <span className={styles.offline}> Offline {ollamaStatus.error ? `(${ollamaStatus.error})` : ''}</span>
            }
          </div>
          
          <div className={styles.inputGroup}>
            <label>Default Ollama Model</label>
            <input 
              type="text"
              value={config.ollama.model}
              onChange={(e) => handleChange('ollama', 'model', e.target.value)}
            />
            <button onClick={loadOllamaModel} disabled={loadingOllama || !ollamaStatus.online}>
              {loadingOllama ? 'Loading...' : 'Load Model'}
            </button>
            {modelLoadMessage && <p className={styles.modelMessage}>{modelLoadMessage}</p>}
          </div>
          
          <div className={styles.modelList}>
            <p>Available Models:</p>
            {ollamaStatus.models.length > 0 ? (
              <ul>
                {ollamaStatus.models.map(m => <li key={m}>{m}</li>)}
              </ul>
            ) : (
              <p>No models loaded or Ollama offline.</p>
            )}
          </div>
        </div>
        
        <div className={styles.configSection}>
          <h3>Milvus Settings</h3>
          
          <div className={styles.inputGroup}>
            <label>Host</label>
            <input 
              type="text"
              value={config.milvus.host}
              onChange={(e) => handleChange('milvus', 'host', e.target.value)}
            />
          </div>
          
          <div className={styles.inputGroup}>
            <label>Port</label>
            <input 
              type="number"
              value={config.milvus.port}
              onChange={(e) => handleChange('milvus', 'port', parseInt(e.target.value))}
            />
          </div>
          
          <div className={styles.inputGroup}>
            <label>Collection Name</label>
            <input 
              type="text"
              value={config.milvus.collection_name}
              onChange={(e) => handleChange('milvus', 'collection_name', e.target.value)}
            />
          </div>
        </div>
        
        <button type="submit" className={styles.button} disabled={loading}>
          {loading ? 'Saving...' : 'Save Configuration'}
        </button>
      </form>
    </div>
  );
};

export default Config; 