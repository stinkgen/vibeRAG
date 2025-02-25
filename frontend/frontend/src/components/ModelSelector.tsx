import React, { useEffect, useState } from 'react';
import axios from 'axios';
import styles from './ModelSelector.module.css';

interface ModelSelectorProps {
  provider: string;
  model: string;
  onProviderChange: (provider: string) => void;
  onModelChange: (model: string) => void;
}

interface OllamaStatusResponse {
  online: boolean;
  models: string[];
  error?: string;
}

interface OpenAIModelsResponse {
  models: string[];
  error?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  provider,
  model,
  onProviderChange,
  onModelChange
}) => {
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [openaiModels, setOpenaiModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      try {
        if (provider === 'ollama') {
          const response = await axios.get<OllamaStatusResponse>('http://localhost:8000/api/providers/ollama/status');
          if (response.data.models) {
            setOllamaModels(response.data.models);
          }
        } else if (provider === 'openai') {
          const response = await axios.get<OpenAIModelsResponse>('http://localhost:8000/api/providers/openai/models');
          if (response.data.models) {
            setOpenaiModels(response.data.models);
          }
        }
      } catch (error) {
        console.error('Failed to fetch models:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [provider]);

  return (
    <div className={styles.modelSelector}>
      <div className={styles.selectorGroup}>
        <label className={styles.selectorLabel}>
          Provider:
          <select
            value={provider}
            onChange={(e) => onProviderChange(e.target.value)}
            className={styles.select}
          >
            <option value="ollama">Ollama</option>
            <option value="openai">OpenAI</option>
          </select>
        </label>
      </div>

      <div className={styles.selectorGroup}>
        <label className={styles.selectorLabel}>
          Model:
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            className={styles.select}
            disabled={loading}
          >
            {loading ? (
              <option value="">Loading models...</option>
            ) : provider === 'ollama' ? (
              ollamaModels.length > 0 ? (
                ollamaModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))
              ) : (
                <option value={model}>{model || 'No models available'}</option>
              )
            ) : openaiModels.length > 0 ? (
              openaiModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))
            ) : (
              <option value={model}>{model || 'No models available'}</option>
            )}
          </select>
        </label>
      </div>
    </div>
  );
}; 