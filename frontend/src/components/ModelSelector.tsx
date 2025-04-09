import React, { useEffect, useState } from 'react';
import axios from 'axios';
import styles from './ModelSelector.module.css';
import API_ENDPOINTS from '../config/api';

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

interface OpenAIModelData {
  id: string;
  created: number;
}

interface OpenAIModelsApiResponse {
  all_models: OpenAIModelData[];
  suggested_default: string;
  error?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  provider,
  model,
  onProviderChange,
  onModelChange
}) => {
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [openaiModels, setOpenaiModels] = useState<OpenAIModelData[]>([]);
  const [loading, setLoading] = useState(false);
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatusResponse | null>(null);

  useEffect(() => {
    const checkProviders = async () => {
      setLoading(true);
      try {
        const response = await axios.get<OllamaStatusResponse>(API_ENDPOINTS.OLLAMA_STATUS);
        setOllamaStatus(response.data);
        if (response.data.online && response.data.models) {
          setOllamaModels(response.data.models);
        }
      } catch (error) {
        console.error('Failed to check Ollama status:', error);
      }

      try {
        const response = await axios.get<OpenAIModelsApiResponse>(API_ENDPOINTS.OPENAI_MODELS);
        setOpenaiModels(response.data.all_models || []);
      } catch (error) {
        console.error('Failed to get OpenAI models:', error);
      } finally {
        setLoading(false);
      }
    };

    checkProviders();
  }, []);

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
            {ollamaStatus?.online && <option value="ollama">Ollama</option>}
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
            disabled={loading || (provider === 'ollama' && !ollamaStatus?.online)}
          >
            {loading ? (
              <option value="">Loading models...</option>
            ) : provider === 'ollama' ? (
              ollamaModels.length > 0 ? (
                ollamaModels.map((ollamaModel) => (
                  <option key={ollamaModel} value={ollamaModel}>
                    {ollamaModel}
                  </option>
                ))
              ) : (
                ollamaStatus?.online ?
                <option value="">No Ollama models found</option> :
                <option value="">Ollama offline</option>
              )
            ) : (
              openaiModels.length > 0 ? (
                openaiModels.map((openaiModel) => (
                  <option key={openaiModel.id} value={openaiModel.id}>
                    {openaiModel.id}
                  </option>
                ))
              ) : (
                <option value="">No OpenAI models available</option>
              )
            )}
          </select>
        </label>
      </div>
    </div>
  );
}; 