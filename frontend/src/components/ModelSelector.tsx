import React from 'react';
import styles from './ModelSelector.module.css';
import { useModelProviderSelection } from '../hooks/useModelProviderSelection';

// Define props for ModelSelector
interface ModelSelectorProps {
    isAuthReady: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ isAuthReady }) => {
  const {
    currentProvider,
    setCurrentProvider,
    currentModel,
    setCurrentModel,
    ollamaModels,
    openaiModels,
    loadingStatus,
    ollamaStatus,
    providerError,
  } = useModelProviderSelection(isAuthReady);

  return (
    <div className={styles.modelSelector}>
      {providerError && <p className={styles.errorMessage}>{providerError}</p>}
      <div className={styles.selectorGroup}>
        <label className={styles.selectorLabel}>
          Provider:
          <select
            value={currentProvider}
            onChange={(e) => setCurrentProvider(e.target.value)}
            className={styles.select}
            disabled={loadingStatus}
          >
            {ollamaStatus?.online === true && <option value="ollama">Ollama</option>}
            {ollamaStatus?.online === false && <option value="ollama" disabled>Ollama (Offline)</option>}
            <option value="openai">OpenAI</option>
            {loadingStatus && !ollamaStatus && <option value="" disabled>Checking providers...</option>}
          </select>
        </label>
      </div>

      <div className={styles.selectorGroup}>
        <label className={styles.selectorLabel}>
          Model:
          <select
            value={currentModel}
            onChange={(e) => setCurrentModel(e.target.value)}
            className={styles.select}
            disabled={loadingStatus || (currentProvider === 'ollama' && (!ollamaStatus?.online || ollamaModels.length === 0))}
          >
            {loadingStatus ? (
              <option value="">Loading models...</option>
            ) : currentProvider === 'ollama' ? (
              ollamaStatus?.online && ollamaModels.length > 0 ? (
                ollamaModels.map((ollamaModel) => (
                  <option key={ollamaModel} value={ollamaModel}>
                    {ollamaModel}
                  </option>
                ))
              ) : (
                ollamaStatus?.online === false ?
                <option value="">Ollama Offline</option> :
                <option value="">No Ollama models found</option>
              )
            ) : currentProvider === 'openai' ? (
              openaiModels.length > 0 ? (
                openaiModels.map((openaiModel) => (
                  <option key={openaiModel.id} value={openaiModel.id}>
                    {openaiModel.id}
                  </option>
                ))
              ) : (
                <option value="">No OpenAI models available</option>
              )
            ) : (
               <option value="">Select Provider</option>
            )}
          </select>
        </label>
      </div>
    </div>
  );
}; 