import { useState, useEffect } from 'react';
import axios, { AxiosError } from 'axios';
import API_ENDPOINTS from '../config/api'; // Assuming API endpoints are defined here

// Types moved or defined here for clarity
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
    compatible_models: OpenAIModelData[];
    suggested_default: string;
    error?: string;
}

// Define the structure for the hook's return value
interface UseModelProviderSelectionReturn {
    currentModel: string;
    setCurrentModel: (model: string) => void;
    currentProvider: string;
    setCurrentProvider: (provider: string) => void;
    ollamaModels: string[];
    openaiModels: OpenAIModelData[];
    loadingStatus: boolean;
    ollamaStatus: OllamaStatusResponse | null;
    providerError: string | null;
}

// Define default values
const DEFAULT_PROVIDER = 'openai';
const DEFAULT_MODEL = 'gpt-4o';

export function useModelProviderSelection(isAuthReady: boolean): UseModelProviderSelectionReturn {
    // State for current selection, initialized from localStorage or defaults
    const [currentProvider, _setCurrentProvider] = useState<string>(() => {
        return localStorage.getItem('chat_provider') || DEFAULT_PROVIDER;
    });

    // State for available models and status
    const [ollamaModels, setOllamaModels] = useState<string[]>([]);
    const [openaiModels, setOpenaiModels] = useState<OpenAIModelData[]>([]);
    const [loadingStatus, setLoadingStatus] = useState(false);
    const [ollamaStatus, setOllamaStatus] = useState<OllamaStatusResponse | null>(null);
    const [providerError, setProviderError] = useState<string | null>(null);
    
    // Recalculate default model based on current provider if needed
    const [currentModel, _setCurrentModel] = useState<string>(() => {
        const storedModel = localStorage.getItem('chat_model');
        if (storedModel) return storedModel;
        // If no stored model, return default based on initial provider
        const initialProvider = localStorage.getItem('chat_provider') || DEFAULT_PROVIDER;
        return initialProvider === 'ollama' ? 'llama3' : DEFAULT_MODEL; // Adjust ollama default if needed
    });

    // Effect to fetch provider status and models, depends on isAuthReady and currentProvider
    useEffect(() => {
        // Only run if auth is ready
        if (isAuthReady) {
             console.log("useModelProviderSelection: isAuthReady=true, checking providers...");
            // No timeout needed now
            const checkProviders = async () => {
                setLoadingStatus(true);
                setProviderError(null);
                let ollamaCheckSuccess = false;
                let openaiCheckSuccess = false;
                
                setOllamaModels([]);
                setOpenaiModels([]);
                setOllamaStatus(null);

                let suggestedOpenAIDefault: string | undefined = undefined;

                try {
                    const response = await axios.get<OllamaStatusResponse>(API_ENDPOINTS.OLLAMA_STATUS);
                    setOllamaStatus(response.data);
                    if (response.data.online && response.data.models) {
                        setOllamaModels(response.data.models);
                        ollamaCheckSuccess = true;
                    } else if (response.data.error) {
                         setProviderError(`Ollama Error: ${response.data.error}`);
                     } else if (!response.data.online) {
                        // Optionally set a specific error or rely on status object
                        // setProviderError("Ollama provider is offline.");
                     }
                } catch (error) {
                     console.error('Failed to check Ollama status:', error);
                     if (axios.isAxiosError(error) && error.response?.status === 401) {
                         setProviderError("Unauthorized to check Ollama status.");
                     } else {
                         setProviderError("Failed to contact Ollama provider.");
                     }
                }

                try {
                    const response = await axios.get<OpenAIModelsApiResponse>(API_ENDPOINTS.OPENAI_MODELS);
                    if (response.data.compatible_models) {
                        setOpenaiModels(response.data.compatible_models);
                        openaiCheckSuccess = true;
                        suggestedOpenAIDefault = response.data.suggested_default;
                    } else if (response.data.error) {
                         setProviderError(prev => prev ? `${prev} | OpenAI Error: ${response.data.error}` : `OpenAI Error: ${response.data.error}`);
                     }
                } catch (error) {
                    console.error('Failed to get OpenAI models:', error);
                     if (axios.isAxiosError(error) && error.response?.status === 401) {
                         setProviderError(prev => prev ? `${prev} | Unauthorized for OpenAI` : "Unauthorized for OpenAI");
                     } else {
                         setProviderError(prev => prev ? `${prev} | Failed to contact OpenAI` : "Failed to contact OpenAI");
                     }
                } finally {
                    setLoadingStatus(false);
                    // ---> Put back auto-selection, running AFTER state is likely set <----
                    // Use a slight delay or check flags to ensure state is updated?
                    // For simplicity, we assume state updates from above apply before this runs
                    // (usually true in React batches, but can be tricky)
                    
                    // Get the latest state directly INSIDE finally block for comparison
                    const latestModel = localStorage.getItem('chat_model') || 
                                      (localStorage.getItem('chat_provider') || DEFAULT_PROVIDER) === 'ollama' ? 'llama3' : DEFAULT_MODEL;
                    const latestProvider = localStorage.getItem('chat_provider') || DEFAULT_PROVIDER;
                    
                    console.log(`Finally block: Provider=${latestProvider}, CheckSuccess=${openaiCheckSuccess}, #Models=${openaiModels.length}, CurrentModel=${latestModel}, Suggested=${suggestedOpenAIDefault}`);

                    // Auto-select a default model if the current one isn't valid for the loaded provider
                    if (latestProvider === 'ollama' && ollamaCheckSuccess && ollamaModels.length > 0 && !ollamaModels.includes(latestModel)) {
                        const newModel = ollamaModels[0];
                        console.log(`Auto-selecting Ollama model: ${newModel} (was ${latestModel})`);
                        setCurrentModel(newModel); // Use the wrapper setter
                     } else if (latestProvider === 'openai' && openaiCheckSuccess && openaiModels.length > 0 && !openaiModels.some(m => m.id === latestModel)) {
                         const newModel = suggestedOpenAIDefault || openaiModels[0]?.id || DEFAULT_MODEL;
                         console.log(`Auto-selecting OpenAI model: ${newModel} (was ${latestModel}, suggested: ${suggestedOpenAIDefault})`);
                         setCurrentModel(newModel); // Use the wrapper setter
                     }
                     // ---> End Auto-selection logic ---
                }
            };
            checkProviders();
        } else {
             console.log("useModelProviderSelection: isAuthReady=false, delaying provider check.");
             // Reset state if auth not ready?
             setOllamaStatus(null);
             setOllamaModels([]);
             setOpenaiModels([]);
             setProviderError("Authentication not ready.");
        }

        // Cleanup function remains the same (if any needed beyond AbortController)
        return () => {};

    // Dependency array includes isAuthReady and currentProvider
    }, [isAuthReady, currentProvider]); 

    // Wrappers for setters to also update localStorage and handle side effects
    const setCurrentProvider = (provider: string) => {
        localStorage.setItem('chat_provider', provider);
        _setCurrentProvider(provider);
        // Reset model when provider changes - the useEffect above will handle fetching
        // and selecting a new appropriate default model after fetch completes.
        // Set a temporary/placeholder default immediately?
        _setCurrentModel(provider === 'ollama' ? 'llama3' : DEFAULT_MODEL);
        localStorage.removeItem('chat_model'); // Remove old model pref
    };

    const setCurrentModel = (model: string) => {
        localStorage.setItem('chat_model', model);
        _setCurrentModel(model);
    };

    return {
        currentModel,
        setCurrentModel,
        currentProvider,
        setCurrentProvider,
        ollamaModels,
        openaiModels,
        loadingStatus,
        ollamaStatus,
        providerError,
    };
} 