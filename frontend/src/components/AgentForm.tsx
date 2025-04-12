import React, { useState, useEffect } from 'react';
import axios, { AxiosResponse } from 'axios';
import styles from './AgentForm.module.css';
import CapabilityManager from './CapabilityManager';

// Re-use Agent interface definition (or import from a shared types file)
interface Agent {
    id: number;
    name: string;
    persona?: string;
    goals?: string;
    base_prompt?: string;
    is_active: boolean;
    llm_provider?: string | null;
    llm_model?: string | null;
    owner_user_id: number;
    created_at: string; 
}

interface AgentFormProps {
    userId: number;
    agentIdToEdit?: number | null; // Changed from agentToEdit
    onFormSubmit: (agent: Agent) => void; // Callback after successful submit
    onCancel: () => void; // Callback to cancel/close the form
}

// Placeholder for LLM options - TODO: Fetch from config API or hardcode more realistically
const llmOptions: { [provider: string]: string[] } = {
    'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
    'ollama': ['llama3', 'mistral', 'codellama'],
    // Add other providers as needed
};

const AgentForm: React.FC<AgentFormProps> = ({ 
    userId, 
    agentIdToEdit, // Changed prop name
    onFormSubmit, 
    onCancel 
}) => {
    const [formData, setFormData] = useState<Partial<Agent>>({
        name: '',
        persona: '',
        goals: '',
        base_prompt: '',
        is_active: true,
        llm_provider: null,
        llm_model: null,
    });
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [fetchingDetails, setFetchingDetails] = useState<boolean>(false); // Loading state for fetching details
    const [availableModels, setAvailableModels] = useState<string[]>([]);

    const isEditing = Boolean(agentIdToEdit);

    // Fetch agent details if editing
    useEffect(() => {
        if (isEditing && agentIdToEdit) {
            const fetchDetails = async () => {
                setFetchingDetails(true);
                setError(null);
                try {
                    console.log(`Fetching details for agent to edit: ${agentIdToEdit}`);
                    const response = await axios.get<Agent>(`/api/v1/agents/${agentIdToEdit}`);
                    const agentData = response.data;
                    setFormData({
                        name: agentData.name,
                        persona: agentData.persona || '',
                        goals: agentData.goals || '',
                        base_prompt: agentData.base_prompt || '',
                        is_active: agentData.is_active,
                        llm_provider: agentData.llm_provider,
                        llm_model: agentData.llm_model,
                    });
                     // Set initial available models if provider exists
                    if (agentData.llm_provider && llmOptions[agentData.llm_provider]) {
                        setAvailableModels(llmOptions[agentData.llm_provider]);
                    }
                } catch (err) {
                    console.error("Error fetching agent details for edit:", err);
                    setError("Failed to load agent details for editing.");
                    // Optionally call onCancel or disable form?
                } finally {
                    setFetchingDetails(false);
                }
            };
            fetchDetails();
        } else {
            // Reset form if creating
            setFormData({
                name: '',
                persona: '',
                goals: '',
                base_prompt: '',
                is_active: true,
                llm_provider: null,
                llm_model: null,
            });
            setAvailableModels([]);
        }
    }, [agentIdToEdit, isEditing]);

    // Update available models when provider changes
    useEffect(() => {
        if (formData.llm_provider && llmOptions[formData.llm_provider]) {
            setAvailableModels(llmOptions[formData.llm_provider]);
            // Reset model if the current one isn't valid for the new provider
            if (!llmOptions[formData.llm_provider].includes(formData.llm_model || '')) {
                setFormData(prev => ({ ...prev, llm_model: null }));
            }
        } else {
            setAvailableModels([]);
            setFormData(prev => ({ ...prev, llm_model: null })); // Clear model if provider cleared
        }
    }, [formData.llm_provider]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { name, value, type } = e.target;
        
        if (type === 'checkbox') {
            const { checked } = e.target as HTMLInputElement;
            setFormData(prev => ({ ...prev, [name]: checked }));
        } else {
            // Handle clearing optional fields (provider/model selects)
            const finalValue = value === '' && (name === 'llm_provider' || name === 'llm_model') ? null : value;
            setFormData(prev => ({ ...prev, [name]: finalValue }));
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        const payload: any = { ...formData };
        if (!isEditing) {
            payload.owner_user_id = userId;
        }

        try {
            let response: AxiosResponse<Agent>;
            if (isEditing && agentIdToEdit) {
                console.log(`Updating agent ${agentIdToEdit} with payload:`, payload);
                response = await axios.put<Agent>(`/api/v1/agents/${agentIdToEdit}`, payload);
            } else {
                console.log(`Creating new agent with payload:`, payload);
                response = await axios.post<Agent>('/api/v1/agents/', payload);
            }
            console.log('Agent form submission successful:', response.data);
            onFormSubmit(response.data); 
        } catch (err) {
            console.error("Error submitting agent form:", err);
             if (axios.isAxiosError(err) && err.response) {
                setError(`Failed to save agent: ${err.response.data.detail || err.message}`);
            } else {
                setError("An unknown error occurred while saving the agent.");
            }
        } finally {
            setLoading(false);
        }
    };

    // Add loading indicator while fetching details
    if (fetchingDetails) {
         return <div className={styles.loading}>Loading agent details...</div>;
    }
    
    return (
        <div className={styles.formContainer}>
             {/* Disable form if loading details failed? */} 
            <h3>{isEditing ? 'Edit Agent' : 'Create New Agent'}</h3>
            <form onSubmit={handleSubmit} className={styles.agentForm}>
                <div className={styles.formGroup}>
                    <label htmlFor="name">Name:</label>
                    <input 
                        type="text" 
                        id="name" 
                        name="name" 
                        value={formData.name || ''}
                        onChange={handleChange} 
                        required 
                    />
                </div>

                <div className={styles.formGroup}>
                    <label htmlFor="persona">Persona:</label>
                    <textarea 
                        id="persona" 
                        name="persona" 
                        value={formData.persona || ''}
                        onChange={handleChange} 
                        rows={3}
                    />
                </div>

                <div className={styles.formGroup}>
                    <label htmlFor="goals">Goals:</label>
                    <textarea 
                        id="goals" 
                        name="goals" 
                        value={formData.goals || ''}
                        onChange={handleChange} 
                        rows={3}
                    />
                </div>
                
                <div className={styles.formGroup}>
                    <label htmlFor="base_prompt">Base Prompt:</label>
                    <textarea 
                        id="base_prompt" 
                        name="base_prompt" 
                        value={formData.base_prompt || ''}
                        onChange={handleChange} 
                        rows={5}
                    />
                </div>

                 <div className={styles.formGroupRow}> {/* Row for LLM config */} 
                    <div className={styles.formGroup}>
                        <label htmlFor="llm_provider">LLM Provider (Optional):</label>
                        <select 
                            id="llm_provider" 
                            name="llm_provider" 
                            value={formData.llm_provider || ''} 
                            onChange={handleChange}
                        >
                            <option value="">-- Use Default --</option>
                            {Object.keys(llmOptions).map(provider => (
                                <option key={provider} value={provider}>{provider}</option>
                            ))}
                        </select>
                    </div>

                    <div className={styles.formGroup}>
                        <label htmlFor="llm_model">LLM Model (Optional):</label>
                        <select 
                            id="llm_model" 
                            name="llm_model" 
                            value={formData.llm_model || ''} 
                            onChange={handleChange}
                            disabled={!formData.llm_provider || availableModels.length === 0}
                        >
                            <option value="">-- Use Default --</option>
                            {availableModels.map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>
                 </div>

                <div className={styles.formGroupCheckbox}>
                    <label htmlFor="is_active">Active:</label>
                    <input 
                        type="checkbox" 
                        id="is_active" 
                        name="is_active" 
                        checked={formData.is_active ?? true} 
                        onChange={handleChange} 
                    />
                </div>

                {/* Capability Manager (only show when editing) */} 
                {isEditing && agentIdToEdit && (
                    <div className={styles.capabilitiesSection}>
                        <CapabilityManager agentId={agentIdToEdit} />
                    </div>
                )}

                {error && <p className={styles.error}>{error}</p>}

                <div className={styles.formActions}>
                    {/* Submit/Cancel Buttons */} 
                     <button type="submit" disabled={loading || fetchingDetails} className={styles.submitButton}>
                        {loading ? 'Saving...' : (isEditing ? 'Update Agent' : 'Create Agent')}
                    </button>
                    <button type="button" onClick={onCancel} disabled={loading || fetchingDetails} className={styles.cancelButton}>
                        Cancel
                    </button>
                </div>
            </form>
        </div>
    );
};

export default AgentForm; 