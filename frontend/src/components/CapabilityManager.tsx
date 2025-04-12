import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import styles from './CapabilityManager.module.css';

// Define Tool Info structure from API
interface ToolInfo {
    name: string;
    description: string;
    // parameters are excluded based on backend model
}

interface CapabilityManagerProps {
    agentId: number;
    // We assume the parent (AgentForm) handles overall save/cancel
}

const CapabilityManager: React.FC<CapabilityManagerProps> = ({ agentId }) => {
    const [availableTools, setAvailableTools] = useState<ToolInfo[]>([]);
    const [assignedTools, setAssignedTools] = useState<Set<string>>(new Set());
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchCapabilities = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [allToolsResponse, agentCapsResponse] = await Promise.all([
                axios.get<ToolInfo[]>('/api/v1/agents/tools/'),
                axios.get<string[]>(`/api/v1/agents/${agentId}/capabilities`)
            ]);
            
            setAvailableTools(allToolsResponse.data);
            setAssignedTools(new Set(agentCapsResponse.data));
            console.log('Fetched available tools:', allToolsResponse.data);
            console.log('Fetched assigned capabilities:', agentCapsResponse.data);

        } catch (err) {
            console.error("Error fetching capabilities data:", err);
            const errorMsg = axios.isAxiosError(err) && err.response 
                            ? `Failed to load tool/capability data: ${err.response.data.detail || err.message}`
                            : "An unknown error occurred.";
            setError(errorMsg);
        } finally {
            setLoading(false);
        }
    }, [agentId]);

    useEffect(() => {
        fetchCapabilities();
    }, [fetchCapabilities]);

    const handleCapabilityChange = async (toolName: string, isChecked: boolean) => {
        setError(null); // Clear previous errors
        const originalAssignedTools = new Set(assignedTools); // Keep backup
        
        // Optimistically update UI state
        const updatedAssignedTools = new Set(assignedTools);
        if (isChecked) {
            updatedAssignedTools.add(toolName);
        } else {
            updatedAssignedTools.delete(toolName);
        }
        setAssignedTools(updatedAssignedTools);

        // Call API
        try {
            if (isChecked) {
                console.log(`Adding capability: ${toolName} to agent ${agentId}`);
                await axios.post(`/api/v1/agents/${agentId}/capabilities`, { tool_name: toolName });
            } else {
                console.log(`Removing capability: ${toolName} from agent ${agentId}`);
                await axios.delete(`/api/v1/agents/${agentId}/capabilities/${toolName}`);
            }
            console.log(`Capability ${toolName} ${isChecked ? 'added' : 'removed'} successfully.`);
            // UI state is already updated
        } catch (err) {
            console.error(`Error ${isChecked ? 'adding' : 'removing'} capability ${toolName}:`, err);
            const errorMsg = axios.isAxiosError(err) && err.response 
                            ? `API Error: ${err.response.data.detail || err.message}`
                            : "An unknown error occurred.";
            setError(`Failed to update capability ${toolName}. ${errorMsg}`);
            // Revert optimistic UI update on error
            setAssignedTools(originalAssignedTools);
        }
    };

    if (loading) {
        return <div className={styles.loading}>Loading capabilities...</div>;
    }

    // Don't render error here directly, maybe show inline with checkboxes?
    // if (error) {
    //     return <div className={styles.error}>{error}</div>;
    // }

    return (
        <div className={styles.capabilityManagerContainer}>
            <h4>Agent Capabilities (Tools):</h4>
            {error && <p className={styles.error}>{error}</p>} {/* Show error above list */}
            {availableTools.length === 0 && !loading && (
                <p>No tools available in the system.</p>
            )}
            <ul className={styles.toolList}>
                {availableTools.map((tool) => (
                    <li key={tool.name} className={styles.toolListItem}>
                        <input 
                            type="checkbox" 
                            id={`tool-${tool.name}`}
                            checked={assignedTools.has(tool.name)}
                            onChange={(e) => handleCapabilityChange(tool.name, e.target.checked)}
                        />
                        <label htmlFor={`tool-${tool.name}`}>{tool.name}</label>
                        <p className={styles.toolDescription}>{tool.description}</p>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default CapabilityManager; 