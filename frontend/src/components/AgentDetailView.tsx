import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './AgentDetailView.module.css';
import LogExplorer from './LogExplorer';
import MemoryExplorer from './MemoryExplorer';
import VisualTaskTrace from './VisualTaskTrace';
// Import shared type
import { ScratchpadEntry } from '../types/agentTypes';

// Re-use Agent interface (or import from shared types)
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

interface AgentDetailViewProps {
    agentId: number;
    userId: number;
    onBackToList: () => void; // Function to navigate back
}

const AgentDetailView: React.FC<AgentDetailViewProps> = ({ agentId, userId, onBackToList }) => {
    const [agent, setAgent] = useState<Agent | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    // State for Task Trace
    const [taskTraceData, setTaskTraceData] = useState<ScratchpadEntry[] | null>(null);
    const [isTraceLoading, setIsTraceLoading] = useState<boolean>(false);
    const [traceError, setTraceError] = useState<string | null>(null);
    const [showTrace, setShowTrace] = useState<boolean>(false); // Control visibility

    useEffect(() => {
        const fetchAgentDetails = async () => {
            setLoading(true);
            setError(null);
            console.log(`Fetching details for agent ID: ${agentId}`);
            try {
                const response = await axios.get<Agent>(`/api/agents/${agentId}`);
                setAgent(response.data);
                console.log('Fetched agent details:', response.data);
            } catch (err) {
                console.error("Error fetching agent details:", err);
                 const errorMsg = axios.isAxiosError(err) && err.response 
                                ? `Failed to load agent details: ${err.response.data.detail || err.message}`
                                : "An unknown error occurred.";
                setError(errorMsg);
            } finally {
                setLoading(false);
            }
        };

        fetchAgentDetails();
    }, [agentId]);

    // Function to fetch the latest task trace
    const fetchLatestTrace = async () => {
        if (!agentId) return;
        setIsTraceLoading(true);
        setTraceError(null);
        setTaskTraceData(null); // Clear previous trace
        setShowTrace(true); // Show loading/error or trace when button clicked
        console.log(`Fetching latest task trace for agent ID: ${agentId}`);
        try {
            // *** TODO: This endpoint needs to be implemented in the backend ***
            // It should return an object like { scratchpad: ScratchpadEntry[] }
            const response = await axios.get<{ scratchpad: ScratchpadEntry[] }>(`/api/agents/${agentId}/runs/latest`);
            setTaskTraceData(response.data.scratchpad || []);
            console.log('Fetched task trace:', response.data);
            if (!response.data.scratchpad || response.data.scratchpad.length === 0) {
                 setTraceError('No trace data found for the latest run.');
            }
        } catch (err) {
            console.error("Error fetching latest task trace:", err);
            const errorMsg = axios.isAxiosError(err) && err.response
                           ? `Failed to load task trace: ${err.response.status} ${err.response.data.detail || err.message}`
                           : "An unknown error occurred while fetching trace.";
            setTraceError(errorMsg);
            setTaskTraceData(null);
        } finally {
            setIsTraceLoading(false);
        }
    };

    if (loading) {
        return <div className={styles.loading}>Loading agent details...</div>;
    }

    if (error) {
        return <div className={styles.error}>{error}</div>;
    }

    if (!agent) {
        return <div className={styles.error}>Agent not found.</div>;
    }

    return (
        <div className={styles.detailViewContainer}>
            <div className={styles.header}>
                <h2>Agent Details: {agent.name} (ID: {agent.id})</h2>
                <button onClick={onBackToList} className={styles.backButton}>&larr; Back to List</button>
            </div>
            
            {/* Agent Info Section */}
            <div className={styles.infoGrid}>
                <div><strong>Status:</strong> {agent.is_active ? 'Active' : 'Inactive'}</div>
                <div><strong>Owner User ID:</strong> {agent.owner_user_id}</div>
                <div><strong>Created:</strong> {new Date(agent.created_at).toLocaleString()}</div>
                <div>
                    <strong>LLM Config:</strong> 
                    {agent.llm_provider && agent.llm_model 
                        ? `${agent.llm_provider} / ${agent.llm_model}` 
                        : 'Using Default'}
                </div>
                {agent.persona && <div className={styles.fullWidth}><strong>Persona:</strong> <p>{agent.persona}</p></div>}
                {agent.goals && <div className={styles.fullWidth}><strong>Goals:</strong> <p>{agent.goals}</p></div>}
                {agent.base_prompt && <div className={styles.fullWidth}><strong>Base Prompt:</strong> <pre>{agent.base_prompt}</pre></div>}
            </div>

            <hr className={styles.divider} />

            {/* Log Explorer Integration */} 
            <div className={styles.explorerSection}>
                <LogExplorer agentId={agentId} />
            </div>

            <hr className={styles.divider} />

            {/* Memory Explorer Integration */} 
            <div className={styles.explorerSection}>
                 <MemoryExplorer agentId={agentId} userId={userId} />
            </div>

            <hr className={styles.divider} />

            {/* Visual Task Trace Section */}
            <div className={styles.explorerSection}>
                <h4>Latest Task Trace</h4>
                <button 
                    onClick={fetchLatestTrace} 
                    disabled={isTraceLoading} 
                    className={styles.traceButton}
                >
                    {isTraceLoading ? 'Loading Trace...' : 'View Latest Task Trace'}
                </button>
                
                {showTrace && (
                    <div className={styles.traceDisplayArea}>
                        {isTraceLoading && <div className={styles.loading}>Loading trace...</div>}
                        {traceError && <div className={styles.error}>{traceError}</div>}
                        {!isTraceLoading && !traceError && taskTraceData && (
                            <VisualTaskTrace scratchpadData={taskTraceData} />
                        )}
                         {!isTraceLoading && !traceError && taskTraceData === null && !showTrace && (
                            <p>Click the button to load the trace.</p>
                        )}
                    </div>
                )}
            </div>
            
             {/* TODO: Integrate VisualTaskTrace here later */} 
        </div>
    );
};

export default AgentDetailView; 