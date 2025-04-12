import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './UserAgentList.module.css'; // Optional CSS module

// Define the expected structure of an Agent object from the API
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
    created_at: string; // Assuming ISO string format from API
}

interface UserAgentListProps {
    userId: number;
    // Add callbacks for actions
    onEdit: (agent: Agent) => void;
    onDelete: (agentId: number) => void; // Assuming delete happens in parent
    onViewDetails: (agentId: number) => void; // Add prop for viewing details
    // Add function for selecting/viewing details later
    // onSelectAgent: (agentId: number) => void;
}

const UserAgentList: React.FC<UserAgentListProps> = ({ userId, onEdit, onDelete, onViewDetails }) => {
    const [agents, setAgents] = useState<Agent[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchAgents = async () => {
            if (!userId) return; // Should not happen if component used correctly
            
            setLoading(true);
            setError(null);
            console.log(`Fetching agents for user ID: ${userId}`);
            
            try {
                // API endpoint assumes user scope based on auth token
                const response = await axios.get<Agent[]>('/api/v1/agents/'); 
                setAgents(response.data);
                console.log('Fetched agents:', response.data);
            } catch (err) {
                console.error("Error fetching agents:", err);
                if (axios.isAxiosError(err) && err.response) {
                    setError(`Failed to fetch agents: ${err.response.data.detail || err.message}`);
                } else {
                    setError("An unknown error occurred while fetching agents.");
                }
            } finally {
                setLoading(false);
            }
        };

        fetchAgents();
    }, [userId]); // Refetch if userId changes (though it shouldn't in this context normally)

    if (loading) {
        return <div className={styles.loading}>Loading agents...</div>;
    }

    if (error) {
        return <div className={styles.error}>{error}</div>;
    }

    if (agents.length === 0) {
        return <p>You haven't created any agents yet.</p>;
    }

    return (
        <div className={styles.agentListContainer}>
            <h3>Your Agents:</h3>
            <ul className={styles.agentList}>
                {agents.map((agent) => (
                    <li key={agent.id} className={styles.agentListItem}>
                        <div className={styles.agentInfo}>
                            <strong>{agent.name}</strong> (ID: {agent.id})
                            <span className={agent.is_active ? styles.activeStatus : styles.inactiveStatus}>
                                {agent.is_active ? 'Active' : 'Inactive'}
                            </span>
                            {agent.llm_model && (
                                <span className={styles.llmInfo}> | LLM: {agent.llm_provider}/{agent.llm_model}</span>
                            )}
                        </div>
                        <div className={styles.agentActions}>
                            <button onClick={() => onViewDetails(agent.id)} className={styles.viewButton}>View Details</button>
                            <button onClick={() => onEdit(agent)}>Edit</button>
                            <button onClick={() => onDelete(agent.id)} className={styles.deleteButton}>Delete</button>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default UserAgentList; 