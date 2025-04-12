import React, { useState, useCallback } from 'react';
import axios from 'axios';
import styles from './MyAgentsPage.module.css'; // Optional: Create a CSS module for styling
import UserAgentList from '../components/UserAgentList'; // Import the list component
import AgentForm from '../components/AgentForm'; // Import the form component
import AgentDetailView from '../components/AgentDetailView'; // Import detail view

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

interface MyAgentsPageProps {
    userId: number | null; // Pass user ID if needed for fetching agents
}

const MyAgentsPage: React.FC<MyAgentsPageProps> = ({ userId }) => {
    
    const [viewMode, setViewMode] = useState<'list' | 'form' | 'detail'>('list');
    const [currentAgentId, setCurrentAgentId] = useState<number | null>(null); // For edit/view
    const [refreshListKey, setRefreshListKey] = useState<number>(0); // Key to force refresh UserAgentList
    const [pageError, setPageError] = useState<string | null>(null); // For page-level errors like delete

    const handleCreateAgent = () => {
        setCurrentAgentId(null);
        setViewMode('form');
        setPageError(null);
    };

    const handleEditAgent = (agent: Agent) => {
        setCurrentAgentId(agent.id);
        setViewMode('form');
        setPageError(null);
    };

    const handleViewDetails = (agentId: number) => {
        setCurrentAgentId(agentId);
        setViewMode('detail');
        setPageError(null);
    };

    const handleDeleteAgent = async (agentId: number) => {
        setPageError(null);
        if (window.confirm(`Are you sure you want to delete agent ID ${agentId}?`)) {
            try {
                console.log(`Deleting agent ${agentId}...`);
                await axios.delete(`/api/agents/${agentId}`);
                console.log(`Agent ${agentId} deleted.`);
                // Force refresh the list by changing the key
                setRefreshListKey(prev => prev + 1);
                // If deleting the currently viewed agent, go back to list
                if (viewMode === 'detail' && currentAgentId === agentId) {
                    setViewMode('list');
                    setCurrentAgentId(null);
                }
            } catch (err) {
                 console.error("Error deleting agent:", err);
                 const errorMsg = axios.isAxiosError(err) && err.response 
                                ? `Failed to delete agent: ${err.response.data.detail || err.message}`
                                : "An unknown error occurred while deleting the agent.";
                 setPageError(errorMsg);
            }
        }
    };
    
    // Wrap in useCallback to prevent re-renders of AgentForm if MyAgentsPage re-renders
    const handleFormSubmit = useCallback((submittedAgent: Agent) => {
        console.log('Form submitted:', submittedAgent);
        setViewMode('list'); // Go back to list after submit
        setCurrentAgentId(null);
        // Force refresh the list after add/edit
        setRefreshListKey(prev => prev + 1);
    }, []); // Empty dependency array as it doesn't depend on state within this scope directly

    const handleFormCancel = useCallback(() => {
        setViewMode('list'); // Go back to list on cancel
        setCurrentAgentId(null);
        setPageError(null);
    }, []);

    const handleBackToList = useCallback(() => {
        setViewMode('list');
        setCurrentAgentId(null);
        setPageError(null);
    }, []);
    
    if (userId === null) {
        // This shouldn't happen if routing/auth is correct, but good practice
        return <p>Error: User not identified.</p>;
    }
    
    let content;
    if (viewMode === 'form') {
        const agentToEdit = currentAgentId ? { id: currentAgentId } : null; // Pass ID or null
        content = (
            <AgentForm 
                key={currentAgentId || 'create'} 
                userId={userId}
                agentIdToEdit={currentAgentId} // NEW PROP
                onFormSubmit={handleFormSubmit}
                onCancel={handleFormCancel}
            />
        );
    } else if (viewMode === 'detail' && currentAgentId !== null) {
        content = (
            <AgentDetailView 
                agentId={currentAgentId}
                userId={userId}
                onBackToList={handleBackToList}
            />
        );
    } else { // Default to list view
         content = (
            <div> {/* Wrapper div for list and header */} 
                <div className={styles.header}>
                     <h2>My Agents</h2>
                     <button onClick={handleCreateAgent} className={styles.createButton}>
                         + Create New Agent
                     </button>
                </div>
                
                {/* Render the agent list, passing the userId */}
                <UserAgentList 
                    key={refreshListKey} // Use key to trigger re-fetch
                    userId={userId} 
                    onEdit={handleEditAgent}
                    onDelete={handleDeleteAgent}
                    onViewDetails={handleViewDetails} // Pass view details handler
                />
            </div>
         );
    }
    
    return (
        <div className={styles.container}> {/* Use optional CSS module */} 
            {pageError && <div className={styles.pageError}>{pageError}</div>}
            
            {content}
        </div>
    );
};

export default MyAgentsPage; 