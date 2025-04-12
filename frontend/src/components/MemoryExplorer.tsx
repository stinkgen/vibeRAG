import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import styles from './MemoryExplorer.module.css';
import { useDebounce } from '../hooks/useDebounce'; // Assuming a debounce hook exists

// Define AgentMemory structure from API (matches AgentMemoryResponse)
interface AgentMemory {
    id: number;
    agent_id: number;
    memory_type: string;
    content: string;
    importance: number;
    timestamp: string; // ISO string format
    related_memory_ids?: string | null; // Assuming JSON string or null
    // Embedding not typically shown in UI
}

interface MemoryExplorerProps {
    agentId: number;
    userId: number; // Needed for ownership checks potentially, though API handles it now
}

const MemoryExplorer: React.FC<MemoryExplorerProps> = ({ agentId, userId }) => {
    const [memories, setMemories] = useState<AgentMemory[]>([]);
    const [searchQuery, setSearchQuery] = useState<string>('');
    const [loadingSearch, setLoadingSearch] = useState<boolean>(false);
    const [searchError, setSearchError] = useState<string | null>(null);
    const [manualAddError, setManualAddError] = useState<string | null>(null);
    const [deleteError, setDeleteError] = useState<string | null>(null);
    const [loadingAction, setLoadingAction] = useState<boolean>(false); // For add/delete actions

    // State for manual add form
    const [manualContentType, setManualContentType] = useState<string>('manual_fact');
    const [manualContent, setManualContent] = useState<string>('');
    const [manualImportance, setManualImportance] = useState<number>(0.7);

    const debouncedSearchQuery = useDebounce(searchQuery, 500); // Debounce search input by 500ms

    // Fetch memories when debounced query changes
    const fetchMemories = useCallback(async (query: string) => {
        if (!query) { // Don't search on empty query? Or show recent?
            setMemories([]);
            setLoadingSearch(false);
            return;
        }
        setLoadingSearch(true);
        setSearchError(null);
        console.log(`Searching memories for agent ${agentId} with query: ${query}`);

        try {
            const params = new URLSearchParams();
            params.append('query_text', query);
            params.append('limit', '10'); // Example limit
            
            const response = await axios.get<AgentMemory[]>(
                `/api/agents/${agentId}/memory`,
                { params }
            );
            setMemories(response.data);
            console.log('Fetched memories:', response.data);
        } catch (err) {
            console.error("Error searching memories:", err);
            const errorMsg = axios.isAxiosError(err) && err.response 
                            ? `Search Error: ${err.response.data.detail || err.message}`
                            : "An unknown error occurred during search.";
            setSearchError(errorMsg);
            setMemories([]);
        } finally {
            setLoadingSearch(false);
        }
    }, [agentId]);

    useEffect(() => {
        fetchMemories(debouncedSearchQuery);
    }, [debouncedSearchQuery, fetchMemories]);

    // --- Manual Add --- 
    const handleManualAdd = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!manualContent.trim()) {
            setManualAddError("Memory content cannot be empty.");
            return;
        }
        setLoadingAction(true);
        setManualAddError(null);
        setDeleteError(null); // Clear other errors
        console.log(`Adding manual memory for agent ${agentId}`);

        try {
            const payload = {
                memory_type: manualContentType,
                content: manualContent,
                importance: manualImportance,
            };
            const response = await axios.post<AgentMemory>(
                `/api/agents/${agentId}/memory`,
                payload
            );
            console.log('Manual memory added:', response.data);
            // Optionally add to list immediately or refetch?
            // Let's clear form and maybe refetch search if query exists?
            setManualContent('');
            setManualContentType('manual_fact');
            setManualImportance(0.7);
            // Refetch if there was a search query active?
            if (searchQuery) {
                 fetchMemories(searchQuery); // Re-run search to potentially see new memory
            }
             // Could also add it to the top of the list optimistically:
             // setMemories(prev => [response.data, ...prev]);
        } catch (err) {
             console.error("Error adding manual memory:", err);
             const errorMsg = axios.isAxiosError(err) && err.response 
                            ? `Failed to add memory: ${err.response.data.detail || err.message}`
                            : "An unknown error occurred.";
             setManualAddError(errorMsg);
        } finally {
            setLoadingAction(false);
        }
    };

    // --- Delete --- 
    const handleDeleteMemory = async (memoryId: number) => {
        if (loadingAction) return; // Prevent multiple deletes
        if (!window.confirm(`Are you sure you want to delete memory ID ${memoryId}?`)) return;
        
        setLoadingAction(true);
        setDeleteError(null);
        setManualAddError(null);
        console.log(`Deleting memory ${memoryId} for agent ${agentId}`);

        try {
            await axios.delete(`/api/agents/${agentId}/memory/${memoryId}`);
            console.log(`Memory ${memoryId} deleted.`);
            // Remove from list optimistically
            setMemories(prev => prev.filter(mem => mem.id !== memoryId));
        } catch (err) {
            console.error("Error deleting memory:", err);
             const errorMsg = axios.isAxiosError(err) && err.response 
                            ? `Failed to delete memory: ${err.response.data.detail || err.message}`
                            : "An unknown error occurred.";
             setDeleteError(errorMsg);
        } finally {
            setLoadingAction(false);
        }
    };

    return (
        <div className={styles.memoryExplorerContainer}>
            <h3>Agent Memory Explorer (Agent ID: {agentId})</h3>

            {/* Manual Add Section */} 
            <div className={styles.manualAddSection}>
                <h4>Add Manual Memory</h4>
                <form onSubmit={handleManualAdd} className={styles.manualAddForm}>
                    <div className={styles.formRow}>
                        <div className={styles.formGroup}>
                            <label htmlFor="manualContentType">Type:</label>
                            <input 
                                type="text"
                                id="manualContentType"
                                value={manualContentType}
                                onChange={(e) => setManualContentType(e.target.value)}
                                required
                            />
                        </div>
                        <div className={styles.formGroup}>
                            <label htmlFor="manualImportance">Importance (0-1):</label>
                            <input 
                                type="number"
                                id="manualImportance"
                                value={manualImportance}
                                onChange={(e) => setManualImportance(parseFloat(e.target.value))}
                                min="0" max="1" step="0.1"
                                required
                            />
                        </div>
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="manualContent">Content:</label>
                        <textarea 
                            id="manualContent"
                            rows={3}
                            value={manualContent}
                            onChange={(e) => setManualContent(e.target.value)}
                            required
                        />
                    </div>
                    {manualAddError && <p className={styles.error}>{manualAddError}</p>}
                    <button type="submit" disabled={loadingAction} className={styles.addButton}>
                        {loadingAction ? 'Adding...' : 'Add Memory'}
                    </button>
                </form>
            </div>

            <hr className={styles.divider} />

            {/* Search Section */} 
            <div className={styles.searchSection}>
                <h4>Search Memories</h4>
                 <div className={styles.formGroup}>
                    <label htmlFor="searchQuery">Search Query:</label>
                    <input 
                        type="text"
                        id="searchQuery"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Enter text for similarity search..."
                        className={styles.searchInput}
                    />
                </div>
                {loadingSearch && <p className={styles.loading}>Searching...</p>}
                {searchError && <p className={styles.error}>{searchError}</p>}
            </div>

            {/* Results Section */} 
            <div className={styles.resultsSection}>
                <h4>Search Results (Top 10 Relevant):</h4>
                {deleteError && <p className={styles.error}>{deleteError}</p>}
                {memories.length === 0 && !loadingSearch && searchQuery && (
                    <p>No relevant memories found for "{searchQuery}".</p>
                )}
                 {memories.length === 0 && !loadingSearch && !searchQuery && (
                    <p>Enter a search query above to find relevant memories.</p>
                )}
                <ul className={styles.memoryList}>
                    {memories.map(memory => (
                        <li key={memory.id} className={styles.memoryItem}>
                            <div className={styles.memoryHeader}>
                                <span className={styles.memoryType}>{memory.memory_type}</span>
                                <span className={styles.memoryTimestamp}>{new Date(memory.timestamp).toLocaleString()}</span>
                                <span className={styles.memoryImportance}>Importance: {memory.importance.toFixed(2)}</span>
                            </div>
                            <p className={styles.memoryContent}>{memory.content}</p>
                            <button 
                                onClick={() => handleDeleteMemory(memory.id)}
                                disabled={loadingAction}
                                className={styles.deleteButton}
                            >
                                Delete
                            </button>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default MemoryExplorer; 