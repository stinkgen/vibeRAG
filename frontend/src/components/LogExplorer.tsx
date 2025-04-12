import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import styles from './LogExplorer.module.css';

// Define AgentLog structure from API (matches AgentLogResponse)
interface AgentLog {
    id: number;
    agent_id: number;
    level: string;
    message: string;
    details?: string | null; // Optional JSON string
    timestamp: string; // ISO string format
}

interface LogExplorerProps {
    // Allow filtering by specific agent or user (for admin view)
    agentId?: number | null;
    userId?: number | null; 
    // Could add more props for default time range, etc.
}

// Available log levels for filtering
const LOG_LEVELS = ['PLAN', 'ACTION', 'OBSERVATION', 'INFO', 'DEBUG', 'WARN', 'ERROR', 'CRITICAL'];

const LogExplorer: React.FC<LogExplorerProps> = ({ agentId, userId }) => {
    const [logs, setLogs] = useState<AgentLog[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    
    // Filter state
    const [filterAgentId, setFilterAgentId] = useState<string>(agentId?.toString() || '');
    const [filterUserId, setFilterUserId] = useState<string>(userId?.toString() || '');
    const [filterLevels, setFilterLevels] = useState<string[]>([]);
    // Add startTime, endTime state later if needed
    const [limit, setLimit] = useState<number>(100);
    const [skip, setSkip] = useState<number>(0); // For pagination later

    // useCallback to memoize the fetch function
    const fetchLogs = useCallback(async () => {
        setLoading(true);
        setError(null);
        console.log('Fetching agent logs with filters:', { 
            agent_id: filterAgentId || undefined,
            user_id: filterUserId || undefined, // Pass user_id if needed (for admin)
            levels: filterLevels,
            limit,
            skip
        });

        try {
            const params = new URLSearchParams();
            if (filterAgentId) params.append('agent_id', filterAgentId);
            if (filterUserId) params.append('user_id', filterUserId); // Only relevant for admin view
            filterLevels.forEach(level => params.append('levels', level));
            params.append('limit', limit.toString());
            params.append('skip', skip.toString());
            
            // Assuming endpoint is /api/v1/agents/logs/ now
            const response = await axios.get<AgentLog[]>('/api/v1/agents/logs/', { params });
            setLogs(response.data);
            console.log('Fetched logs:', response.data);
        } catch (err) {
            console.error("Error fetching agent logs:", err);
             const errorMsg = axios.isAxiosError(err) && err.response 
                            ? `Failed to fetch logs: ${err.response.data.detail || err.message}`
                            : "An unknown error occurred while fetching logs.";
            setError(errorMsg);
            setLogs([]); // Clear logs on error
        } finally {
            setLoading(false);
        }
    }, [filterAgentId, filterUserId, filterLevels, limit, skip]);

    // Fetch logs on initial mount and when filters change
    useEffect(() => {
        fetchLogs();
    }, [fetchLogs]);

    // Update internal filter state if props change (e.g., when used in AgentDetailView)
    useEffect(() => {
        setFilterAgentId(agentId?.toString() || '');
    }, [agentId]);
    useEffect(() => {
        setFilterUserId(userId?.toString() || '');
    }, [userId]);

    const handleLevelChange = (level: string, isChecked: boolean) => {
        setFilterLevels(prev => 
            isChecked ? [...prev, level] : prev.filter(l => l !== level)
        );
        // Fetch logs automatically on filter change? Or require button click?
        // Let's fetch automatically for now.
    };
    
    const renderLogDetails = (details: string | null | undefined): React.ReactNode => {
        if (!details) return 'N/A';
        try {
            const parsed = JSON.parse(details);
            // Pretty print JSON
            return <pre>{JSON.stringify(parsed, null, 2)}</pre>;
        } catch (e) {
            // If not JSON, display as plain text
            return <span>{details}</span>;
        }
    };

    return (
        <div className={styles.logExplorerContainer}>
            <h3>Agent Log Explorer</h3>
            
            {/* Filter Controls */} 
            <div className={styles.filters}>
                {/* Allow changing agent/user ID only if not passed as prop (e.g., admin view) */} 
                {!agentId && (
                    <div className={styles.filterGroup}>
                        <label htmlFor="filterAgentId">Agent ID:</label>
                        <input 
                            type="number" 
                            id="filterAgentId" 
                            value={filterAgentId}
                            onChange={(e) => setFilterAgentId(e.target.value)}
                            placeholder="Filter by Agent ID"
                        />
                    </div>
                )}
                 {!userId && (
                    <div className={styles.filterGroup}>
                        <label htmlFor="filterUserId">User ID:</label>
                        <input 
                            type="number" 
                            id="filterUserId" 
                            value={filterUserId}
                            onChange={(e) => setFilterUserId(e.target.value)}
                             placeholder="Filter by User ID (Admin)"
                        />
                    </div>
                 )}
                 <div className={styles.filterGroupLevels}>
                     <label>Log Levels:</label>
                     <div className={styles.levelCheckboxes}>
                         {LOG_LEVELS.map(level => (
                             <div key={level} className={styles.checkboxItem}>
                                 <input 
                                     type="checkbox" 
                                     id={`level-${level}`}
                                     checked={filterLevels.includes(level)}
                                     onChange={(e) => handleLevelChange(level, e.target.checked)}
                                 />
                                 <label htmlFor={`level-${level}`}>{level}</label>
                             </div>
                         ))}
                     </div>
                 </div>
                 {/* Add Date Pickers later if needed */} 
                 <button onClick={fetchLogs} disabled={loading} className={styles.refreshButton}>
                     {loading ? 'Refreshing...' : 'Refresh Logs'}
                 </button>
            </div>

            {/* Log Display */} 
            {loading && <div className={styles.loading}>Loading logs...</div>}
            {error && <div className={styles.error}>{error}</div>}
            {!loading && !error && (
                <table className={styles.logTable}>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Agent ID</th>
                            <th>Level</th>
                            <th>Message</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {logs.length === 0 ? (
                            <tr><td colSpan={5}>No logs found matching criteria.</td></tr>
                        ) : (
                            logs.map(log => (
                                <tr key={log.id}>
                                    <td>{new Date(log.timestamp).toLocaleString()}</td>
                                    <td>{log.agent_id}</td>
                                    <td><span className={`${styles.logLevel} ${styles[`logLevel${log.level}`]}`}>{log.level}</span></td>
                                    <td>{log.message}</td>
                                    <td>{renderLogDetails(log.details)}</td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            )}
            {/* TODO: Add Pagination controls */}
        </div>
    );
};

export default LogExplorer; 