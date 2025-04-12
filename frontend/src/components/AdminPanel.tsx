import React, { useState, useEffect, useCallback, FormEvent } from 'react';
import axios, { AxiosError } from 'axios';
import styles from './AdminPanel.module.css'; // Create this CSS module later
// Import Tremor Dialog for edit modal
import { Dialog, DialogPanel, Button, TextInput, Switch, Textarea } from '@tremor/react';
// Import Log Explorer
import LogExplorer from './LogExplorer';
// Import Memory Explorer
import MemoryExplorer from './MemoryExplorer';

// Define the User type based on backend UserResponse
interface User {
    id: number;
    username: string;
    role: string;
    is_active: boolean;
    is_admin: boolean;
    created_at: string; // Assuming string representation
}

// Backend UserUpdate type (subset of fields)
interface UserUpdatePayload {
    role?: string;
    is_active?: boolean;
    password?: string;
}

// Backend UserCreate type
interface UserCreatePayload {
    username: string;
    password: string;
    role?: string; // Optional, backend defaults to 'user'
}

// --- Agent Types (based on backend AgentResponse/AgentCreate/AgentUpdate) ---
interface Agent {
    id: number;
    owner_user_id: number;
    name: string;
    persona: string | null;
    goals: string | null;
    base_prompt: string | null;
    created_at: string;
    is_active: boolean;
}
interface AgentCreatePayload {
    name: string;
    persona?: string;
    goals?: string;
    base_prompt?: string;
    is_active?: boolean;
    // owner_user_id will be set implicitly by backend based on logged-in user
}
interface AgentUpdatePayload {
    name?: string;
    persona?: string;
    goals?: string;
    base_prompt?: string;
    is_active?: boolean;
}

// --- Add Props Interface ---
interface AdminPanelProps {
    currentUserId: number; // ID of the logged-in admin user
}

function AdminPanel({ currentUserId }: AdminPanelProps) {
    // --- User Management State ---
    const [users, setUsers] = useState<User[]>([]);
    const [userIsLoading, setUserIsLoading] = useState<boolean>(true);
    const [userError, setUserError] = useState<string | null>(null);
    const [userSuccessMessage, setUserSuccessMessage] = useState<string | null>(null);
    
    // --- State for Forms ---
    const [newUsername, setNewUsername] = useState<string>('');
    const [newUserPassword, setNewUserPassword] = useState<string>('');
    const [newUserIsAdmin, setNewUserIsAdmin] = useState<boolean>(false);
    const [changePassword, setChangePassword] = useState<string>('');

    // --- Agent Management State ---
    const [agents, setAgents] = useState<Agent[]>([]);
    const [agentIsLoading, setAgentIsLoading] = useState<boolean>(true);
    const [agentError, setAgentError] = useState<string | null>(null);
    const [agentSuccessMessage, setAgentSuccessMessage] = useState<string | null>(null);
    // State for Agent Create/Edit Form (could use a modal later)
    const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
    const [agentFormData, setAgentFormData] = useState<AgentCreatePayload | AgentUpdatePayload>({ name: '' });
    const [showAgentForm, setShowAgentForm] = useState<boolean>(false);
    const [showAgentModal, setShowAgentModal] = useState<boolean>(false);
    const [agentFormError, setAgentFormError] = useState<string | null>(null);

    // --- Combined Loading/Error State ---
    const [isLoading, setIsLoading] = useState<boolean>(true); // Use a combined loading state
    const [error, setError] = useState<string | null>(null); // Use a combined error state
    const [successMessage, setSuccessMessage] = useState<string | null>(null); // Use a combined success state

    // --- New State Variable ---
    const [selectedAgentIdForMemory, setSelectedAgentIdForMemory] = useState<number | null>(null);

    // --- Utility to handle API errors (generic) ---
    const handleApiError = (err: unknown, defaultMessage: string) => {
        let message = defaultMessage;
        if (axios.isAxiosError(err)) {
            const axiosError = err as AxiosError<{ detail?: string }>;
            message = axiosError.response?.data?.detail || defaultMessage;
        }
        setError(message);
        setSuccessMessage(null); // Clear success on new error
        console.error(message, err);
    };

    const fetchUsers = useCallback(async () => {
        setUserIsLoading(true);
        try {
            const response = await axios.get<User[]>('/api/v1/users');
            setUsers(response.data);
            setUserError(null); // Clear user-specific error
        } catch (err) {
            handleApiError(err, 'Failed to load users.');
            setUserError('Failed to load users.'); // Set specific error too
        } finally {
            setUserIsLoading(false);
        }
    }, []);

    const fetchAgents = useCallback(async () => {
        setAgentIsLoading(true);
        try {
            // Use the new admin endpoint
            const response = await axios.get<Agent[]>('/api/v1/agents/all'); // Assuming /api/v1 prefix
            setAgents(response.data);
            setAgentError(null);
        } catch (err) {
            handleApiError(err, 'Failed to load agents.');
            setAgentError('Failed to load agents.');
        } finally {
            setAgentIsLoading(false);
        }
    }, []);

    // Fetch all data on mount
    useEffect(() => {
        const fetchAllData = async () => {
            setIsLoading(true);
            setError(null);
            setSuccessMessage(null);
            await Promise.all([fetchUsers(), fetchAgents()]);
            setIsLoading(false);
        };
        fetchAllData();
    }, [fetchUsers, fetchAgents]);

    // --- Handler Functions ---

    const handleToggleActive = async (user: User) => {
        setError(null); setSuccessMessage(null); // Clear global messages
        const newActiveState = !user.is_active;
        const payload: UserUpdatePayload = { is_active: newActiveState };
        try {
            await axios.put(`/api/v1/users/${user.id}`, payload);
            setSuccessMessage(`User ${user.username} ${newActiveState ? 'activated' : 'deactivated'}.`);
            fetchUsers(); // Refresh list
        } catch (err) {
            console.error("Failed to toggle user active state:", err);
            handleApiError(err, `Failed to update user ${user.username}.`);
        }
    };

    const handleResetPassword = async (userId: number, username: string) => {
        setError(null); setSuccessMessage(null); // Clear global messages
        const newPassword = prompt(`Enter new password for user ${username}:`);
        if (!newPassword || newPassword.trim() === '') {
            handleApiError(null, 'Password reset cancelled or password empty.');
            return;
        }
        const payload: UserUpdatePayload = { password: newPassword };
        try {
            await axios.put(`/api/v1/users/${userId}`, payload);
            setSuccessMessage(`Password reset successfully for user ${username}.`);
            // No need to refresh user list for password change
        } catch (err) {
            console.error("Failed to reset password:", err);
            handleApiError(err, `Failed to reset password for ${username}.`);
        }
    };
    
    const handleDeleteUser = async (userId: number, username: string) => {
        setError(null); setSuccessMessage(null); // Clear global messages
        if (!window.confirm(`Are you sure you want to delete user ${username}? This action cannot be undone.`)) {
            return;
        }
        try {
            await axios.delete(`/api/v1/users/${userId}`);
            setSuccessMessage(`User ${username} deleted successfully.`);
            fetchUsers(); // Refresh list
        } catch (err) {
            console.error("Failed to delete user:", err);
            handleApiError(err, `Failed to delete user ${username}.`);
        }
    };

    const handleCreateUser = async (event: FormEvent) => {
        setError(null); setSuccessMessage(null); // Clear global messages
        event.preventDefault();
        if (!newUsername.trim() || !newUserPassword.trim()) {
            handleApiError(null, 'Username and password are required.');
            return;
        }
        const payload: UserCreatePayload = {
            username: newUsername,
            password: newUserPassword,
            role: newUserIsAdmin ? 'admin' : 'user',
        };
        try {
            await axios.post('/api/v1/users', payload);
            setSuccessMessage(`User ${newUsername} created successfully.`);
            // Clear form and refresh list
            setNewUsername('');
            setNewUserPassword('');
            setNewUserIsAdmin(false);
            fetchUsers();
        } catch (err: unknown) {
            console.error("Failed to create user:", err);
            let errorMsg = 'Failed to create user.';
            if (axios.isAxiosError(err) && err.response?.data?.detail) {
                 errorMsg = `Failed to create user: ${err.response.data.detail}`;
            } else if (err instanceof Error) {
                 errorMsg = `Failed to create user: ${err.message}`;
            }
            handleApiError(null, errorMsg);
        }
    };

    const handleChangeOwnPassword = async (event: FormEvent) => {
        setError(null); setSuccessMessage(null); // Clear global messages
        event.preventDefault();
        if (!changePassword.trim()) {
            handleApiError(null, 'New password cannot be empty.');
            return;
        }
        const payload: UserUpdatePayload = { password: changePassword };
        try {
            await axios.put(`/api/v1/users/${currentUserId}`, payload);
            setSuccessMessage('Your password has been changed successfully.');
            setChangePassword(''); // Clear field
        } catch (err) {
            console.error("Failed to change password:", err);
            handleApiError(err, 'Failed to change your password.');
        }
    };

    // --- Agent Handler Functions ---
    
    // Open Edit Modal
    const handleEditAgent = (agent: Agent) => {
        setEditingAgent(agent); // Store the agent being edited
        // Pre-populate form data from the agent
        setAgentFormData({
            name: agent.name,
            persona: agent.persona ?? '',
            goals: agent.goals ?? '',
            base_prompt: agent.base_prompt ?? '',
            is_active: agent.is_active,
        });
        setAgentFormError(null); // Clear previous form errors
        setShowAgentModal(true); // Open the modal
    };

    // Close Modal / Cancel Edit
    const handleCloseAgentModal = () => {
        setShowAgentModal(false);
        setEditingAgent(null); // Clear editing state
        setAgentFormData({ name: '' }); // Reset form data
        setAgentFormError(null);
    };

    // Handle input changes in the agent form
    const handleAgentFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value, type } = e.target;
        
        // Special handling for Tremor Switch (or standard checkbox)
        if (type === 'checkbox') {
            const checked = (e.target as HTMLInputElement).checked;
             setAgentFormData(prev => ({
                 ...prev,
                 [name]: checked,
             }));
        } else {
            // Handle text inputs and textareas
            setAgentFormData(prev => ({
                ...prev,
                [name]: value,
            }));
        }
    };
    
    // Separate handler specifically for Tremor Switch boolean value
    const handleAgentSwitchChange = (value: boolean) => {
         setAgentFormData(prev => ({
             ...prev,
             is_active: value,
         }));
    };

    // Handle Form Submission (Create/Update)
    const handleAgentFormSubmit = async (event: FormEvent) => {
        event.preventDefault();
        setError(null); // Clear main panel error
        setAgentFormError(null);
        setSuccessMessage(null);

        if (!agentFormData.name?.trim()) {
            setAgentFormError('Agent name is required.');
            return;
        }

        const payload = { ...agentFormData };
        const apiUrl = editingAgent 
            ? `/api/v1/agents/admin/${editingAgent.id}` // Use admin update endpoint
            : '/api/v1/agents/'; // Regular create endpoint (admins can create for themselves? Or adjust needed?)
            // NOTE: Current create API sets owner_user_id to current_user. Admins might need a way to set owner.
            // For now, admin creates agents for themselves.
        const method = editingAgent ? 'put' : 'post';

        try {
            await axios({ method, url: apiUrl, data: payload });
            setSuccessMessage(`Agent ${editingAgent ? 'updated' : 'created'} successfully.`);
            handleCloseAgentModal(); // Close modal on success
            fetchAgents(); // Refresh agent list
        } catch (err) {
            let message = `Failed to ${editingAgent ? 'update' : 'create'} agent.`;
            if (axios.isAxiosError(err)) {
                message = (err as AxiosError<{ detail?: string }>).response?.data?.detail || message;
            }
            setAgentFormError(message);
            console.error("Agent form error:", err);
        }
    };

    // Handle Delete Agent
    const handleDeleteAgent = async (agentId: number, agentName: string) => {
        setError(null); setSuccessMessage(null);
        if (!window.confirm(`Are you sure you want to delete agent ${agentName} (ID: ${agentId})? This action cannot be undone.`)) {
            return;
        }
        try {
            await axios.delete(`/api/v1/agents/admin/${agentId}`); // Use admin delete endpoint
            setSuccessMessage(`Agent ${agentName} deleted successfully.`);
            fetchAgents(); // Refresh agent list
        } catch (err) {
            console.error("Failed to delete agent:", err);
            handleApiError(err, `Failed to delete agent ${agentName}.`);
        }
    };

    // Handler to show memory explorer for a specific agent
    const handleViewMemory = (agentId: number) => {
        setSelectedAgentIdForMemory(agentId);
    };

    return (
        <div className={styles.adminPanelContainer}>
            <h2>Admin User Management</h2>

            {/* Display Loading/Error/Success Messages */}
            {isLoading && <p>Loading...</p>}
            {error && <p className={styles.errorText}>{error}</p>}
            {successMessage && <p className={styles.successText}>{successMessage}</p>}

            {/* Section 1: User List */}
            {!isLoading && users.length === 0 && !error && <p>No users found.</p>}
            {!isLoading && users.length > 0 && (
                <table className={styles.userTable}> {/* Add class for styling */}
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Username</th>
                            <th>Role</th>
                            <th>Active</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {users.map(user => (
                            <tr key={user.id}>
                                <td>{user.id}</td>
                                <td>{user.username}</td>
                                <td>{user.role}</td>
                                <td>{user.is_active ? 'Yes' : 'No'}</td>
                                <td className={styles.actionsCell}>
                                    <button 
                                        onClick={() => handleToggleActive(user)}
                                        className={styles.actionButton}
                                        disabled={user.id === currentUserId} // Prevent admin deactivating self
                                        title={user.id === currentUserId ? "Cannot change own status" : (user.is_active ? "Deactivate" : "Activate")} >
                                        {user.is_active ? 'Deactivate' : 'Activate'}
                                    </button>
                                    <button 
                                        onClick={() => handleResetPassword(user.id, user.username)}
                                        className={styles.actionButton}
                                        disabled={user.id === currentUserId} // Prevent admin resetting password
                                        title={user.id === currentUserId ? "Cannot reset password" : "Reset Password"} >
                                        Reset PW
                                    </button>
                                    <button 
                                        onClick={() => handleDeleteUser(user.id, user.username)}
                                        className={`${styles.actionButton} ${styles.deleteButton}`}
                                        disabled={user.id === currentUserId} // Prevent admin deleting self
                                        title={user.id === currentUserId ? "Cannot delete self" : "Delete User"} >
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
             )}

            {/* Section 2: Create New User */}
            <section className={styles.createUserSection}>
                <h3>Create New User</h3>
                <form onSubmit={handleCreateUser} className={styles.form}>
                    <div className={styles.formGroup}>
                        <label htmlFor="newUsername">Username:</label>
                        <input 
                            id="newUsername"
                            type="text" 
                            value={newUsername} 
                            onChange={(e) => setNewUsername(e.target.value)} 
                            required 
                        />
                    </div>
                    <div className={styles.formGroup}>
                        <label htmlFor="newUserPassword">Password:</label>
                        <input 
                            id="newUserPassword"
                            type="password" 
                            value={newUserPassword} 
                            onChange={(e) => setNewUserPassword(e.target.value)} 
                            required 
                        />
                    </div>
                    <div className={styles.formGroupInline}> 
                        <input 
                            id="newUserIsAdmin"
                            type="checkbox" 
                            checked={newUserIsAdmin} 
                            onChange={(e) => setNewUserIsAdmin(e.target.checked)} 
                        />
                        <label htmlFor="newUserIsAdmin">Assign Admin Role?</label>
                    </div>
                    <button type="submit" className={styles.submitButton}>Create User</button>
                </form>
            </section>

            {/* Section 3: Change Own Password */}
            <section className={styles.changePasswordSection}>
                <h3>Change Your Password</h3>
                 <form onSubmit={handleChangeOwnPassword} className={styles.form}>
                     <div className={styles.formGroup}>
                        <label htmlFor="changePassword">New Password:</label>
                        <input 
                            id="changePassword"
                            type="password" 
                            value={changePassword} 
                            onChange={(e) => setChangePassword(e.target.value)} 
                            required 
                        />
                     </div>
                    <button type="submit" className={styles.submitButton}>Update My Password</button>
                 </form>
            </section>

            {/* Section 4: Agent List */}
            <section className={styles.agentSection}>
                <h2>Agent Management</h2>
                {/* Display Agent Loading/Error Messages */}
                {agentIsLoading && <p>Loading agents...</p>}
                {agentError && <p className={styles.errorText}>{agentError}</p>}
                {/* Agent List Table */}
                {!agentIsLoading && agents.length > 0 && (
                    <table className={styles.agentTable}>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Name</th>
                                <th>Owner User ID</th>
                                <th>Active</th>
                                <th>Created</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {agents.map(agent => (
                                <tr key={agent.id}>
                                    <td>{agent.id}</td>
                                    <td>{agent.name}</td>
                                    <td>{agent.owner_user_id}</td>
                                    <td>{agent.is_active ? 'Yes' : 'No'}</td>
                                    <td>{new Date(agent.created_at).toLocaleString()}</td>
                                    <td className={styles.actionsCell}>
                                        <button onClick={() => handleEditAgent(agent)} className={styles.actionButton}>Edit</button>
                                        <button onClick={() => handleDeleteAgent(agent.id, agent.name)} className={`${styles.actionButton} ${styles.deleteButton}`}>Delete</button>
                                        <button onClick={() => handleViewMemory(agent.id)} className={styles.actionButton}>View Memory</button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
                {!agentIsLoading && agents.length === 0 && !agentError && <p>No agents found.</p>}
            </section>

            {/* Agent Edit/Create Modal */} 
            <Dialog open={showAgentModal} onClose={handleCloseAgentModal} static={true}>
                <DialogPanel>
                    <h3 className="text-lg font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                        {editingAgent ? `Edit Agent: ${editingAgent.name}` : 'Create New Agent'}
                    </h3>
                    <form onSubmit={handleAgentFormSubmit} className={styles.modalForm}>
                        {agentFormError && <p className={styles.errorText}>{agentFormError}</p>}
                        
                        <div className={styles.formRow}>
                            <label htmlFor="agentName">Name*:</label>
                            <TextInput id="agentName" name="name" value={agentFormData.name || ''} onChange={handleAgentFormChange} required />
                        </div>
                        <div className={styles.formRow}>
                            <label htmlFor="agentPersona">Persona:</label>
                            <Textarea id="agentPersona" name="persona" value={agentFormData.persona || ''} onChange={handleAgentFormChange} rows={3} />
                        </div>
                        <div className={styles.formRow}>
                            <label htmlFor="agentGoals">Goals:</label>
                            <Textarea id="agentGoals" name="goals" value={agentFormData.goals || ''} onChange={handleAgentFormChange} rows={3} />
                        </div>
                        <div className={styles.formRow}>
                            <label htmlFor="agentBasePrompt">Base Prompt:</label>
                            <Textarea id="agentBasePrompt" name="base_prompt" value={agentFormData.base_prompt || ''} onChange={handleAgentFormChange} rows={5} />
                        </div>
                        <div className={styles.formRowInline}>
                             <label htmlFor="agentIsActive">Active:</label>
                             {/* Use the separate handler for Tremor Switch */}
                             <Switch id="agentIsActive" name="is_active" checked={agentFormData.is_active ?? true} onChange={handleAgentSwitchChange} />
                        </div>
                        
                        <div className={styles.modalActions}>
                            <Button type="submit" variant="primary">{editingAgent ? 'Save Changes' : 'Create Agent'}</Button>
                            <Button type="button" variant="secondary" onClick={handleCloseAgentModal}>Cancel</Button>
                        </div>
                    </form>
                </DialogPanel>
            </Dialog>

            {/* --- NEW Section: Log Explorer (Admin View) --- */}
            <section className={styles.sectionBox}>
                <h2>Log Explorer (Admin View)</h2>
                <p>View logs across all agents and users. Use filters below.</p>
                <LogExplorer 
                    // Pass no agentId or userId initially, allowing admin to filter within the component
                />
            </section>
            
            {/* --- NEW Section: Memory Explorer (Conditional) --- */}
            {selectedAgentIdForMemory && (
                 <section className={styles.sectionBox}>
                    <h2>Memory Explorer (Agent ID: {selectedAgentIdForMemory})</h2>
                    <Button variant="secondary" onClick={() => setSelectedAgentIdForMemory(null)} className={styles.closeExplorerButton}>Close Memory Explorer</Button>
                    <MemoryExplorer 
                        agentId={selectedAgentIdForMemory} 
                        userId={currentUserId} // Pass admin user ID 
                    />
                </section>
            )}

        </div>
    );
}

export default AdminPanel; 