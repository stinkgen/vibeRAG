import React, { useState, useEffect, useCallback, FormEvent } from 'react';
import axios, { AxiosError } from 'axios';
import styles from './AdminPanel.module.css'; // Create this CSS module later

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

// --- Add Props Interface ---
interface AdminPanelProps {
    currentUserId: number; // ID of the logged-in admin user
}

function AdminPanel({ currentUserId }: AdminPanelProps) {
    const [users, setUsers] = useState<User[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    
    // --- State for Forms ---
    const [newUsername, setNewUsername] = useState<string>('');
    const [newUserPassword, setNewUserPassword] = useState<string>('');
    const [newUserIsAdmin, setNewUserIsAdmin] = useState<boolean>(false);
    const [changePassword, setChangePassword] = useState<string>('');

    // --- Utility to handle API errors ---
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
        setIsLoading(true);
        // Don't clear previous error immediately, let success clear it
        // setError(null); 
        try {
            const response = await axios.get<User[]>('/api/v1/users');
            setUsers(response.data);
            setError(null); // Clear error on successful fetch
        } catch (err) {
            console.error("Failed to fetch users:", err);
            handleApiError(err, 'Failed to load users. You might not have admin privileges.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchUsers();
    }, [fetchUsers]);

    // --- Handler Functions ---

    const handleToggleActive = async (user: User) => {
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
        const newPassword = prompt(`Enter new password for user ${username}:`);
        if (!newPassword || newPassword.trim() === '') {
            handleApiError(null, 'Password reset cancelled or password empty.');
            return;
        }
        const payload: UserUpdatePayload = { password: newPassword };
        try {
            await axios.put(`/api/v1/users/${userId}`, payload);
            handleApiError(null, `Password reset successfully for user ${username}.`);
            // No need to refresh user list for password change
        } catch (err) {
            console.error("Failed to reset password:", err);
            handleApiError(err, `Failed to reset password for ${username}.`);
        }
    };
    
    const handleDeleteUser = async (userId: number, username: string) => {
        if (!window.confirm(`Are you sure you want to delete user ${username}? This action cannot be undone.`)) {
            return;
        }
        try {
            await axios.delete(`/api/v1/users/${userId}`);
            handleApiError(null, `User ${username} deleted successfully.`);
            fetchUsers(); // Refresh list
        } catch (err) {
            console.error("Failed to delete user:", err);
            handleApiError(err, `Failed to delete user ${username}.`);
        }
    };

    const handleCreateUser = async (event: FormEvent) => {
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
            handleApiError(null, `User ${newUsername} created successfully.`);
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
        event.preventDefault();
        if (!changePassword.trim()) {
            handleApiError(null, 'New password cannot be empty.');
            return;
        }
        const payload: UserUpdatePayload = { password: changePassword };
        try {
            await axios.put(`/api/v1/users/${currentUserId}`, payload);
            handleApiError(null, 'Your password has been changed successfully.');
            setChangePassword(''); // Clear field
        } catch (err) {
            console.error("Failed to change password:", err);
            handleApiError(err, 'Failed to change your password.');
        }
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
        </div>
    );
}

export default AdminPanel; 