import React, { useState } from 'react';
import axios, { AxiosError } from 'axios'; // Assuming axios is used for API calls
import styles from './Login.module.css'; // We'll create this CSS module

// Define the expected props for the Login component
interface LoginProps {
    onLoginSuccess: (token: string) => void; // Callback function when login is successful
}

// Define the expected shape of the login API response
interface LoginResponse {
    access_token: string;
    token_type: string; // Usually 'bearer'
}

const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError(null);
        setLoading(true);

        try {
            // Use URLSearchParams for application/x-www-form-urlencoded
            const params = new URLSearchParams();
            params.append('username', username);
            params.append('password', password);
            
            // Make the API call with URLSearchParams and correct header
            const response = await axios.post<LoginResponse>('/api/v1/auth/login', params, {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded' 
                }
            });

            // Now TypeScript knows about response.data.access_token
            if (response.data && response.data.access_token) {
                onLoginSuccess(response.data.access_token);
            } else {
                setError('Login failed: No token received.');
            }
        } catch (err) { // Use unknown type for better type safety
            console.error('Login error:', err);
            // Use the imported AxiosError type and the static axios.isAxiosError method
            if (axios.isAxiosError(err)) {
                // err is now confirmed to be an AxiosError
                const axiosError = err as AxiosError<any>; // Cast to access response data, <any> for simplicity if backend error shape is unknown/varied
                setError(axiosError.response?.data?.detail || `Login failed: ${axiosError.response?.statusText || axiosError.message}`);
            } else if (err instanceof Error) {
                 setError(`Login failed: ${err.message}`);
            } else {
                setError('Login failed: An unexpected error occurred.');
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.loginContainer}>
            <div className={styles.loginBox}>
                <h1 className={styles.title}>VibeRAG Login</h1>
                <form onSubmit={handleSubmit} className={styles.loginForm}>
                    <div className={styles.inputGroup}>
                        <label htmlFor="username">Username</label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            disabled={loading}
                        />
                    </div>
                    <div className={styles.inputGroup}>
                        <label htmlFor="password">Password</label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            disabled={loading}
                        />
                    </div>
                    {error && <p className={styles.errorMessage}>{error}</p>}
                    <button type="submit" className={styles.loginButton} disabled={loading}>
                        {loading ? 'Logging in...' : 'Login'}
                    </button>
                </form>
                 <p className={styles.defaultCredsInfo}>
                    Default admin: <code>admin</code> / <code>admin</code> (Change after login!)
                </p>
            </div>
        </div>
    );
};

export default Login; 