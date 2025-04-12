import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Import axios
import Chat from './components/Chat';
import DocumentManager from './components/DocumentManager';
import PresentationViewer from './components/PresentationViewer';
import ResearchReport from './components/ResearchReport';
import Config from './components/Config';
import Login from './components/Login'; // Import Login component
import AdminPanel from './components/AdminPanel'; // Import AdminPanel
import styles from './App.module.css';
import './styles/global.css';

// --- Decode Token Helper (Simplified) ---
interface DecodedToken {
    sub?: string; // username
    id?: number;
    role?: string;
    exp?: number;
}

function decodeJwt(token: string): DecodedToken | null {
    try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(
            atob(base64)
                .split('')
                .map(function (c) {
                    return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
                })
                .join('')
        );
        return JSON.parse(jsonPayload);
    } catch (error) {
        console.error("Failed to decode JWT:", error);
        return null;
    }
}

// --- Axios Interceptor Setup ---
let interceptorInitialized = false;

function setupAxiosInterceptor() {
    // Ensure it's only set up once
    if (interceptorInitialized) {
        // console.log("Axios interceptor already initialized.");
        return;
    }
    
    axios.interceptors.request.use(
        (config) => {
            // Read token directly from localStorage for every request
            const token = localStorage.getItem('vibeRAG_authToken');
            if (token) {
                config.headers.Authorization = `Bearer ${token}`;
                // console.log('Interceptor adding token:', token.substring(0, 10) + '...');
            } else {
                 // Optionally remove header if no token? Might cause issues if server expects it.
                 // delete config.headers.Authorization;
                 // console.log('Interceptor: No token found in localStorage.');
            }
            return config;
        },
        (error) => {
            // Keep existing error handling
            if (axios.isAxiosError(error) && error.response?.status === 401) {
                console.error('Auth error (401) detected by interceptor.');
                // TODO: Consider triggering a global logout mechanism here
            }
            return Promise.reject(error);
        }
    );
    interceptorInitialized = true;
    console.log('Axios interceptor INITIALIZED (reads localStorage directly).');
}

// --- IMMEDIATE Interceptor Setup on Load ---
setupAxiosInterceptor(); // Call once when the module loads
// -------------------------------------------

// Cyberpunk icons
const LogoIcon = () => (
    <svg 
        className={styles.iconEffect}
        viewBox="0 0 24 24" 
        xmlns="http://www.w3.org/2000/svg"
    >
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" 
            strokeWidth="1.5" 
            stroke="currentColor" 
            fill="none" 
            strokeLinecap="round" 
            strokeLinejoin="round"
        />
    </svg>
);

// Navigation icons
const ChatIcon = () => (
    <span className={styles.navButtonIcon}>💬</span>
);

const DocsIcon = () => (
    <span className={styles.navButtonIcon}>📚</span>
);

const PresentationIcon = () => (
    <span className={styles.navButtonIcon}>🎨</span>
);

const ResearchIcon = () => (
    <span className={styles.navButtonIcon}>🔬</span>
);

const ConfigIcon = () => (
    <span className={styles.navButtonIcon}>⚙️</span>
);

// Add Logout Icon
const LogoutIcon = () => (
    <span className={styles.navButtonIcon}>🚪</span>
);

// Add Admin Icon
const AdminIcon = () => (
    <span className={styles.navButtonIcon}>🛡️</span> // Example icon
);

function App() {
    const [activeTab, setActiveTab] = useState<'chat' | 'docs' | 'presentation' | 'research' | 'config' | 'admin'>('chat');
    const [authToken, setAuthToken] = useState<string | null>(localStorage.getItem('vibeRAG_authToken')); // Read initial directly
    const [isAdmin, setIsAdmin] = useState<boolean>(false);
    const [currentUserId, setCurrentUserId] = useState<number | null>(null);
    const [isAuthReady, setIsAuthReady] = useState<boolean>(false); 

    // Effect to process token changes and set auth ready flag
    useEffect(() => {
        console.log('AuthToken Effect triggered. Processing token...');
        setIsAuthReady(false); // Start processing

        // No need to call setupAxiosInterceptor here anymore
        
        // Process token status
        let ready = false;
        if (authToken) {
            const decoded = decodeJwt(authToken);
            setIsAdmin(decoded?.role === 'admin');
            setCurrentUserId(decoded?.id ?? null);
            console.log('User state set based on token.');
            ready = true; // Ready if token processed
        } else {
            setIsAdmin(false);
            setCurrentUserId(null);
            console.log('User state cleared (no token).');
            ready = true; // Also ready if no token (login screen)
        }
        
        // Signal that auth processing is complete *after* current execution context
        if (ready) {
            const timer = setTimeout(() => {
                setIsAuthReady(true);
                console.log('Auth processing complete. isAuthReady set to true via setTimeout.');
            }, 0); // Zero delay pushes to end of event loop
             // Cleanup the timer if the effect re-runs before it fires
             return () => clearTimeout(timer);
        }

        // If not ready for some reason (e.g., token decoding failed - though decodeJwt handles errors),
        // ensure cleanup is still possible.
        return () => {}; 

    }, [authToken]);

    // --- Handlers --- 
    const handleLoginSuccess = (token: string) => {
        localStorage.setItem('vibeRAG_authToken', token);
        setAuthToken(token); // Update state, which triggers the useEffect above
        console.log('Login successful, token state updated.');
    };

    const handleLogout = () => {
        localStorage.removeItem('vibeRAG_authToken');
        setAuthToken(null); // Update state to null, triggers useEffect
        console.log('Logout triggered, token state set to null.');
    };

    // --- Conditional Rendering --- 

    // 1. Initial load / Auth state processing
    if (!isAuthReady) {
        return <div className={styles.loadingApp}>Initializing...</div>;
    }

    // 2. Auth is ready, but no token -> Show Login
    if (!authToken) {
         return <Login onLoginSuccess={handleLoginSuccess} />;
    }

    // 3. Auth is ready AND token exists -> Show Main App Layout
    return (
        <div className={styles.container}>
            <div className={styles.sidebar}>
                <header className={styles.header}>
                    <div className={styles.headerTop}>
                        <LogoIcon />
                        <h1>vibeRAG</h1>
                    </div>
                    <p>Your AI Research Assistant</p>
                </header>

                <nav className={styles.nav}>
                    <button
                        className={`${styles.navButton} ${activeTab === 'chat' ? styles.active : ''}`}
                        onClick={() => setActiveTab('chat')}
                    >
                        <ChatIcon />
                        Chat
                    </button>
                    <button
                        className={`${styles.navButton} ${activeTab === 'docs' ? styles.active : ''}`}
                        onClick={() => setActiveTab('docs')}
                    >
                        <DocsIcon />
                        Documents
                    </button>
                    <button
                        className={`${styles.navButton} ${activeTab === 'presentation' ? styles.active : ''}`}
                        onClick={() => setActiveTab('presentation')}
                    >
                        <PresentationIcon />
                        Presentations
                    </button>
                    <button
                        className={`${styles.navButton} ${activeTab === 'research' ? styles.active : ''}`}
                        onClick={() => setActiveTab('research')}
                    >
                        <ResearchIcon />
                        Research
                    </button>
                    <button
                        className={`${styles.navButton} ${activeTab === 'config' ? styles.active : ''}`}
                        onClick={() => setActiveTab('config')}
                    >
                        <ConfigIcon />
                        Config
                    </button>
                    
                    {/* Conditional Admin Button */}
                    {isAdmin && (
                        <button
                            className={`${styles.navButton} ${activeTab === 'admin' ? styles.active : ''}`}
                            onClick={() => setActiveTab('admin')}
                        >
                            <AdminIcon />
                            Admin
                        </button>
                    )}

                    <button
                        className={styles.navButton}
                        onClick={handleLogout} // Add logout handler
                        title="Logout"
                    >
                        <LogoutIcon />
                        Logout
                    </button>
                </nav>

                <footer className={styles.footer}>
                    <p>Built with 🔥 by the vibeRAG crew</p>
                </footer>
            </div>

            <main className={styles.main}>
                {activeTab === 'chat' && <Chat isAuthReady={isAuthReady} />}
                {activeTab === 'docs' && <DocumentManager />}
                {activeTab === 'presentation' && <PresentationViewer isAuthReady={isAuthReady} />}
                {activeTab === 'research' && <ResearchReport isAuthReady={isAuthReady} />}
                {activeTab === 'config' && <Config />}
                {activeTab === 'admin' && isAdmin && currentUserId && <AdminPanel currentUserId={currentUserId} />}
                {activeTab === 'admin' && (!isAdmin || !currentUserId) && <p>Access Denied.</p>} 
            </main>
        </div>
    );
}

export default App;
