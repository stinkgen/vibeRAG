import React, { useState } from 'react';
import Chat from './components/Chat';
import DocumentManager from './components/DocumentManager';
import PresentationViewer from './components/PresentationViewer';
import ResearchReport from './components/ResearchReport';
import Config from './components/Config';
import styles from './App.module.css';
import './styles/global.css';

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

function App() {
    const [activeTab, setActiveTab] = useState<'chat' | 'docs' | 'presentation' | 'research' | 'config'>('chat');

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
                </nav>

                <footer className={styles.footer}>
                    <p>Built with 🔥 by the vibeRAG crew</p>
                </footer>
            </div>

            <main className={styles.main}>
                {activeTab === 'chat' && <Chat />}
                {activeTab === 'docs' && <DocumentManager />}
                {activeTab === 'presentation' && <PresentationViewer />}
                {activeTab === 'research' && <ResearchReport />}
                {activeTab === 'config' && <Config />}
            </main>
        </div>
    );
}

export default App;
