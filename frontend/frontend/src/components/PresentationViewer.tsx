import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './PresentationViewer.module.css';
import { ModelSelector } from './ModelSelector';

// Slide game strong—high-impact presentations!
interface Slide {
    title: string;
    content: string[];
    visual: string;
}

interface PresentationResponse {
    slides: Slide[];
    sources: string[];
}

// Add model configs interface
interface ModelConfig {
    provider: string;
    model: string;
}

const PresentationViewer: React.FC = () => {
    const [prompt, setPrompt] = useState('');
    const [filename, setFilename] = useState('');
    const [presentation, setPresentation] = useState<PresentationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [expandedSlides, setExpandedSlides] = useState<number[]>([]);
    const [modelConfig, setModelConfig] = useState<ModelConfig>({
        provider: 'openai',
        model: 'gpt-4',
    });

    // Fetch initial config
    useEffect(() => {
        const fetchConfig = async () => {
            try {
                const response = await axios.get('http://localhost:8000/api/config');
                const data = response.data as { chat: { provider: string; model: string } };
                setModelConfig({
                    provider: data.chat.provider, // Default to chat provider
                    model: data.chat.model, // Default to chat model
                });
            } catch (error) {
                console.error('Failed to fetch config:', error);
            }
        };
        fetchConfig();
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await axios.post('http://localhost:8000/api/presentation', {
                prompt,
                filename: filename || undefined,
                provider: modelConfig.provider,
                model: modelConfig.model,
            });
            const data = response.data as PresentationResponse;
            setPresentation(data);
            setExpandedSlides([]); // Reset expanded state
            console.log('Presentation slides are vibing! 🎨');
        } catch (error) {
            console.error('Presentation request failed to vibe:', error);
        } finally {
            setLoading(false);
        }
    };

    const toggleSlide = (index: number) => {
        setExpandedSlides(prev => 
            prev.includes(index) 
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    };

    const handleSourceClick = async (source: string) => {
        // Extract filename from source (e.g., "Page 76 of whitepaper.pdf" -> "whitepaper.pdf")
        const match = source.match(/of\s+([^,\s]+)$/);
        if (match && match[1]) {
            const filename = match[1];
            try {
                const response = await axios.get(`http://localhost:8000/get_pdf/${filename}`);
                // Add type assertion to fix the TypeScript error
                interface PDFResponse { url: string }
                window.open((response.data as PDFResponse).url, '_blank');
            } catch (error) {
                console.error('Failed to open PDF:', error);
            }
        }
    };

    return (
        <div className={styles.container}>
            <div className={styles.cyberHeader}>
                <h2 className={styles.title}>
                    <svg 
                        className={styles.presentationIcon}
                        xmlns="http://www.w3.org/2000/svg" 
                        viewBox="0 0 24 24"
                    >
                        <path d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h14v2H5v-2zm0-4h14v2H5V6zm0 8h14v2H5v-2z"
                        fill="currentColor"/>
                    </svg>
                    Generate Presentation
                </h2>
                <div className={styles.gridBackground}></div>
            </div>
            
            <form onSubmit={handleSubmit} className={styles.form}>
                <div className={styles.inputGroup}>
                    <input
                        type="text"
                        value={filename}
                        onChange={(e) => setFilename(e.target.value)}
                        placeholder="Optional: PDF filename (e.g., whitepaper.pdf)"
                        className={styles.input}
                    />
                </div>
                
                <div className={styles.inputGroup}>
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="What should the presentation be about?"
                        className={styles.textarea}
                        rows={4}
                    />
                </div>
                
                <div className={styles.modelSelector}>
                    <ModelSelector
                        provider={modelConfig.provider}
                        model={modelConfig.model}
                        onProviderChange={(provider) => setModelConfig({ ...modelConfig, provider })}
                        onModelChange={(model) => setModelConfig({ ...modelConfig, model })}
                    />
                </div>
                
                <button type="submit" className={styles.button} disabled={loading}>
                    {loading ? 'Creating...' : 'Generate Slides 🚀'}
                </button>
            </form>

            {presentation && (
                <div className={styles.presentation}>
                    <p className={styles.note}>Click to expand slides—presentation vibes FTW! 🚀</p>
                    
                    <div className={styles.slides}>
                        {presentation.slides.map((slide, index) => {
                            const isExpanded = expandedSlides.includes(index);
                            const bulletPoints = slide.content ? slide.content.filter(line => line.startsWith('•')).map(line => line.substring(1).trim()) : [];
                            const visual = slide.content ? slide.content.find(line => line.startsWith('Visual:'))?.substring(7).trim() : undefined;

                            return (
                                <div 
                                    key={index} 
                                    className={`${styles.slide} ${isExpanded ? styles.expanded : ''}`}
                                    onClick={() => toggleSlide(index)}
                                >
                                    <div className={styles.slideHeader}>
                                        <h2 className={styles.slideTitle}>{slide.title}</h2>
                                        <span className={styles.expandIcon}>
                                            {isExpanded ? '▼' : '▶'}
                                        </span>
                                    </div>
                                    
                                    <div className={`${styles.slideContent} ${isExpanded ? styles.visible : ''}`}>
                                        <ul>
                                            {bulletPoints.map((point: string, i: number) => (
                                                <li key={i}>{point}</li>
                                            ))}
                                        </ul>
                                        {visual && (
                                            <p className={styles.visual}>
                                                <em>{visual}</em>
                                            </p>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                    
                    {presentation.sources && presentation.sources.length > 0 && (
                        <div className={styles.sources}>
                            <h3>Sources:</h3>
                            <ul>
                                {presentation.sources.map((source, i) => (
                                    <li 
                                        key={i}
                                        onClick={() => handleSourceClick(source)}
                                        className={styles.sourceLink}
                                    >
                                        {source}
                                    </li>
                                ))}
                            </ul>
                            <p className={styles.sourceNote}>Click sources to view documents 📚</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default PresentationViewer; 