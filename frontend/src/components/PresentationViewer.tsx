import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './PresentationViewer.module.css';
import { ModelSelector } from './ModelSelector';
import API_ENDPOINTS from '../config/api';

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
    const [error, setError] = useState('');
    const [currentSlide, setCurrentSlide] = useState(0);
    const [downloading, setDownloading] = useState(false);
    const [numSlides, setNumSlides] = useState<number>(5);

    // Generate presentation slides
    const generatePresentation = async () => {
        if (!prompt.trim()) {
            setError('Please enter a presentation topic');
            return;
        }
        
        setLoading(true);
        setError('');
        
        try {
            const response = await axios.post(API_ENDPOINTS.PRESENTATION, {
                prompt: prompt,
                n_slides: numSlides,
                model: modelConfig.model,
                provider: modelConfig.provider
            });
            
            const responseData = response.data as { 
                slides: Slide[]; 
                sources: string[];
            };
            
            setPresentation(responseData);
            setExpandedSlides([]);
            console.log('Presentation slides received!');
        } catch (error) {
            console.error('Error generating presentation:', error);
            setError('Failed to generate presentation. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    // Download presentation as PDF
    const downloadPDF = async () => {
        if (!presentation || !presentation.slides) {
            setError('No presentation content to download.');
            return;
        }
        setDownloading(true);
        setError('');
        try {
            // NOTE: This requires the jsPDF library to be installed (`npm install jspdf` or `yarn add jspdf`)
            const { jsPDF } = await import('jspdf');
            const doc = new jsPDF();
            const pageHeight = doc.internal.pageSize.height;
            const pageWidth = doc.internal.pageSize.width;
            const margin = 15;
            let y = margin; // Current y position on the page

            const addText = (text: string, size: number, isBold: boolean = false) => {
                doc.setFontSize(size);
                doc.setFont('helvetica', isBold ? 'bold' : 'normal');
                const lines = doc.splitTextToSize(text, pageWidth - margin * 2);
                lines.forEach((line: string) => {
                    if (y + 5 > pageHeight - margin) { // Check for page break
                        doc.addPage();
                        y = margin;
                    }
                    doc.text(line, margin, y);
                    y += 7; // Adjust line height
                });
            };

            presentation.slides.forEach((slide, index) => {
                if (index > 0) {
                    doc.addPage();
                    y = margin;
                }
                addText(`Slide ${index + 1}: ${slide.title}`, 16, true);
                y += 5; // Space after title
                
                // Parse and add bullet points
                slide.content?.forEach(line => {
                    const bulletMatch = line.match(/^[-•*]\s*(.*)/);
                    if (bulletMatch && bulletMatch[1]) {
                        addText(`• ${bulletMatch[1]}`, 12);
                    } else if (!line.toLowerCase().startsWith('visual:')) { // Add non-bullet, non-visual lines
                        addText(line, 12);
                    }
                });
                 y += 10; // Space after content
            });

            // Add sources page if sources exist
            if (presentation.sources && presentation.sources.length > 0) {
                doc.addPage();
                y = margin;
                addText('Sources', 16, true);
                y += 5;
                presentation.sources.forEach(source => {
                    addText(source, 10);
                    y += 5; // Smaller line height for sources
                });
            }

            doc.save(`presentation_${Date.now()}.pdf`);
            // NOTE: End jsPDF Logic

        } catch (error: any) {
            console.error('Error generating or downloading presentation PDF:', error);
            setError('Failed to download presentation. Ensure jsPDF is installed and PDF generation logic is correct.');
        } finally {
            setDownloading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt.trim()) {
            setError('Please enter a presentation topic');
            return;
        }
        setLoading(true);
        setError(''); // Clear previous errors
        setPresentation(null); // Clear previous presentation
        try {
            const response = await axios.post(API_ENDPOINTS.PRESENTATION, {
                prompt,
                filename: filename || undefined,
                provider: modelConfig.provider,
                model: modelConfig.model,
                n_slides: numSlides, // Pass numSlides as n_slides
            });
            const data = response.data as PresentationResponse;
            setPresentation(data);
            setExpandedSlides([]); // Reset expanded state
            console.log('Presentation slides received!');
        } catch (error: any) {
            console.error('Presentation request failed:', error);
            setError(error.response?.data?.detail || error.message || 'Failed to generate presentation.');
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
        // Source is now just the filename
        const pdfFilename = source;
        // Remove complex regex matching
        // const match = source.match(/of\s+([^,\s]+)$/);
        // if (match && match[1]) { 
        //    const pdfFilename = match[1];
        try {
            const pdfUrl = API_ENDPOINTS.GET_DOCUMENT(pdfFilename);
            console.log("Attempting to open PDF from URL:", pdfUrl);
            window.open(pdfUrl, '_blank'); 

        } catch (error: any) {
            console.error('Failed to open PDF:', error);
            setError(error.response?.data?.detail || error.message || 'Could not open PDF source.');
        }
        // } else {
        //     console.error('Could not extract filename from source string:', source);
        //     setError('Could not identify the source document filename.');
        // }
    };

    return (
        <div className={styles.presentationContainer}>
            <div className={styles.controls}>
                <h2 className={styles.header}>Generate Presentation</h2>
                <form onSubmit={handleSubmit} className={styles.form}>
                    {/* Row 1: File Upload and Prompt */}
                    <div className={styles.inputRow}>
                        {/* File Upload Group */}
                        <div className={styles.inputGroup}>
                            <label htmlFor="pdfFile" className={styles.inputLabel}>Optional PDF Context</label>
                            <input
                                type="file"
                                id="pdfFile"
                                accept=".pdf"
                                onChange={(e) => {
                                    if (e.target.files && e.target.files.length > 0) {
                                        const file = e.target.files[0];
                                        setFilename(file.name);
                                    }
                                }}
                                className={styles.input}
                            />
                        </div>
                        {/* Prompt Group */}
                        <div className={styles.inputGroup}>
                            <label htmlFor="prompt" className={styles.inputLabel}>Presentation Prompt</label>
                            <textarea
                                id="prompt"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                required
                                placeholder="e.g., Create a 5-slide presentation about the future of AI"
                                className={styles.textarea}
                            />
                        </div>
                    </div>

                    {/* Row 2: Num Slides, Model Selector, and Button */}
                    <div className={styles.inputRow} style={{ alignItems: 'flex-end' }}>
                        {/* Num Slides Group */}
                        <div className={styles.inputGroup}>
                            <label htmlFor="numSlides" className={styles.inputLabel}>Number of Slides</label>
                            <input
                                type="number"
                                id="numSlides"
                                value={numSlides}
                                onChange={(e) => setNumSlides(parseInt(e.target.value, 10))}
                                min="1"
                                max="20"
                                required
                                className={styles.input}
                            />
                        </div>
                        {/* Model Selector Group */}
                        <div className={styles.modelSelectorGroup}>
                            <label htmlFor="llmModel" className={styles.inputLabel}>LLM Model</label>
                            <ModelSelector 
                                provider={modelConfig.provider}
                                model={modelConfig.model}
                                onProviderChange={(provider) => setModelConfig({ ...modelConfig, provider })}
                                onModelChange={(model) => setModelConfig({ ...modelConfig, model })}
                            />
                        </div>
                        {/* Submit Button */}
                        <button type="submit" disabled={loading} className={styles.button}>
                            {loading ? 'Generating...' : 'Generate Slides'}
                        </button>
                    </div>

                    {error && <p className={styles.error}>{error}</p>}
                </form>
            </div>

            {presentation && (
                <div className={styles.presentation}>
                    <button 
                        onClick={downloadPDF} 
                        className={styles.downloadButton} 
                        disabled={downloading}
                    >
                        {downloading ? 'Downloading...' : 'Download PDF 📄'}
                    </button>

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