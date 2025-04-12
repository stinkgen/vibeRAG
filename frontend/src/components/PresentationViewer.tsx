import React, { useState } from 'react';
import axios, { AxiosError } from 'axios';
import styles from './PresentationViewer.module.css';
import { ModelSelector } from './ModelSelector';
import API_ENDPOINTS from '../config/api';
import { useModelProviderSelection } from '../hooks/useModelProviderSelection';
import { jsPDF } from "jspdf";

// Slide game strong—high-impact presentations!
interface Slide {
    title: string;
    content: string[];
    visual?: string;
}

interface PresentationResponse {
    slides: Slide[];
    sources?: string[];
}

// Define component props
interface PresentationViewerProps {
    isAuthReady: boolean;
}

const PresentationViewer: React.FC<PresentationViewerProps> = ({ isAuthReady }) => {
    const [prompt, setPrompt] = useState('');
    const [numSlides, setNumSlides] = useState<number>(5);
    const [filename, setFilename] = useState<string>('');
    const [presentation, setPresentation] = useState<PresentationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [expandedSlides, setExpandedSlides] = useState<number[]>([]);
    const [downloading, setDownloading] = useState(false);
    const [currentSlideIndex, setCurrentSlideIndex] = useState(0);

    const { currentModel, currentProvider } = useModelProviderSelection(isAuthReady);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt.trim()) {
            setError('Please enter a presentation prompt.');
            return;
        }
        setLoading(true);
        setError(null);
        setPresentation(null);
        setCurrentSlideIndex(0);

        try {
            const response = await axios.post<PresentationResponse>(API_ENDPOINTS.PRESENTATION, {
                prompt,
                filename: filename || undefined,
                provider: currentProvider,
                model: currentModel,
                n_slides: numSlides,
            });
            setPresentation(response.data);
            setExpandedSlides([]);
        } catch (err: any) {
            console.error('Presentation request failed:', err);
            const errorDetail = err.response?.data?.detail || err.message || 'Failed to generate presentation.';
            setError(errorDetail);
        } finally {
            setLoading(false);
        }
    };

    const downloadPDF = async () => {
        if (!presentation || !presentation.slides || presentation.slides.length === 0) {
            setError('No presentation content to download.');
            return;
        }
        setDownloading(true);
        setError(null);
        try {
            const doc = new jsPDF();
            const pageHeight = doc.internal.pageSize.height;
            const pageWidth = doc.internal.pageSize.width;
            const margin = 15;
            let y = margin;

            const addWrappedText = (text: string, size: number, isBold = false, indent = 0) => {
                doc.setFontSize(size);
                doc.setFont('helvetica', isBold ? 'bold' : 'normal');
                const lines = doc.splitTextToSize(text, pageWidth - margin * 2 - indent);
                lines.forEach((line: string) => {
                    if (y + 7 > pageHeight - margin) {
                        doc.addPage();
                        y = margin;
                    }
                    doc.text(line, margin + indent, y);
                    y += 7;
                });
            };

            presentation.slides.forEach((slide, index) => {
                if (index > 0) {
                    doc.addPage();
                    y = margin;
                }
                addWrappedText(`Slide ${index + 1}: ${slide.title}`, 16, true);
                y += 5;

                if (slide.content && slide.content.length > 0) {
                    slide.content.forEach(line => {
                        if (line.trim().startsWith('-') || line.trim().startsWith('*') || line.trim().startsWith('•')) {
                            addWrappedText(line.trim(), 12, false, 5);
                        } else {
                            addWrappedText(line.trim(), 12);
                        }
                    });
                    y += 5;
                } else {
                    addWrappedText("[No content provided for this slide]", 12, false);
                    y += 5;
                }

                if (slide.visual) {
                    addWrappedText(`Visual Suggestion: ${slide.visual}`, 10, false);
                    y += 10;
                }
            });

            if (presentation.sources && presentation.sources.length > 0) {
                doc.addPage();
                y = margin;
                addWrappedText('Sources', 16, true);
                y += 5;
                presentation.sources.forEach(source => {
                    addWrappedText(source, 10);
                    y += 5;
                });
            }

            doc.save(`presentation_${prompt.substring(0, 15).replace(/\s+/g, '_')}.pdf`);

        } catch (error: any) {
            console.error('Error generating presentation PDF:', error);
            setError('Failed to download presentation PDF.');
        } finally {
            setDownloading(false);
        }
    };

    const toggleSlide = (index: number) => {
        setExpandedSlides(prev =>
            prev.includes(index)
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    };

    const goToNextSlide = () => {
        if (presentation && currentSlideIndex < presentation.slides.length - 1) {
            setCurrentSlideIndex(currentSlideIndex + 1);
        }
    };

    const goToPrevSlide = () => {
        if (currentSlideIndex > 0) {
            setCurrentSlideIndex(currentSlideIndex - 1);
        }
    };

    const handleSourceClick = async (source: string) => {
        const pdfFilename = source;
        try {
            const pdfUrl = API_ENDPOINTS.GET_DOCUMENT(pdfFilename);
            console.log("Attempting to open PDF source:", pdfUrl);
            window.open(pdfUrl, '_blank', 'noopener,noreferrer');
        } catch (error: any) {
            console.error('Failed to open PDF source:', error);
            setError(error.response?.data?.detail || error.message || 'Could not open PDF source.');
        }
    };

    return (
        <div className={styles.container}>
            <div className={styles.gridBackground}></div>
            
            <div className={styles.cyberHeader}>
                 <h1 className={styles.title}>
                     Generate Presentation
                 </h1>
            </div>

            <form onSubmit={handleSubmit} className={styles.form}>
                 <div className={styles.inputRow}>
                     <div className={styles.inputGroup} style={{ flexGrow: 2 }}>
                         <label className={styles.inputLabel} htmlFor="prompt">Presentation Prompt</label>
                         <textarea
                             id="prompt"
                             value={prompt}
                             onChange={(e) => setPrompt(e.target.value)}
                             required
                             placeholder="e.g., Create a presentation about the future of AI"
                             className={styles.textarea}
                         />
                     </div>
                     <div className={styles.inputGroup} >
                         <label className={styles.inputLabel} htmlFor="numSlides">Number of Slides</label>
                         <input
                             type="number"
                             id="numSlides"
                             value={numSlides}
                             onChange={(e) => setNumSlides(parseInt(e.target.value, 10) || 1)}
                             min="1"
                             max="20"
                             className={styles.input}
                         />
                     </div>
                     <div className={styles.inputGroup}>
                         <label className={styles.inputLabel} htmlFor="pdfFile">Optional PDF Context</label>
                         <input
                             type="file"
                             id="pdfFile"
                             accept=".pdf"
                             onChange={(e) => {
                                 setFilename(e.target.files && e.target.files.length > 0 ? e.target.files[0].name : '');
                             }}
                             className={styles.input}
                         />
                     </div>
                 </div>
                 <div className={styles.inputRow} style={{ alignItems: 'flex-end', marginTop: '1rem' }}>
                     <div className={styles.modelSelectorGroup}>
                         <label className={styles.inputLabel}>LLM Model</label>
                         <ModelSelector isAuthReady={isAuthReady} />
                     </div>
                     <button type="submit" className={styles.button} disabled={loading || !prompt.trim()}>
                         {loading ? 'Generating...' : 'Generate Slides'}
                     </button>
                 </div>
                 {error && <p className={styles.errorMessage}>{error}</p>}
             </form>

            {presentation && (
                <div className={styles.presentation}>
                    <p className={styles.note}>Click to expand slides—presentation vibes FTW! 🚀</p>
                    <div className={styles.slides}>
                        {presentation.slides.map((slide, index) => (
                            <div 
                                key={index} 
                                className={`${styles.slide} ${expandedSlides.includes(index) ? styles.expanded : ''}`}
                                onClick={() => toggleSlide(index)}
                            >
                                 <div className={styles.slideHeader}> 
                                     <h4 className={styles.slideTitle}>{slide.title || 'Untitled Slide'}</h4>
                                     <span className={styles.expandIcon}>{expandedSlides.includes(index) ? '▼' : '▶'}</span>
                                 </div>
                                 <div className={`${styles.slideContent} ${expandedSlides.includes(index) ? styles.visible : ''}`}> 
                                     <ul>
                                         {slide.content?.map((item, idx) => (
                                            <li key={idx}>{item}</li>
                                        ))}
                                    </ul>
                                     {slide.visual && <p className={styles.visual}><i>Visual: {slide.visual}</i></p>} 
                                 </div>
                             </div>
                        ))}
                    </div>

                    {presentation.sources && presentation.sources.length > 0 && (
                        <div className={styles.sources}>
                             <h3>Sources Used:</h3>
                             <ul>
                                 {presentation.sources.map((source, index) => (
                                    <li key={index}>
                                         <span className={styles.sourceLink} onClick={() => handleSourceClick(source)}>
                                             {source}
                                         </span>
                                     </li>
                                ))}
                             </ul>
                         </div>
                    )}

                     <button onClick={downloadPDF} className={styles.downloadButton} disabled={downloading}>
                         {downloading ? 'Downloading...' : 'Download PDF'}
                     </button>
                 </div>
             )}
        </div>
    );
};

export default PresentationViewer; 