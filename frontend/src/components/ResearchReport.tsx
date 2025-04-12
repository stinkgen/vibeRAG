import React, { useState, useEffect, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import styles from './ResearchReport.module.css';
import API_ENDPOINTS from '../config/api';
import { useModelProviderSelection } from '../hooks/useModelProviderSelection';
import { ModelSelector } from './ModelSelector';

// Research vibes strong—knowledge report FTW!
interface ResearchReportData {
  title: string;
  summary: string;
  insights: string[];
  analysis: string;
  sources: string[];
}

interface ResearchResponse {
  report: ResearchReportData;
}

// Define component props
interface ResearchReportProps {
  isAuthReady: boolean;
}

const ResearchReportViewer: React.FC<ResearchReportProps> = ({ isAuthReady }) => {
  const [query, setQuery] = useState<string>('');
  const [useWeb, setUseWeb] = useState<boolean>(false);
  const [useKnowledge, setUseKnowledge] = useState<boolean>(true);
  const [report, setReport] = useState<ResearchReportData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Use the hook for model/provider selection, passing isAuthReady
  const {
    currentModel,
    setCurrentModel,
    currentProvider,
    setCurrentProvider,
    ollamaModels,
    openaiModels,
    loadingStatus: modelLoadingStatus,
    ollamaStatus,
    providerError: modelProviderError
  } = useModelProviderSelection(isAuthReady);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) {
      setError('Please enter a research query.');
      return;
    }
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const { data } = await axios.post<ResearchResponse>(API_ENDPOINTS.RESEARCH, {
        query: query,
        use_web: useWeb,
        use_knowledge: useKnowledge,
        model: currentModel,
        provider: currentProvider
      });

      setReport(data.report);
    } catch (err: any) {
      console.error('Research generation failed:', err);
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (axios.isAxiosError(err) && err.message) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred during research generation.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSourceClick = (source: string) => {
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
      <div className={styles.cyberHeader}>
        <h2 className={styles.title}>
          <svg 
            className={styles.researchIcon}
            xmlns="http://www.w3.org/2000/svg" 
            viewBox="0 0 24 24"
          >
            <path d="M9.5 3C7.56 3 5.96 4.51 5.75 6.42L2 11.92V20C2 20.55 2.45 21 3 21H10C10.55 21 11 20.55 11 20V16H13V20C13 20.55 13.45 21 14 21H21C21.55 21 22 20.55 22 20V11.92L18.25 6.42C18.04 4.51 16.44 3 14.5 3H9.5ZM9.5 5H14.5C15.33 5 16 5.67 16 6.5V7H8V6.5C8 5.67 8.67 5 9.5 5ZM18 10H20L22 12V19H15V16C15 15.45 14.55 15 14 15H10C9.45 15 9 15.45 9 16V19H3V12L5 10H7V9H17V10H18Z" 
              fill="currentColor"
            />
          </svg>
          Research Report
        </h2>
        <div className={styles.gridBackground}></div>
      </div>
      
      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.inputGroup}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What should we research?"
            className={styles.textarea}
            rows={3}
            required
          />
        </div>
        
        <div className={styles.controlsRow}>
          <div className={styles.toggleGroup}>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={useWeb}
                onChange={(e) => setUseWeb(e.target.checked)}
              />
              <span className={styles.toggleLabel}>Web Search</span>
            </label>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={useKnowledge}
                onChange={(e) => setUseKnowledge(e.target.checked)}
              />
              <span className={styles.toggleLabel}>Knowledge Base</span>
            </label>
          </div>

          <ModelSelector 
            isAuthReady={isAuthReady}
          />

          <button type="submit" className={styles.button} disabled={loading || !query.trim() || modelLoadingStatus}>
            {loading ? 'Researching...' : 'Generate Report'}
          </button>
        </div>
      </form>

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}

      {report && (
        <div className={styles.report}>
          {/* Title Section */}
          <h1 className={styles.reportTitle}>{report.title}</h1>
          
          {/* Summary Section */}
          <section className={styles.section}>
            <h2>Executive Summary</h2>
            <div className={styles.content}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{report.summary}</ReactMarkdown>
            </div>
          </section>
          
          {/* Insights Section */}
          <section className={styles.section}>
            <h2>Key Insights</h2>
            <ul className={styles.insights}>
              {report.insights.map((insight, i) => (
                <li key={i}>
                  <span className={styles.insightNumber}>{i + 1}</span>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{insight}</ReactMarkdown>
                </li>
              ))}
            </ul>
          </section>
          
          {/* Analysis Section */}
          <section className={styles.section}>
            <h2>Detailed Analysis</h2>
            <div className={styles.analysis}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{report.analysis}</ReactMarkdown>
            </div>
          </section>
          
          {/* Sources Section */}
          <section className={styles.section}>
            <h2>Sources</h2>
            <div className={styles.sourceNote}>Click source filename to view document</div>
            <ul className={styles.sources}>
              {report.sources.map((source, i) => (
                <li key={i}>
                  <button 
                    onClick={() => handleSourceClick(source)}
                    className={styles.sourceLink}
                  >
                    {source}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </div>
  );
};

export default ResearchReportViewer; 