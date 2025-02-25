import React, { useState } from 'react';
import axios from 'axios';
import styles from './ResearchReport.module.css';

// Research vibes strong—knowledge report FTW!
interface ResearchReport {
  title: string;
  summary: string;
  insights: string[];
  analysis: string;
  sources: string[];
}

interface ResearchResponse {
  report: ResearchReport;
}

const ResearchReportViewer: React.FC = () => {
  const [query, setQuery] = useState('');
  const [useWeb, setUseWeb] = useState(true);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const { data } = await axios.post<ResearchResponse>('http://localhost:8000/research', {
        query,
        use_web: useWeb,
      });
      setReport(data.report);
      console.log('Research crew dropped some heat! 🔬');
    } catch (error) {
      console.error('Research request failed to vibe:', error);
      setError('No report yet, brah—try again! 😅');
    } finally {
      setLoading(false);
    }
  };

  const handleSourceClick = (source: string) => {
    // Extract filename from source (e.g., "Page 76 of whitepaper.pdf" -> "whitepaper.pdf")
    const match = source.match(/of\s+([^,\s]+\.pdf)/i);
    if (match) {
      const pdfFilename = match[1];
      // Open PDF in new tab
      window.open(`http://localhost:8000/get_pdf/${pdfFilename}`, '_blank');
      console.log('PDF vibes opening up! 📚');
    } else if (source.startsWith('Web:')) {
      // Extract URL from web source
      const match = source.match(/\((https?:\/\/[^)]+)\)/);
      if (match) {
        window.open(match[1], '_blank');
        console.log('Web vibes opening up! 🌐');
      }
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
            placeholder="What should we research? (e.g., 'AI and Machine Learning')"
            className={styles.textarea}
            rows={4}
          />
        </div>
        
        <div className={styles.toggleGroup}>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={useWeb}
              onChange={(e) => setUseWeb(e.target.checked)}
            />
            <span className={styles.toggleLabel}>Include Web Research</span>
          </label>
        </div>
        
        <button type="submit" className={styles.button} disabled={loading}>
          {loading ? 'Researching...' : 'Start Research 🚀'}
        </button>
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
              {report.summary.split('\n').map((paragraph, i) => (
                <p key={i}>{paragraph}</p>
              ))}
            </div>
          </section>
          
          {/* Insights Section */}
          <section className={styles.section}>
            <h2>Key Insights</h2>
            <ul className={styles.insights}>
              {report.insights.map((insight, i) => (
                <li key={i}>
                  <span className={styles.insightNumber}>{i + 1}</span>
                  {insight}
                </li>
              ))}
            </ul>
          </section>
          
          {/* Analysis Section */}
          <section className={styles.section}>
            <h2>Detailed Analysis</h2>
            <div className={styles.analysis}>
              {report.analysis.split('\n').map((paragraph, i) => (
                paragraph.trim() && <p key={i}>{paragraph}</p>
              ))}
            </div>
          </section>
          
          {/* Sources Section */}
          <section className={styles.section}>
            <h2>Sources</h2>
            <div className={styles.sourceNote}>Click sources to view documents 📚</div>
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