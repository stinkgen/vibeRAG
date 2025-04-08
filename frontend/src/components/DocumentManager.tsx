import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import styles from './DocumentManager.module.css';
import API_ENDPOINTS from '../config/api';

// Doc management types—keeping it organized!
interface DocInfo {
    doc_id: string;
    filename: string;
    tags: string[];
    metadata: Record<string, any>;
    [key: string]: any; // Add index signature for dynamic access
}

interface UploadResponse {
    filename: string;
    num_chunks: number;
    tags: string[];
    metadata: Record<string, string>;
    status: string;
}

const DocumentIcon = () => (
    <svg 
        className={styles.documentIcon}
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 24 24"
    >
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm4 18H6V4h7v5h5v11zM8 15h8v2H8v-2zm0-4h8v2H8v-2z" 
            fill="currentColor" 
            stroke="none"
        />
    </svg>
);

// Helper function to truncate long strings
const truncateString = (str: string, maxLength: number): string => {
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength - 3) + '...';
};

const DocumentManager: React.FC = () => {
    // State management vibes
    const [activeTab, setActiveTab] = useState<'upload' | 'manage'>('manage');
    const [docs, setDocs] = useState<DocInfo[]>([]);
    const [file, setFile] = useState<File | null>(null);
    const [tags, setTags] = useState('');
    const [collection, setCollection] = useState('');
    const [metadataKey, setMetadataKey] = useState('');
    const [metadataValue, setMetadataValue] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);

    // Add view type state
    const [viewType, setViewType] = useState<'card' | 'list'>('card');
    const [sortField, setSortField] = useState<string>('filename');
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
    
    // Ref for file input
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Load docs on mount
    useEffect(() => {
        fetchDocs();
    }, []);

    // Fetch all docs from the backend
    const fetchDocs = async () => {
        try {
            const response = await axios.get(API_ENDPOINTS.LIST_DOCUMENTS);
            // Ensure we're getting an array
            const docsData = Array.isArray(response.data) ? response.data : [];
            setDocs(docsData);
            console.log('Docs loaded—knowledge base vibing! 📚');
        } catch (error) {
            console.error('Failed to fetch docs:', error);
            setError('Failed to load docs, brah! Try refreshing 😅');
            setDocs([]); // Reset to empty array on error
        }
    };

    // Handle drag events
    const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    // Handle drop event
    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    // Handle file input click
    const handleDropzoneClick = () => {
        fileInputRef.current?.click();
    };

    // Handle file selection via input
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    // Handle file upload with tags and metadata
    const handleUpload = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) {
            setError('No file selected, brah! 😅');
            return;
        }

        setLoading(true);
        setError(null);
        setSuccessMessage(null);

        const formData = new FormData();
        formData.append('file', file);
        
        // Parse tags and metadata
        const tagList = tags.split(',').map(t => t.trim()).filter(t => t);
        const metadata: Record<string, string> = {
            collection: collection || 'default'
        };
        if (metadataKey && metadataValue) {
            metadata[metadataKey] = metadataValue;
        }

        formData.append('tags', JSON.stringify(tagList));
        formData.append('metadata', JSON.stringify(metadata));

        try {
            const response = await axios.post<UploadResponse>(API_ENDPOINTS.UPLOAD, formData);
            console.log('Upload complete—doc vibes strong! 📂');
            
            // Set success message with chunk count
            setSuccessMessage(`Successfully uploaded ${file.name} with ${response.data.num_chunks} chunks! 🎯`);
            
            // Reset form and refresh docs
            setFile(null);
            setTags('');
            setCollection('');
            setMetadataKey('');
            setMetadataValue('');
            await fetchDocs();  // Wait for docs to refresh
            
            // Switch to list view after successful upload
            setTimeout(() => {
                setActiveTab('manage');
            }, 1500);
            
        } catch (error) {
            console.error('Upload failed:', error);
            setError('Upload failed, brah—try again! 😅');
        } finally {
            setLoading(false);
        }
    };

    // Handle document deletion
    const handleDelete = async (filename: string) => {
        if (!window.confirm('Sure you wanna delete this doc, brah?')) {
            return;
        }

        try {
            await axios.delete(API_ENDPOINTS.DELETE_DOCUMENT_BY_FILENAME(filename));
            console.log('Doc deleted—cleanup complete! 🧹');
            fetchDocs();
        } catch (error) {
            console.error('Delete failed:', error);
            setError('Delete failed, brah! Try again 😅');
        }
    };

    // Handle search
    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        // The filtering is already happening in filteredDocs
        console.log('Searching for:', searchTerm);
    };

    // Filter docs based on search term
    const filteredDocs = docs.filter(doc => {
        if (!searchTerm) return true; // Show all docs when no search term
        
        const searchLower = searchTerm.toLowerCase();
        return (
            doc.filename.toLowerCase().includes(searchLower) ||
            doc.tags.some(tag => tag.toLowerCase().includes(searchLower)) ||
            Object.entries(doc.metadata).some(([key, value]) => 
                key.toLowerCase().includes(searchLower) ||
                value.toString().toLowerCase().includes(searchLower)
            )
        );
    });

    const getSortValue = (doc: DocInfo, field: string): any => {
        // Handle nested paths (e.g., 'metadata.page')
        const parts = field.split('.');
        let value: any = doc;
        for (const part of parts) {
            value = value?.[part];
        }
        return value ?? '';
    };

    // Sort documents
    const sortedDocs = [...filteredDocs].sort((a, b) => {
        const aValue = getSortValue(a, sortField);
        const bValue = getSortValue(b, sortField);
        return sortDirection === 'asc'
            ? String(aValue).localeCompare(String(bValue))
            : String(bValue).localeCompare(String(aValue));
    });

    // Format metadata value for display
    const formatMetadataValue = (value: any): string => {
        if (typeof value === 'string' && (value.startsWith('[') || value.startsWith('{'))) {
            try {
                const parsed = JSON.parse(value);
                return Array.isArray(parsed) ? parsed.join(', ') : JSON.stringify(parsed, null, 2);
            } catch {
                return value;
            }
        }
        return String(value);
    };

    // Render document list or grid
    const renderDocuments = () => {
        if (sortedDocs.length === 0) {
            return (
                <div className={styles.noResults}>
                    {searchTerm ? 'No documents found matching your search.' : 'No documents uploaded yet.'}
                </div>
            );
        }

        return (
            <div className={viewType === 'card' ? styles.docGrid : styles.docList}>
                {sortedDocs.map(doc => (
                    <div key={doc.doc_id} className={`${styles.docItem} ${viewType === 'card' ? styles.docCard : styles.docRow}`}>
                        <div className={styles.docHeader}>
                            <h3 className={styles.docTitle}>
                                <a
                                    href={API_ENDPOINTS.GET_DOCUMENT(doc.filename)}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className={styles.documentLink}
                                >
                                    {truncateString(doc.filename, 25)}
                                </a>
                            </h3>
                            <button
                                onClick={() => handleDelete(doc.filename)}
                                className={styles.deleteButton}
                                title="Delete document"
                            >
                                🗑️
                            </button>
                        </div>

                        {doc.tags.length > 0 && (
                            <div className={styles.docTags}>
                                {doc.tags.map((tag, i) => (
                                    <span key={i} className={styles.tag}>
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        )}

                        <div className={styles.docMetadata}>
                            {Object.entries(doc.metadata).map(([key, value]) => (
                                <div key={key} className={styles.metadataItem}>
                                    <span className={styles.metadataKey}>{key}:</span>
                                    <span className={styles.metadataValue}>
                                        {formatMetadataValue(value)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        );
    };

    return (
        <div className={styles.container}>
            <div className={styles.gridBackground}></div>
            <div className={styles.containerHeader}>
                <h1 className={styles.title}>
                    <DocumentIcon />
                    Document Manager
                </h1>
                <div className={styles.tabs}>
                    <button
                        className={`${styles.tab} ${activeTab === 'upload' ? styles.active : ''}`}
                        onClick={() => setActiveTab('upload')}
                    >
                        Upload
                    </button>
                    <button
                        className={`${styles.tab} ${activeTab === 'manage' ? styles.active : ''}`}
                        onClick={() => setActiveTab('manage')}
                    >
                        Manage
                    </button>
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className={styles.error}>
                    {error}
                </div>
            )}

            {/* Success Message */}
            {successMessage && (
                <div className={styles.success}>
                    {successMessage}
                </div>
            )}

            {/* Document List View */}
            {activeTab === 'manage' && (
                <>
                    <div className={styles.listControls}>
                        <form onSubmit={handleSearch} className={styles.searchBar}>
                            <input
                                type="text"
                                value={searchTerm}
                                onChange={e => setSearchTerm(e.target.value)}
                                placeholder="Search documents..."
                                className={styles.searchInput}
                            />
                        </form>
                        
                        <div className={styles.viewControls}>
                            <button
                                className={`${styles.viewButton} ${viewType === 'card' ? styles.active : ''}`}
                                onClick={() => setViewType('card')}
                                title="Card View"
                            >
                                📱
                            </button>
                            <button
                                className={`${styles.viewButton} ${viewType === 'list' ? styles.active : ''}`}
                                onClick={() => setViewType('list')}
                                title="List View"
                            >
                                📋
                            </button>
                        </div>

                        <select
                            className={styles.sortSelect}
                            value={sortField}
                            onChange={e => setSortField(e.target.value)}
                        >
                            <option value="filename">Filename</option>
                            <option value="metadata.total_pages">Pages</option>
                            <option value="metadata.collection">Collection</option>
                        </select>

                        <button
                            className={styles.sortButton}
                            onClick={() => setSortDirection(d => d === 'asc' ? 'desc' : 'asc')}
                            title="Toggle Sort Direction"
                        >
                            {sortDirection === 'asc' ? '↑' : '↓'}
                        </button>
                    </div>

                    {renderDocuments()}
                </>
            )}

            {/* Upload Form */}
            {activeTab === 'upload' && (
                <form onSubmit={handleUpload} className={styles.uploadForm}>
                    <div 
                        className={`${styles.dropzone} ${dragActive ? styles.dragActive : ''}`}
                        onClick={handleDropzoneClick}
                        onDragEnter={handleDrag}
                        onDragOver={handleDrag}
                        onDragLeave={handleDrag}
                        onDrop={handleDrop}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            onChange={handleFileChange}
                            className={styles.fileInput}
                            accept=".pdf,.txt,.doc,.docx"
                        />
                        <p>Drop your doc here or click to browse 📂</p>
                        {file && <p className={styles.fileName}>{file.name}</p>}
                    </div>

                    <div className={styles.inputGroup}>
                        <input
                            type="text"
                            value={tags}
                            onChange={e => setTags(e.target.value)}
                            placeholder="Tags (comma-separated)"
                            className={styles.input}
                        />
                    </div>

                    <div className={styles.inputGroup}>
                        <input
                            type="text"
                            value={collection}
                            onChange={e => setCollection(e.target.value)}
                            placeholder="Collection (e.g., Tech Papers)"
                            className={styles.input}
                        />
                    </div>

                    <div className={styles.metadataGroup}>
                        <input
                            type="text"
                            value={metadataKey}
                            onChange={e => setMetadataKey(e.target.value)}
                            placeholder="Metadata Key"
                            className={styles.input}
                        />
                        <input
                            type="text"
                            value={metadataValue}
                            onChange={e => setMetadataValue(e.target.value)}
                            placeholder="Metadata Value"
                            className={styles.input}
                        />
                    </div>

                    <button
                        type="submit"
                        className={styles.button}
                        disabled={loading}
                    >
                        {loading ? 'Uploading...' : 'Upload Doc 🚀'}
                    </button>
                </form>
            )}
        </div>
    );
};

export default DocumentManager; 