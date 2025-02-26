import React, { useState, useEffect, useRef } from 'react';
import styles from './KnowledgeFilter.module.css';
import API_ENDPOINTS from '../config/api';

interface FilterOption {
  id: string;
  name: string;
  type: 'file' | 'collection' | 'tag';
  metadata?: {
    pages?: number;
    fileSize?: string;
    timestamp?: string;
    description?: string;
  };
}

interface KnowledgeFilterProps {
  onFilterChange: (selectedOptions: FilterOption[]) => void;
  initialFilters?: FilterOption[];
}

const KnowledgeFilter: React.FC<KnowledgeFilterProps> = ({ onFilterChange, initialFilters = [] }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [filterType, setFilterType] = useState<'file' | 'collection' | 'tag'>('file');
  const [searchQuery, setSearchQuery] = useState('');
  const [options, setOptions] = useState<FilterOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedOptions, setSelectedOptions] = useState<FilterOption[]>(initialFilters);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch filter options based on filter type
  useEffect(() => {
    const fetchOptions = async () => {
      setLoading(true);
      try {
        let endpoint = '';
        if (filterType === 'file') {
          endpoint = API_ENDPOINTS.DOCUMENTS;
        } else if (filterType === 'collection') {
          endpoint = API_ENDPOINTS.COLLECTIONS;
        } else if (filterType === 'tag') {
          endpoint = API_ENDPOINTS.TAGS;
        }
        
        const response = await fetch(endpoint);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${filterType} options`);
        }
        
        const data = await response.json();
        
        // Transform data based on the structure expected from each endpoint
        let transformedData: FilterOption[] = [];
        if (filterType === 'file') {
          transformedData = data.documents.map((doc: any) => ({
            id: doc.id,
            name: doc.filename || doc.title || `File ${doc.id}`,
            type: 'file',
            metadata: {
              pages: doc.pages || null,
              fileSize: doc.file_size || null,
              timestamp: doc.uploaded_at || null,
              description: doc.description || null
            }
          }));
        } else if (filterType === 'collection') {
          transformedData = data.collections.map((collection: any) => ({
            id: collection.id,
            name: collection.name || `Collection ${collection.id}`,
            type: 'collection',
            metadata: {
              description: collection.description || null
            }
          }));
        } else if (filterType === 'tag') {
          transformedData = data.tags.map((tag: any) => ({
            id: tag.id || tag.name,
            name: tag.name,
            type: 'tag',
            metadata: {
              description: tag.description || null
            }
          }));
        }
        
        setOptions(transformedData);
      } catch (error) {
        console.error(`Error fetching ${filterType} options:`, error);
        // If API fails, provide some sample data for demonstration
        if (filterType === 'file') {
          const sampleData: FilterOption[] = [
            { 
              id: 'whitepaper_pdf', 
              name: 'whitepaper.pdf', 
              type: 'file',
              metadata: {
                pages: 110,
                fileSize: '8.2 MB',
                timestamp: new Date().toISOString()
              }
            },
            { 
              id: 'documentation_pdf', 
              name: 'documentation.pdf', 
              type: 'file',
              metadata: {
                pages: 45,
                fileSize: '2.5 MB',
                timestamp: new Date().toISOString()
              }
            }
          ];
          setOptions(sampleData);
        } else {
          setOptions([]);
        }
      } finally {
        setLoading(false);
      }
    };
    
    if (isOpen) {
      fetchOptions();
    }
  }, [filterType, isOpen]);

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Filter options based on search query
  const filteredOptions = searchQuery.trim() === '' 
    ? options 
    : options.filter(option => 
        option.name.toLowerCase().includes(searchQuery.toLowerCase())
      );

  // Toggle option selection
  const toggleOption = (option: FilterOption) => {
    const isSelected = selectedOptions.some(item => item.id === option.id && item.type === option.type);
    
    const newSelectedOptions = isSelected
      ? selectedOptions.filter(item => !(item.id === option.id && item.type === option.type))
      : [...selectedOptions, option];
    
    setSelectedOptions(newSelectedOptions);
    onFilterChange(newSelectedOptions);
  };

  // Remove a selected option
  const removeOption = (option: FilterOption) => {
    const newSelectedOptions = selectedOptions.filter(
      item => !(item.id === option.id && item.type === option.type)
    );
    setSelectedOptions(newSelectedOptions);
    onFilterChange(newSelectedOptions);
  };

  // Get the badge color based on filter type
  const getBadgeColor = (type: 'file' | 'collection' | 'tag') => {
    switch (type) {
      case 'file':
        return styles.fileBadge;
      case 'collection':
        return styles.collectionBadge;
      case 'tag':
        return styles.tagBadge;
      default:
        return '';
    }
  };

  // Format file size in human-readable format
  const formatFileSize = (size: string | undefined) => {
    if (!size) return '';
    return size;
  };

  // Format timestamp to readable date
  const formatDate = (timestamp: string | undefined) => {
    if (!timestamp) return '';
    try {
      const date = new Date(timestamp);
      return date.toLocaleDateString();
    } catch (e) {
      return timestamp;
    }
  };

  return (
    <div className={styles.filterContainer} ref={dropdownRef}>
      <div className={styles.selectedFilters}>
        {selectedOptions.length > 0 ? (
          selectedOptions.map(option => (
            <div key={`${option.type}-${option.id}`} className={`${styles.filterBadge} ${getBadgeColor(option.type)}`}>
              <span className={styles.badgeType}>{option.type}:</span>
              <span className={styles.badgeName}>{option.name}</span>
              <button 
                className={styles.badgeRemove} 
                onClick={() => removeOption(option)}
                title="Remove filter"
              >
                ×
              </button>
            </div>
          ))
        ) : (
          <div className={styles.placeholderText}>No filters selected</div>
        )}
        <button 
          className={styles.addFilterButton} 
          onClick={() => setIsOpen(!isOpen)}
          title="Add filter"
        >
          {isOpen ? '−' : '+'}
        </button>
      </div>
      
      {isOpen && (
        <div className={styles.filterDropdown}>
          <div className={styles.filterTypeSelector}>
            <button 
              className={`${styles.filterTypeButton} ${filterType === 'file' ? styles.activeFilterType : ''}`}
              onClick={() => setFilterType('file')}
            >
              Files
            </button>
            <button 
              className={`${styles.filterTypeButton} ${filterType === 'collection' ? styles.activeFilterType : ''}`}
              onClick={() => setFilterType('collection')}
            >
              Collections
            </button>
            <button 
              className={`${styles.filterTypeButton} ${filterType === 'tag' ? styles.activeFilterType : ''}`}
              onClick={() => setFilterType('tag')}
            >
              Tags
            </button>
          </div>
          
          <div className={styles.searchBar}>
            <input 
              type="text" 
              placeholder={`Search ${filterType}s...`}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={styles.searchInput}
            />
          </div>
          
          <div className={styles.optionsContainer}>
            {loading ? (
              <div className={styles.loading}>Loading...</div>
            ) : filteredOptions.length > 0 ? (
              <div className={styles.optionsList}>
                {filteredOptions.map(option => (
                  <div 
                    key={`${option.type}-${option.id}`} 
                    className={`${styles.optionItem} ${
                      selectedOptions.some(item => item.id === option.id && item.type === option.type) 
                        ? styles.selectedOption 
                        : ''
                    }`}
                    onClick={() => toggleOption(option)}
                  >
                    <div className={styles.optionContent}>
                      <span className={styles.optionName}>{option.name}</span>
                      {option.metadata && (
                        <div className={styles.optionMetadata}>
                          {option.type === 'file' && option.metadata.pages && (
                            <span className={styles.metadataItem}>{option.metadata.pages} pages</span>
                          )}
                          {option.type === 'file' && option.metadata.fileSize && (
                            <span className={styles.metadataItem}>{formatFileSize(option.metadata.fileSize)}</span>
                          )}
                          {option.metadata.timestamp && (
                            <span className={styles.metadataItem}>{formatDate(option.metadata.timestamp)}</span>
                          )}
                        </div>
                      )}
                    </div>
                    <span className={`${styles.optionBadge} ${getBadgeColor(option.type)}`}>
                      {option.type}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className={styles.noResults}>
                {searchQuery.trim() !== '' 
                  ? `No ${filterType}s matching "${searchQuery}"`
                  : `No ${filterType}s available`}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeFilter; 