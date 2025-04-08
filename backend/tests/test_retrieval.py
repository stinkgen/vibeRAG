"""Unit tests for the retrieval module.

Testing our search capabilities with some dummy data to make sure
everything's working smoother than butter on toast.
"""

import unittest
import numpy as np
from pymilvus import Collection
from typing import List, Dict, Any
from config.config import CONFIG
from vector_store.milvus_ops import (
    connect_milvus,
    init_collection,
    store_with_metadata,
)
from retrieval.search import (
    semantic_search,
    search_by_tag_list,
    search_by_metadata_field,
    hybrid_search
)
from embedding.embed import embed_chunks

class TestRetrieval(unittest.TestCase):
    """Test cases for retrieval functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Connect to Milvus
        connect_milvus()
        
        # Initialize collection with recreate=True to ensure clean state
        cls.collection = init_collection(recreate=True)
        
        # Create test chunks with meaningful content
        cls.test_chunks = [
            {
                'text': 'Deep learning models are revolutionizing AI with neural networks.',
                'metadata': {
                    'filename': 'ai_paper.txt',
                    'page': 1,
                    'category': 'technical',
                    'author': 'Tech Team'
                },
                'tags': ['ai', 'deep_learning', 'performance']
            },
            {
                'text': 'Machine learning algorithms improve performance through training data.',
                'metadata': {
                    'filename': 'ml_guide.txt',
                    'page': 2,
                    'category': 'technical',
                    'author': 'Tech Team'
                },
                'tags': ['machine_learning', 'performance', 'data']
            },
            {
                'text': 'Analytics platforms help businesses scale their operations.',
                'metadata': {
                    'filename': 'business.txt',
                    'page': 1,
                    'category': 'business',
                    'author': 'Business Team'
                },
                'tags': ['analytics', 'scale', 'business']
            }
        ]
        
        # Generate real embeddings using sentence-transformers
        embedded_chunks = embed_chunks(cls.test_chunks)
        
        # Store test chunks in Milvus
        store_with_metadata(
            embedded_chunks,
            tags=['test', 'retrieval'],
            metadata={'test_suite': 'retrieval'}
        )
        
        # Load collection for searching
        cls.collection.load()
    
    def test_semantic_search(self):
        """Test semantic search with filters."""
        # Test basic search
        results = semantic_search("neural networks in AI", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertLessEqual(len(results), 2)
        self.assertTrue(any('neural' in r['text'].lower() for r in results))
        
        # Test search with filename filter
        results = semantic_search("analytics", filename="business.txt")
        self.assertTrue(len(results) > 0)
        self.assertTrue(any('analytics' in r['text'].lower() for r in results))
        
        # Test search with metadata filter
        results = semantic_search(
            "analytics",
            metadata_filter={'category': 'business'}
        )
        self.assertTrue(len(results) > 0)
        self.assertTrue(all(
            r['metadata'].get('category') == 'business'
            for r in results
        ))
    
    def test_tag_search(self):
        """Test search by tags."""
        # Search for single tag
        results = search_by_tag_list(['performance'])
        self.assertTrue(len(results) > 0)
        self.assertTrue(all(
            'performance' in r['tags']
            for r in results
        ))
        
        # Search for multiple tags
        results = search_by_tag_list(['analytics', 'scale'])
        self.assertTrue(len(results) > 0)
        self.assertTrue(all(
            set(['analytics', 'scale']).issubset(set(r['tags']))
            for r in results
        ))
        
        # Search for non-existent tag
        results = search_by_tag_list(['nonexistent'])
        self.assertEqual(len(results), 0)
    
    def test_metadata_search(self):
        """Test search by metadata fields."""
        # Search by category
        results = search_by_metadata_field('category', 'technical')
        self.assertTrue(len(results) > 0)
        self.assertTrue(all(
            r['metadata'].get('category') == 'technical'
            for r in results
        ))
        
        # Search by author
        results = search_by_metadata_field('author', 'Tech Team')
        self.assertTrue(len(results) > 0)
        self.assertTrue(all(
            r['metadata'].get('author') == 'Tech Team'
            for r in results
        ))
        
        # Search by non-existent field
        results = search_by_metadata_field('nonexistent', 'value')
        self.assertEqual(len(results), 0)
    
    def test_hybrid_search(self):
        """Test hybrid search with filters."""
        results = hybrid_search(
            "machine learning performance",
            semantic_weight=0.7,
            keyword_weight=0.3,
            top_k=3
        )
        
        # Check results
        self.assertTrue(len(results) > 0)
        self.assertLessEqual(len(results), 3)
        
        # Check score computation
        for result in results:
            self.assertGreaterEqual(result['score'], 0.0)
            self.assertLessEqual(result['score'], 1.0)
        
        # Results should be sorted by score
        scores = [r['score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Check content relevance
        self.assertTrue(any(
            'machine learning' in r['text'].lower() or
            'performance' in r['text'].lower()
            for r in results
        ))

if __name__ == '__main__':
    unittest.main() 