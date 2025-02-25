"""Unit tests for the search module.

Testing our search capabilities with some dummy data to make sure
everything's working smoother than butter on toast.
"""

import unittest
import numpy as np
from pymilvus import Collection
from vector_store.milvus_ops import connect_milvus, init_collection, store_with_metadata
from retrieval.search import semantic_search, keyword_search, hybrid_search
from config.config import CONFIG  # Config's in the house! 🏠
import uuid

class TestSearch(unittest.TestCase):
    """Test cases for search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Initialize collection with recreate=True to ensure clean state
        init_collection(recreate=True)
        
        # Create test chunks with meaningful content
        cls.test_chunks = []
        for i, content in enumerate([
            "Deep learning models are revolutionizing AI with neural networks.",
            "Machine learning algorithms improve performance through training data.",
            "Natural language processing enables human-computer interaction."
        ]):
            embedding = np.random.rand(CONFIG.embedding.embedding_dim).astype(np.float32)
            chunk = {
                'chunk_id': str(uuid.uuid4()),
                'doc_id': f'test_doc_{i}',
                'embedding': embedding,
                'text': content,
                'metadata': {'filename': f'test_{i}.txt', 'page': i, 'category': 'technical'},
                'tags': ['ai', 'machine_learning']
            }
            cls.test_chunks.append(chunk)
        
        # Store test chunks in Milvus
        store_with_metadata(
            cls.test_chunks,
            tags=['test', 'search'],
            metadata={'source': 'test_search'}
        )
        
        # Load collection for searching
        cls.collection = Collection(CONFIG.milvus.collection_name)
        cls.collection.load()
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        results = semantic_search("machine learning models", top_k=2)
        
        # Check results structure and content
        self.assertTrue(len(results) > 0)
        self.assertLessEqual(len(results), 2)
        self.assertIn('text', results[0])
        self.assertIn('metadata', results[0])
        self.assertIn('score', results[0])
        
        # Scores should be between 0 and 1
        for result in results:
            self.assertGreaterEqual(result['score'], 0.0)
            self.assertLessEqual(result['score'], 1.0)
    
    def test_keyword_search(self):
        """Test keyword search functionality."""
        results = keyword_search("neural", top_k=2)
        
        # Check results
        self.assertTrue(len(results) > 0)
        self.assertLessEqual(len(results), 2)
        
        # All results should contain the keyword
        for result in results:
            self.assertIn('neural', result['text'].lower())
    
    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        results = hybrid_search(
            "deep learning neural networks",
            semantic_weight=0.6,
            keyword_weight=0.4,
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

if __name__ == '__main__':
    unittest.main() 

# Magic strings smoked—code's pure fire! 🔥 