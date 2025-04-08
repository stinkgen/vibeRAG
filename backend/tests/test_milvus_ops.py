"""Unit tests for Milvus vector store operations."""

import unittest
import numpy as np
import uuid
from vector_store.milvus_ops import (
    connect_milvus,
    init_collection,
    store_with_metadata,
    delete_document,
    clean_collection
)
from config.config import CONFIG  # Config's in the house! 🏠
from pymilvus import Collection

class TestMilvusOps(unittest.TestCase):
    """Test cases for Milvus operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Connect to Milvus first
        connect_milvus()
        
        # Initialize collection with recreate=True to ensure clean state
        init_collection(recreate=True)
        
        # Create test chunks with random embeddings
        cls.test_chunks = []
        for i in range(3):
            embedding = np.random.rand(CONFIG.milvus.embedding_dim).astype(np.float32)
            chunk = {
                'embedding': embedding,
                'text': f'Test chunk {i} about AI technology.',
                'metadata': {'page': i},
                'tags': ['test', 'ai']
            }
            cls.test_chunks.append(chunk)
        
        # Get collection
        cls.collection = Collection(CONFIG.milvus.collection_name)
        cls.collection.load()
    
    def test_store_and_delete(self):
        """Test storing chunks and deleting them."""
        # Store test chunks
        result = store_with_metadata(
            self.test_chunks,
            tags=['test'],
            metadata={'filename': 'test.txt'}
        )
        
        # Verify storage
        self.assertTrue(len(result) > 0)
        
        # Delete document by filename
        success = delete_document('test.txt')
        self.assertTrue(success)
        
        # Verify deletion
        results = self.collection.query(
            expr=f'{CONFIG.milvus.filename_field} == "test.txt"',
            output_fields=[CONFIG.milvus.chunk_id_field],
            limit=1
        )
        self.assertEqual(len(results), 0)
    
    def test_clean_collection(self):
        """Test collection cleanup functionality."""
        # Store some test data
        store_with_metadata(
            self.test_chunks,
            tags=['test'],
            metadata={'filename': 'test.txt'}
        )
        
        # Verify data exists
        results = self.collection.query(
            expr='',
            output_fields=[CONFIG.milvus.chunk_id_field],
            limit=1
        )
        self.assertTrue(len(results) > 0)
        
        # Clean collection
        success = clean_collection()
        self.assertTrue(success)
        
        # Verify collection is empty
        results = self.collection.query(
            expr='',
            output_fields=[CONFIG.milvus.chunk_id_field],
            limit=1
        )
        self.assertEqual(len(results), 0)
    
    def test_delete_nonexistent_document(self):
        """Test deleting a document that doesn't exist."""
        success = delete_document("nonexistent_doc")
        self.assertTrue(success)  # Should return True even if doc doesn't exist
    
    def test_clean_nonexistent_collection(self):
        """Test cleaning when collection doesn't exist."""
        # Drop collection if it exists
        if self.collection:
            self.collection.drop()
        
        # Try to clean non-existent collection
        success = clean_collection()
        self.assertFalse(success)  # Should return False when no collection exists

if __name__ == '__main__':
    unittest.main() 

# Magic strings smoked—code's pure fire! 🔥 