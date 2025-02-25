"""Unit tests for the text embedding module—4090 edition! 🔥"""

import unittest
import numpy as np
from sentence_transformers import SentenceTransformer
from embedding.embed import embed_chunks
from config.config import CONFIG

# Testing this shit like a pro
class TestEmbedding(unittest.TestCase):
    """Test cases for embedding functionality—keeping it tight! 🚀"""
    
    def setUp(self):
        """Set up test fixtures—one chunk, no bullshit."""
        # Create a simple test chunk
        self.test_chunks = [{
            'text': 'Yo, test this shit',
            'metadata': {'source': 'test.txt', 'language': 'en'}
        }]
        
        # Get expected embedding dimension from the model
        self.model = SentenceTransformer(CONFIG.embedding.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def test_embed_chunks(self):
        """Tests embed_chunks doesn't shit the bed—simple and clean."""
        # Embed chunks
        result = embed_chunks(self.test_chunks)
        
        # Basic checks
        self.assertEqual(len(result), 1, "Embedding's fucked—shoulda got one result")
        self.assertIn('embedding', result[0], "No vector in result? Lame as hell")
        self.assertIn('text', result[0], "Text got lost—wtf?")
        self.assertIn('metadata', result[0], "Metadata bounced—not cool")
        
        # Check embedding is legit
        embedding = result[0]['embedding']
        self.assertIsInstance(embedding, np.ndarray, "Embedding ain't numpy? You trippin'")
        self.assertEqual(embedding.shape, (self.embedding_dim,), f"Wrong dimensions—expected {self.embedding_dim}")
        self.assertGreater(np.linalg.norm(embedding), 0, "Got zero vector—4090's crying rn")
        
        # Check text preserved
        self.assertEqual(result[0]['text'], self.test_chunks[0]['text'], "Text changed—why you mutating?")
        self.assertEqual(result[0]['metadata'], self.test_chunks[0]['metadata'], "Metadata changed—keep it pure!")

if __name__ == '__main__':
    unittest.main() 