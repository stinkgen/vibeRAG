"""Unit tests for the document ingestion module."""

import os
import tempfile
import unittest
from pathlib import Path
from io import BytesIO

from ingestion.ingest import parse_document, chunk_text, extract_metadata, upload_document, DOCS_DIR
from config.config import CONFIG
from vector_store.milvus_ops import connect_milvus, init_collection

class TestIngestion(unittest.TestCase):
    """Test cases for document ingestion functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests."""
        # Connect to Milvus
        connect_milvus()
        
        # Initialize collection (recreate=True to start fresh)
        init_collection(recreate=True)
        
        # Set up test file path
        cls.test_pdf_path = Path('whitepaper.pdf')
        cls.assertTrue(cls.test_pdf_path.exists(), "whitepaper.pdf must exist for tests")
        
        # Create docs directory if it doesn't exist
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        # Clean up test files in docs directory
        for file in DOCS_DIR.glob('*.pdf'):
            file.unlink(missing_ok=True)
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_text = """
        This is a sample document for testing the ingestion module.
        It contains multiple sentences and some named entities like Google and Microsoft.
        The document is written in English and should be processed correctly.
        We need enough text to test the chunking functionality properly.
        This paragraph should contain sufficient content to demonstrate the chunking
        with overlap capability of our module. We'll make sure it has more than 512 tokens
        by adding some technical terms and descriptions about artificial intelligence,
        machine learning, and natural language processing. These topics should also
        provide good test cases for entity recognition and keyword extraction.
        """
        
        # Create a temporary test file
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_document.txt"
        with open(self.test_file, "w") as f:
            f.write(self.test_text)
            
        # Use the actual whitepaper.pdf for testing
        self.test_pdf_path = Path("whitepaper.pdf")
        self.assertTrue(self.test_pdf_path.exists(), "whitepaper.pdf must exist for tests")
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_upload_document(self):
        """Test document upload with tags and metadata."""
        # Use the actual whitepaper.pdf
        with open(self.test_pdf_path, 'rb') as f:
            test_pdf = BytesIO(f.read())
            test_pdf.name = "test.pdf"
        
        # Test tags and metadata
        tags = ["test", "pdf", "vibeRAG"]
        metadata = {
            "author": "Test Author",
            "category": "Documentation",
            "version": "1.0"
        }
        
        # Upload the document
        result = upload_document(test_pdf, test_pdf.name, tags, metadata)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['filename'], "test.pdf")
        self.assertGreater(result['num_chunks'], 0)
        self.assertEqual(result['tags'], tags)
        self.assertEqual(result['metadata'], metadata)
        self.assertEqual(result['status'], 'success')
        
        # Verify file was saved
        saved_file = Path(DOCS_DIR) / "test.pdf"
        self.assertTrue(saved_file.exists())
        
        # Clean up
        saved_file.unlink()
    
    def test_parse_document(self):
        """Test document parsing and metadata extraction."""
        result = parse_document(self.test_file)
        
        # Check basic structure
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertIn('text', result[0])
        self.assertIn('metadata', result[0])
        
        # Check metadata
        metadata = result[0]['metadata']
        self.assertEqual(metadata['title'], 'test_document')
        self.assertEqual(metadata['language'], 'en')
        self.assertIsInstance(metadata['entities'], dict)
        self.assertIsInstance(metadata['keywords'], list)
        
        # Check entities
        entities = metadata['entities']
        org_entities = [ent for label in entities.values() for ent in label]
        self.assertTrue(any('Google' in ent for ent in org_entities))
        self.assertTrue(any('Microsoft' in ent for ent in org_entities))
    
    def test_chunk_text(self):
        """Tests chunk_text doesn't shit the bed—config's got this! 🚀"""
        # Create a fat text to chunk
        text = " ".join(["word"] * 1000)  # Plenty to split
        
        # Let it rip
        chunks = chunk_text(text)
        
        # Basic checks
        self.assertTrue(len(chunks) > 1, "Chunking's fucked—shoulda split this shit!")
        
        # Size checks
        for chunk in chunks:
            tokens = chunk.split()  # Simple tokenization for test
            self.assertLessEqual(
                len(tokens), 
                CONFIG.ingestion.chunk_size,
                f"Chunk's too fat ({len(tokens)} > {CONFIG.ingestion.chunk_size})—config's off!"
            )
        
        # Overlap checks
        if len(chunks) > 1:
            chunk1_tokens = chunks[0].split()
            chunk2_tokens = chunks[1].split()
            overlap_found = any(
                token in chunk2_tokens[-CONFIG.ingestion.overlap:]
                for token in chunk1_tokens[:CONFIG.ingestion.overlap]
            )
            self.assertTrue(overlap_found, "No overlap? Config's crying!")
        
        # Content checks
        for chunk in chunks:
            self.assertIn("word", chunk, "Words got lost—lame-ass bug!")
            self.assertTrue(len(chunk.strip()) > 0, "Empty chunk? Weak sauce!")
    
    def test_extract_metadata(self):
        """Test metadata extraction functionality."""
        metadata = extract_metadata(self.test_text)
        
        # Check metadata structure
        self.assertIn('language', metadata)
        self.assertIn('entities', metadata)
        self.assertIn('keywords', metadata)
        
        # Check language detection
        self.assertEqual(metadata['language'], 'en')
        
        # Check keywords
        self.assertTrue(len(metadata['keywords']) > 0)
        self.assertTrue(any('intelligence' in keyword.lower() for keyword in metadata['keywords']))
    
    def test_upload_with_empty_metadata(self):
        """Test document upload with empty metadata."""
        # Use the actual whitepaper.pdf
        with open(self.test_pdf_path, 'rb') as f:
            test_pdf = BytesIO(f.read())
            test_pdf.name = "test_empty.pdf"
        
        # Upload with minimal info
        result = upload_document(test_pdf, test_pdf.name)
        
        # Check defaults
        self.assertEqual(result['tags'], [])
        self.assertEqual(result['metadata'], {})
        self.assertEqual(result['status'], 'success')
        
        # Clean up
        saved_file = Path(DOCS_DIR) / "test_empty.pdf"
        saved_file.unlink()
    
    def test_upload_invalid_file(self):
        """Test upload with invalid file."""
        # Create an invalid file
        invalid_file = BytesIO(b"Not a valid PDF")
        invalid_file.name = "invalid.pdf"
        
        # Should raise an exception
        with self.assertRaises(Exception):
            upload_document(invalid_file, invalid_file.name)

if __name__ == '__main__':
    unittest.main() 