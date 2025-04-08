"""Test suite for the FastAPI backend - keeping it real with them tests."""

from typing import AsyncGenerator, Dict, List, Any
from fastapi.testclient import TestClient
from frontend.backend.app import app
from vector_store.milvus_ops import connect_milvus, init_collection, store_with_metadata, search_collection, disconnect_milvus
from config.config import CONFIG
from pymilvus import Collection
from embedding.embed import embed_chunks
import pytest
import logging
import asyncio
import numpy as np
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

client = TestClient(app)

# Test data with type hints—4090's precision-tuned! 🔥
TEST_CHUNKS: List[Dict[str, Any]] = [
    {
        'text': 'This is a test document about VibeRAG.',
        'metadata': {'filename': 'test.txt', 'page': '1', 'author': 'test_author'}
    },
    {
        'text': 'VibeRAG is a RAG system that slaps.',
        'metadata': {'filename': 'test.txt', 'page': '2', 'author': 'test_author'}
    }
]

@pytest.fixture(scope="session")
async def setup_milvus():
    """Setup test data in Milvus."""
    print("\n*** Setting up Milvus test data ***")
    
    # Connect to Milvus
    connect_milvus()
    print("Connected to Milvus")
    
    # Get embeddings for test chunks
    print(f"Test chunks: {TEST_CHUNKS}")
    texts = [chunk['text'] for chunk in TEST_CHUNKS]
    embeddings = embed_chunks(texts)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Combine embeddings with test chunks
    chunks_with_embeddings = []
    for chunk, embedding in zip(TEST_CHUNKS, embeddings):
        chunks_with_embeddings.append({
            'text': chunk['text'],
            'metadata': chunk['metadata'],
            'embedding': embedding
        })
    print(f"Combined {len(chunks_with_embeddings)} chunks with embeddings")
    
    # Store test data in Milvus
    print("Storing data in Milvus...")
    collection = init_collection()
    
    # Trying direct storage
    doc_ids = []
    text_list = []
    embedding_list = []
    metadata_list = []
    tags_list = []
    filename_list = []
    category_list = []
    
    for chunk in chunks_with_embeddings:
        doc_ids.append("test_doc")
        text_list.append(chunk['text'])
        embedding_list.append(chunk['embedding'])
        metadata_list.append(json.dumps(chunk['metadata']))
        tags_list.append(['test'])
        filename_list.append('test.txt')
        category_list.append('')
    
    # Insert directly
    insert_data = [
        doc_ids,
        embedding_list,
        text_list,
        metadata_list,
        tags_list,
        filename_list,
        category_list
    ]
    
    print(f"Inserting data: {len(doc_ids)} items")
    collection.insert(insert_data)
    collection.flush()
    print("Data inserted and flushed")
    
    # Verify data was stored
    print("Verifying data storage...")
    results = collection.query(
        expr=f"{CONFIG.milvus.filename_field} == 'test.txt'",
        output_fields=["text", "metadata", "filename"],
        limit=10
    )
    if not results:
        print("WARNING: No test data found in Milvus after setup!")
    else:
        print(f"Found {len(results)} test entries in Milvus")
        print(f"First entry: {results[0]}")
    
    yield
    
    # Cleanup
    print("\n*** Cleaning up Milvus test data ***")
    try:
        collection.delete(f"{CONFIG.milvus.filename_field} == 'test.txt'")
        collection.flush()
        print("Test data removed from Milvus")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint—system check! 🔍"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "4090" in data["message"]

@pytest.mark.asyncio
async def test_chat_streaming():
    """Test streaming chat endpoint—flow check! 🌊"""
    request_data = {
        "query": "What is VibeRAG?",
        "filename": "test.txt",
        "knowledge_only": True,
        "use_web": False,
        "stream": True,
        "tags": None,
        "metadata": None
    }
    
    response = client.post("/chat", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Read streaming response
    chunks = []
    for line in response.iter_lines():
        if line:
            chunk = line.decode() if isinstance(line, bytes) else line
            chunks.append(chunk)
            
    # Verify we got some response
    assert len(chunks) > 0
    assert any("VibeRAG" in chunk for chunk in chunks)

@pytest.mark.asyncio
async def test_chat_with_invalid_request():
    """Test chat endpoint with invalid request—error check! 🚨"""
    request_data = {}  # Missing required 'query' field
    
    response = client.post("/chat", json=request_data)
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_chat_with_nonexistent_file():
    """Test chat with nonexistent file—404 check! 🔍"""
    request_data = {
        "query": "Test query",
        "filename": "nonexistent.txt",
        "stream": True
    }
    
    response = client.post("/chat", json=request_data)
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "File not found" in data["error"]

@pytest.mark.asyncio
async def test_chat_with_metadata():
    """Test chat with metadata filtering."""
    request_data = {
        "query": "What is VibeRAG?",
        "stream": True,
        "metadata": {"author": "test_author"}
    }
    
    response = client.post("/chat", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Read streaming response
    chunks = []
    async for line in response.iter_lines():
        if line:
            if isinstance(line, bytes):
                chunk = line.decode()
                chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert "VibeRAG" in full_response  # Should find content from test chunks

@pytest.mark.asyncio
async def test_chat_with_tags():
    """Test chat with tag filtering."""
    request_data = {
        "query": "What is VibeRAG?",
        "stream": True,
        "tags": ["test", "viberag"]
    }
    
    response = client.post("/chat", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Read streaming response
    chunks = []
    async for line in response.iter_lines():
        if line:
            if isinstance(line, bytes):
                chunk = line.decode()
                chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert "VibeRAG" in full_response  # Should find content from test chunks

@pytest.mark.asyncio
async def test_chat_error_handling():
    """Test chat endpoint error handling."""
    # Test with empty query
    response = client.post(
        "/chat",
        json={
            "query": "",
            "stream": False
        }
    )
    assert response.status_code == 422  # Validation error

    # Test with invalid stream parameter
    response = client.post(
        "/chat",
        json={
            "query": "What is AI?",
            "stream": "invalid"
        }
    )
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_chat_nonexistent_file():
    """Tests /chat with nonexistent file—error handling's on point! 💀"""
    response = client.post(
        "/chat",
        json={
            "query": "What's good?",
            "filename": "nonexistent.pdf",
            "knowledge_only": True,
            "use_web": False
        }
    )
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "File not found" in data["error"]
    logger.info("Nonexistent file test passed—4090's handling it smooth! 🎯")

@pytest.mark.asyncio
async def test_presentation_generation():
    """Test presentation generation endpoint."""
    request_data = {
        "prompt": "Create a presentation about VibeRAG",
        "filename": "test.txt",
        "n_slides": 3
    }
    
    response = client.post("/api/presentation", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "slides" in data
    assert "sources" in data
    assert len(data["slides"]) <= 3
    
    # Check slide content
    for slide in data["slides"]:
        assert "title" in slide
        assert "content" in slide
        assert isinstance(slide["content"], list)
        assert any("VibeRAG" in str(content) for content in slide["content"])

@pytest.mark.asyncio
async def test_research_report():
    """Test research report generation endpoint."""
    request_data = {
        "query": "Explain VibeRAG's capabilities",
        "use_web": True
    }
    
    response = client.post("/research", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    # Check report structure
    assert "report" in data
    report = data["report"]
    assert "title" in report
    assert "summary" in report
    assert "insights" in report
    assert "analysis" in report
    assert "sources" in report
    
    # Check content relevance
    assert "VibeRAG" in report["summary"] or "VibeRAG" in report["analysis"]
    assert len(report["insights"]) > 0
    assert len(report["sources"]) > 0

@pytest.mark.asyncio
async def test_file_operations():
    """Test file upload and delete operations."""
    # Test file upload
    test_content = b"This is a test PDF content"
    files = {"file": ("test.pdf", test_content, "application/pdf")}
    upload_data = {
        "tags": json.dumps(["test", "pdf"]),
        "metadata": json.dumps({"author": "test_user"})
    }
    
    response = client.post("/upload", files=files, data=upload_data)
    assert response.status_code == 200
    upload_result = response.json()
    assert upload_result["filename"] == "test.pdf"
    assert "test" in upload_result["tags"]
    assert upload_result["metadata"]["author"] == "test_user"
    
    # Test file deletion
    response = client.delete(f"/delete/test.pdf")
    assert response.status_code == 200
    delete_result = response.json()
    assert delete_result["success"] is True
    
    # Verify file is gone
    response = client.delete(f"/delete/test.pdf")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_list_documents():
    """Test document listing endpoint."""
    response = client.get("/list")
    assert response.status_code == 200
    docs = response.json()
    
    # Check document structure
    assert isinstance(docs, list)
    for doc in docs:
        assert "doc_id" in doc
        assert "filename" in doc
        assert "tags" in doc
        assert "metadata" in doc

@pytest.mark.asyncio
async def test_get_pdf():
    """Test PDF retrieval endpoint."""
    # Test nonexistent PDF
    response = client.get("/get_pdf/nonexistent.pdf")
    assert response.status_code == 404
    
    # Test existing PDF (assuming test.txt exists from setup)
    response = client.get("/get_pdf/test.txt")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

# Types locked—code's sharp as fuck! 🔥 