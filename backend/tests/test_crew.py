"""Unit tests for the research crew module.

Testing our dream team of research agents to make sure
they're dropping knowledge bombs like pros.
"""

import os
import pytest
from typing import List, Dict, Any
from config.config import CONFIG
from research.research import create_research_report
from vector_store.milvus_ops import connect_milvus, init_collection, store_with_metadata, clean_collection
from embedding.embed import embed_chunks
import numpy as np

class TestResearchCrew:
    """Tests for research crew functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up test data and Milvus collection."""
        connect_milvus()
        init_collection(recreate=True)
        
        # Create and store test chunks with real embeddings
        cls.test_chunks = [
            {
                "text": "AI systems use machine learning algorithms for pattern recognition.",
                "metadata": {"filename": "ai_basics.txt", "page": 1}
            },
            {
                "text": "Deep learning is a subset of machine learning that uses neural networks.",
                "metadata": {"filename": "ai_basics.txt", "page": 2}
            }
        ]
        
        # Generate real embeddings
        embeddings = embed_chunks([chunk["text"] for chunk in cls.test_chunks])
        for chunk, embedding in zip(cls.test_chunks, embeddings):
            chunk["embedding"] = embedding
        
        # Store in Milvus
        store_with_metadata(
            cls.test_chunks,
            tags=["test", "ai"],
            metadata={"filename": "ai_basics.txt"}
        )
    
    @classmethod
    def teardown_class(cls):
        """Clean up test data."""
        clean_collection()
    
    @pytest.mark.asyncio
    async def test_research_report(self):
        """Test research report generation with real operations."""
        result = await create_research_report(
            query="What are the key features of AI systems?",
            use_web=True  # Real web search
        )
        
        assert "report" in result
        assert "title" in result["report"]
        assert "summary" in result["report"]
        assert "insights" in result["report"]
        assert "analysis" in result["report"]
        assert "sources" in result["report"]
        
        assert isinstance(result["report"]["insights"], list)
        assert len(result["report"]["insights"]) > 0
        assert isinstance(result["report"]["sources"], list)
        assert len(result["report"]["sources"]) > 0
        
        # Verify content relevance
        report_text = str(result["report"]).lower()
        assert any(term in report_text for term in ["ai", "machine learning", "deep learning"])

@pytest.mark.asyncio
async def test_research_report_no_results():
    """Test research report generation with no matching results."""
    result = await create_research_report(
        query="xyzabc123 completely irrelevant query",
        use_web=True
    )
    
    assert "report" in result
    assert "title" in result["report"]
    assert "summary" in result["report"]
    assert isinstance(result["report"]["insights"], list)
    assert "analysis" in result["report"]
    assert isinstance(result["report"]["sources"], list)
    
    # Verify fallback behavior
    assert "No relevant information found" in str(result["report"])

@pytest.mark.asyncio
async def test_research_report_web_only():
    """Test research report generation using only web results."""
    result = await create_research_report(
        query="Latest breaking news about artificial intelligence",
        use_web=True,
        use_local=False
    )
    
    assert "report" in result
    assert len(result["report"]["sources"]) > 0
    assert all("http" in source["link"] for source in result["report"]["sources"])

if __name__ == '__main__':
    pytest.main()