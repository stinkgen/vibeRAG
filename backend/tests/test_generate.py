"""Unit tests for the generation module.

Testing our chat and presentation generation to make sure
we're dropping knowledge bombs like a pro.
"""

import pytest
from typing import List, Dict, Any
from config.config import CONFIG
from generation.generate import chat_with_knowledge, generate_with_provider
from generation.slides import create_presentation
from generation.exceptions import GenerationError

@pytest.mark.asyncio
class TestGeneration:
    """Test cases for generation functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.test_chunks = [
            {
                'text': 'AI and machine learning have revolutionized data processing.',
                'metadata': {'filename': 'test.txt', 'page': 1}
            },
            {
                'text': 'Deep learning models show remarkable capabilities.',
                'metadata': {'filename': 'test.txt', 'page': 2}
            }
        ]
    
    async def test_chat_with_knowledge(self):
        """Test chat with knowledge base."""
        response_gen = chat_with_knowledge(
            "What is AI?",
            filename="test.txt",
            provider="openai",
            model=CONFIG.openai.default_model
        )
        
        # Collect response chunks
        response = ""
        async for chunk in response_gen:
            assert isinstance(chunk, str)
            response += chunk
        
        # Verify response
        assert len(response) > 0
        assert "AI" in response or "artificial intelligence" in response.lower()
    
    async def test_chat_with_knowledge_no_results(self):
        """Test chat with no relevant knowledge."""
        response_gen = chat_with_knowledge(
            "What is quantum computing?",  # Topic not in test chunks
            filename="test.txt",
            provider="openai",
            model=CONFIG.openai.default_model
        )
        
        # Collect response chunks
        response = ""
        async for chunk in response_gen:
            assert isinstance(chunk, str)
            response += chunk
        
        # Verify response indicates no information found
        assert len(response) > 0
        assert "don't have" in response.lower() or "no relevant" in response.lower()
    
    async def test_chat_with_invalid_model(self):
        """Test chat with invalid model name."""
        with pytest.raises(GenerationError):
            response_gen = chat_with_knowledge(
                "What is AI?",
                provider="openai",
                model="nonexistent-model"
            )
            async for _ in response_gen:
                pass
    
    async def test_create_presentation(self):
        """Test presentation generation."""
        result = await create_presentation(
            prompt="Create a presentation about AI",
            n_slides=3
        )
        
        # Check structure
        assert isinstance(result, dict)
        assert 'slides' in result
        assert 'sources' in result
        
        # Check slides
        slides = result['slides']
        assert isinstance(slides, list)
        assert len(slides) <= 3
        
        # Check slide content
        for slide in slides:
            assert 'title' in slide
            assert 'content' in slide
            assert isinstance(slide['content'], list)
            assert all(isinstance(item, str) for item in slide['content'])
            # Verify slide has proper bullet points and visual suggestions
            assert any(point.startswith('• ') for point in slide['content'])
            assert any('Visual:' in point for point in slide['content'])
    
    async def test_generate_with_provider(self):
        """Test provider-specific generation."""
        messages = [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "Write a haiku about recursion in programming."}
        ]
        
        # Test OpenAI provider with different temperatures
        for temp in [0.0, 0.7, 1.0]:
            response_gen = generate_with_provider(
                messages=messages,
                model=CONFIG.openai.default_model,
                provider="openai",
                temperature=temp
            )
            
            # Collect response chunks
            response = ""
            async for chunk in response_gen:
                assert isinstance(chunk, str)
                response += chunk
            
            # Verify response
            assert len(response) > 0
            # Check for haiku-like structure (though not strict syllable count)
            assert len(response.split('\n')) >= 3  # At least 3 lines for haiku
            assert "recursion" in response.lower()
    
    async def test_generate_with_invalid_provider(self):
        """Test generation with invalid provider."""
        messages = [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "Hello"}
        ]
        
        with pytest.raises(ValueError):
            response_gen = generate_with_provider(
                messages=messages,
                model=CONFIG.openai.default_model,
                provider="invalid_provider"
            )
            async for _ in response_gen:
                pass
    
    async def test_chat_with_web_search(self):
        """Test chat with web search integration."""
        response_gen = chat_with_knowledge(
            "What is VibeRAG?",
            use_web=True,
            provider="openai",
            model=CONFIG.openai.default_model
        )
        
        # Collect response chunks
        response = ""
        async for chunk in response_gen:
            assert isinstance(chunk, str)
            response += chunk
        
        # Verify response includes web content
        assert len(response) > 0
        assert any(term in response.lower() for term in ["search", "found", "web"])
    
    async def test_chat_with_different_models(self):
        """Test chat with different model configurations."""
        # Test with different temperatures
        for temp in [0.0, 0.5, 1.0]:
            response_gen = chat_with_knowledge(
                "Write a creative story about AI",
                provider="openai",
                model=CONFIG.openai.default_model,
                temperature=temp
            )
            
            response = ""
            async for chunk in response_gen:
                assert isinstance(chunk, str)
                response += chunk
            
            assert len(response) > 0
            # Higher temps should yield more varied responses
            # but we can't test that directly
    
    async def test_presentation_with_variations(self):
        """Test presentation generation with different configurations."""
        # Test with different slide counts
        for n_slides in [3, 5, 10]:
            result = await create_presentation(
                prompt="Create a presentation about AI",
                n_slides=n_slides
            )
            
            assert isinstance(result, dict)
            assert 'slides' in result
            assert len(result['slides']) <= n_slides
            
            # Check slide structure
            for slide in result['slides']:
                assert 'title' in slide
                assert 'content' in slide
                assert len(slide['content']) >= 2  # At least 2 bullet points
                
                # Check for visual suggestions
                visual_suggestions = [
                    content for content in slide['content']
                    if content.startswith('Visual:')
                ]
                assert len(visual_suggestions) > 0
    
    async def test_presentation_with_file_filter(self):
        """Test presentation generation with specific file filtering."""
        result = await create_presentation(
            prompt="Create a presentation about machine learning",
            filename="test.txt",  # Using our test file
            n_slides=3
        )
        
        assert isinstance(result, dict)
        assert 'slides' in result
        assert 'sources' in result
        
        # Check that sources reference our test file
        assert any('test.txt' in source for source in result['sources'])
        
        # Content should reflect test file topics
        slides_text = str(result['slides']).lower()
        assert any(term in slides_text for term in [
            'machine learning',
            'neural networks',
            'deep learning'
        ]) 