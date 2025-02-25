"""Slide generation module for creating presentations.

This module handles the creation of slide decks using LLMs, with support for
semantic search and structured JSON output.
"""

import logging
from typing import Dict, List, Any
import asyncio
import json

from config.config import CONFIG
from retrieval.search import semantic_search
from generation.generate import generate_with_provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into a context string for the LLM."""
    return "\n\n".join(chunk["text"] for chunk in chunks)

async def create_presentation(
    prompt: str,
    filename: str = None,
    n_slides: int = 5,
    model: str = None,
    provider: str = None
) -> Dict[str, List[Dict[str, str]]]:
    """Create a presentation from your docs using direct chat generation.
    
    Args:
        prompt: What the presentation should be about
        filename: Optional filename to filter chunks
        n_slides: Number of slides to generate
        model: Optional model to use (defaults to CONFIG.presentation.model)
        provider: Optional provider to use (defaults to CONFIG.presentation.provider)
    
    Returns:
        JSON slide deck with titles, content, and sources
    """
    logger.info("Gathering knowledge for presentation creation... 📚")
    chunks = semantic_search(prompt, filename=filename, limit=CONFIG.presentation.chunks_limit)
    
    if not chunks:
        logger.warning("No relevant content found, creating minimal presentation 💪")
        return {
            "slides": [{
                "title": "Overview",
                "content": ["• No specific information found"],
            }],
            "sources": []
        }

    # Format context and build messages
    context = format_context(chunks)
    messages = [
        {
            "role": "system",
            "content": """You are an expert presentation creator. Your task is to create clear,
            structured presentations in a specific JSON format. Follow these rules exactly:
            1. Each slide must have a title and content as an array of strings
            2. Keep titles short and impactful (3-5 words)
            3. Format each main point with a bullet point prefix "• "
            4. Include 2-3 bullet points per slide
            5. Add visual and design suggestions prefixed with "Visual:" and "Design:"
            6. Return ONLY valid JSON - no other text or prefixes/suffixes
            7. Never include empty strings in the content array"""
        },
        {
            "role": "user",
            "content": f"""Based on this context:
{context}

Create a {n_slides}-slide presentation about: {prompt}

You MUST return your response in this EXACT format, with bullet points for main content:
{{
    "slides": [
        {{
            "title": "Short Impactful Title",
            "content": [
                "• First key point about the topic",
                "• Second important point to consider",
                "Visual: A clear visual suggestion for the slide",
                "Design: A specific design note for layout and style"
            ]
        }}
    ]
}}"""
        }
    ]

    # Use provided model/provider or fall back to config defaults
    model_to_use = model or CONFIG.presentation.model
    provider_to_use = provider or CONFIG.presentation.provider
    logger.info(f"Using {provider_to_use}/{model_to_use} to generate presentation")
    
    # Generate presentation using config values
    logger.info("Generating presentation content... 🎨")
    try:
        response_gen = generate_with_provider(
            messages=messages,
            model=model_to_use,
            provider=provider_to_use,
            temperature=CONFIG.presentation.temperature
        )
        
        # Collect all chunks into a single response
        response = ""
        async for chunk in response_gen:
            response += chunk
        
        # Clean the response to help with JSON parsing
        response = response.strip()
        
        # Parse JSON response
        try:
            slides = json.loads(response)
            logger.info("Successfully generated presentation—4090's got style! 🎨")
            return slides
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse presentation JSON: {str(e)}—4090's confused! 😅")
            raise ValueError(f"Invalid presentation format: {str(e)}")
            
    except Exception as e:
        logger.error(f"Failed to generate presentation: {str(e)}")
        raise 