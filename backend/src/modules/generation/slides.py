"""Slide generation module for creating presentations.

This module handles the creation of slide decks using LLMs, with support for
semantic search and structured JSON output.
"""

import logging
from typing import Dict, List, Any
import asyncio
import json

from src.modules.config.config import CONFIG
from src.modules.retrieval.search import semantic_search
from src.modules.generation.generate import generate_with_provider

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
        response_text = ""
        async for chunk in response_gen:
            response_text += chunk

        # Clean the response and attempt to extract JSON block
        response_text = response_text.strip()
        logger.debug(f"Raw presentation response received: {response_text[:500]}...") # Log start of raw response

        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                slides_data = json.loads(json_str)
                # Basic validation (optional but recommended)
                if isinstance(slides_data, dict) and "slides" in slides_data and isinstance(slides_data["slides"], list):
                     logger.info("Successfully generated and parsed presentation JSON.")
                     # Add sources (filenames from context chunks) to the response if needed by frontend
                     sources = set()
                     for c in chunks:
                         raw_metadata = c.get('metadata')
                         metadata_dict = {}
                         if isinstance(raw_metadata, str):
                             try:
                                 metadata_dict = json.loads(raw_metadata)
                             except json.JSONDecodeError:
                                 logger.warning(f"[Slides] Failed to parse metadata JSON: {raw_metadata}")
                         elif isinstance(raw_metadata, dict):
                             metadata_dict = raw_metadata
                         
                         filename = metadata_dict.get('filename')
                         if filename:
                             sources.add(filename)
                     slides_data["sources"] = list(sources) # Send unique list of filenames
                     return slides_data
                else:
                    logger.error("Parsed JSON does not match expected presentation structure.")
                    raise ValueError("Parsed JSON structure is invalid.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON from presentation response: {e}")
                logger.debug(f"Extracted JSON string was: {json_str}")
                raise ValueError(f"Invalid presentation format: Failed to parse JSON - {e}")
        else:
            logger.error("Could not find valid JSON block in presentation response.")
            logger.debug(f"Full response was: {response_text}")
            raise ValueError("Invalid presentation format: No JSON block found.")

    except ValueError as ve: # Re-raise specific ValueErrors
        raise ve
    except Exception as e:
        logger.exception(f"Failed to generate presentation: {e}") # Use logger.exception
        raise RuntimeError(f"Failed to generate presentation due to an unexpected error: {e}") 