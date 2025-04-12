"""Slide generation module for creating presentations.

This module handles the creation of slide decks using LLMs, with support for
semantic search and structured JSON output.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
import re # Import regex module

from src.modules.config.config import CONFIG
from src.modules.retrieval.search import semantic_search
from src.modules.generation.generate import generate_with_provider, GenerationError # Import GenerationError
from src.modules.auth.database import User # Import User model
from fastapi import HTTPException # Import HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into a context string for the LLM."""
    # Increased robustness: handle non-string text, filter None
    return "\n\n".join(str(chunk.get("text", "")) for chunk in chunks if chunk.get("text") is not None)

def _parse_single_slide_json(response_text: str, slide_num: int) -> Optional[Dict[str, Any]]:
    """Attempts to parse a single slide JSON object from the LLM response."""
    response_text = response_text.strip()
    logger.debug(f"Attempting to parse JSON for slide {slide_num}. Raw response: {response_text[:500]}...")
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            slide_data_wrapper = json.loads(json_str)
            # Expecting structure like {"slide": {"title": ..., "content": ...}}
            if isinstance(slide_data_wrapper, dict) and "slide" in slide_data_wrapper and isinstance(slide_data_wrapper["slide"], dict):
                slide_content = slide_data_wrapper["slide"]
                if "title" in slide_content and "content" in slide_content:
                     logger.info(f"Successfully parsed JSON for slide {slide_num}.")
                     return slide_content # Return the inner slide object
                else:
                     logger.error(f"Parsed JSON for slide {slide_num} lacks title or content.")
            else:
                logger.error(f"Parsed JSON for slide {slide_num} does not match expected structure {{\"slide\": ...}}.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for slide {slide_num}: {e}")
            logger.debug(f"Invalid JSON string was: {json_str}")
    else:
        logger.error(f"Could not find valid JSON block for slide {slide_num} using regex.")
        logger.debug(f"Full response for slide {slide_num} was: {response_text}")
    return None # Return None if parsing fails

async def create_presentation(
    prompt: str,
    user: User,
    filename: Optional[str] = None,
    n_slides: int = 5,
    model: Optional[str] = None,
    provider: Optional[str] = None
) -> Dict[str, Any]: # Return type includes slides list and sources list
    """Create a presentation iteratively, one slide at a time."""
    log_prefix = f"[Slides {user.username}]"
    logger.info(f"{log_prefix} Starting presentation generation for: '{prompt}'. Target slides: {n_slides}")

    # Determine model and provider, defaulting to presentation config
    model_to_use = model or CONFIG.presentation.model
    provider_to_use = provider or CONFIG.presentation.provider
    logger.info(f"{log_prefix} Using {provider_to_use}/{model_to_use}")

    # 1. Gather Context
    filters = None
    if filename:
        filters = [{'type': 'filename', 'value': filename}]
        logger.info(f"{log_prefix} Filtering context by filename: {filename}")

    try:
        chunks = await semantic_search(
            query=prompt,
            user=user,
            limit=CONFIG.presentation.chunks_limit,
            filters=filters
        )
    except Exception as search_err:
         logger.error(f"{log_prefix} Semantic search failed during presentation generation: {search_err}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Knowledge base search failed: {search_err}")

    context = format_context(chunks)
    final_sources = []
    if chunks:
        final_sources = list(set(c.get('metadata', {}).get('filename') for c in chunks if c.get('metadata', {}).get('filename')))
        logger.info(f"{log_prefix} Retrieved {len(chunks)} context chunks. Sources: {final_sources}")
    else:
        logger.warning(f"{log_prefix} No relevant context chunks found.")
        # Don't necessarily fail, LLM might generate from prompt alone, but quality may suffer

    # 2. Generate Slides Iteratively
    generated_slides = []
    previous_slide_title = None
    generation_errors = 0

    for i in range(n_slides):
        slide_num = i + 1
        logger.info(f"{log_prefix} Generating slide {slide_num}/{n_slides}...")

        # Define the single-slide prompt
        system_prompt = (
            "You are an expert presentation creator generating ONE slide JSON object at a time for a larger presentation. "
            "Follow the requested JSON structure precisely. Synthesize provided context for insights. "
            "Provide concrete visual and design ideas."
        )
        user_prompt = (
            f"Context:\n---\n{context if context else 'No specific context provided.'}\n---\n\n"
            f"Overall Presentation Topic: {prompt}\n"
            f"This is Slide {slide_num} of {n_slides}.\n"
            f"The previous slide title was: {previous_slide_title if previous_slide_title else 'None (This is the first slide)'}.\n\n"
            f"Generate the JSON for *only* Slide {slide_num}, ensuring the title is concise and content/visuals are relevant to this specific slide's place in the overall flow. "
            f"Format the output as *only* a raw JSON object matching this structure:\n"
            "```json\n"
            "{\n"
            "  \"slide\": {\n"
            "    \"title\": \"Short Title for Slide {slide_num}\",\n"
            "    \"content\": [\n"
            "      \"• Bullet point 1 for slide {slide_num}\",\n"
            "      \"• Bullet point 2 for slide {slide_num}\",\n"
            "      \"Visual: Specific visual for slide {slide_num}\",\n"
            "      \"Design: Specific design note for slide {slide_num}\"\n"
            "    ]\n"
            "  }\n"
            "}\n"
            "```"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text = ""
        try:
            response_gen = generate_with_provider(
                messages=messages,
                model=model_to_use,
                provider=provider_to_use,
                temperature=CONFIG.presentation.temperature
            )
            async for chunk in response_gen:
                response_text += chunk

            # Parse the response for this slide
            parsed_slide = _parse_single_slide_json(response_text, slide_num)

            if parsed_slide:
                generated_slides.append(parsed_slide)
                previous_slide_title = parsed_slide.get("title", "Untitled")
            else:
                # Failed to parse - add placeholder or stop?
                logger.error(f"{log_prefix} Failed to parse slide {slide_num}. Adding placeholder.")
                generated_slides.append({
                    "title": f"Slide {slide_num} Generation Error",
                    "content": ["• Failed to generate or parse content for this slide."]
                })
                previous_slide_title = f"Slide {slide_num} Error"
                generation_errors += 1

        except GenerationError as e:
            logger.error(f"{log_prefix} LLM Generation failed for slide {slide_num}: {e}", exc_info=True)
            generated_slides.append({
                "title": f"Slide {slide_num} Generation Error",
                "content": [f"• LLM Error: {e}"]
            })
            previous_slide_title = f"Slide {slide_num} Error"
            generation_errors += 1
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error generating slide {slide_num}: {e}")
            generated_slides.append({
                "title": f"Slide {slide_num} Unexpected Error",
                "content": [f"• Error: {e}"]
            })
            previous_slide_title = f"Slide {slide_num} Error"
            generation_errors += 1
            # Optionally break the loop on unexpected errors
            # break

    # 3. Return Final Structure
    if generation_errors > 0:
         logger.warning(f"{log_prefix} Completed presentation generation with {generation_errors} slide errors.")
    else:
         logger.info(f"{log_prefix} Successfully generated all {n_slides} slides.")

    return {
        "slides": generated_slides,
        "sources": final_sources
    } 