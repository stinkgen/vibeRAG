"""Research module for generating comprehensive reports.

This module handles the creation of detailed research reports using LLMs,
with support for semantic search and structured JSON output.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from fastapi import HTTPException

from src.modules.config.config import CONFIG
from src.modules.retrieval.search import semantic_search, google_search
from src.modules.generation.generate import generate_with_provider, GenerationError
from src.modules.auth.database import User # Import User model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into a context string for the LLM."""
    return "\n\n".join(chunk["text"] for chunk in chunks)

async def create_research_report(
    query: str,
    user: User, # Added user object
    use_web: bool = True,
    use_knowledge: bool = True,
    model: Optional[str] = None, # Add optional model arg
    provider: Optional[str] = None # Add optional provider arg
) -> Dict[str, Any]:
    """Create a research report from chunks and web results for a specific user.
    
    Args:
        query: Research query
        user: The authenticated user object.
        use_web: Whether to include web results
        use_knowledge: Whether to use internal knowledge base
        model: Optional model to use for generation
        provider: Optional provider to use for generation
    
    Returns:
        Research report with title, summary, insights, analysis, and sources
    """
    try:
        chunks = []
        context = ""
        sources = [] # Initialize sources list
        if use_knowledge:
            logger.info(f"[Research] Performing semantic search for user {user.username}, query: '{query[:50]}...'")
            # Pass user object to semantic_search
            retrieved_chunks = await semantic_search(
                query=query,
                user=user, # Pass user object
                limit=CONFIG.research.chunks_limit
                # Filters could be added here if Research UI supports them
            )
            if retrieved_chunks:
                chunks = retrieved_chunks # Assign if found
                context = format_context(chunks)
                # Extract filenames safely
                sources = list(set(c.get('metadata', {}).get('filename') for c in chunks if c.get('metadata', {}).get('filename')))
                logger.info(f"[Research] Found {len(chunks)} relevant chunks. Sources: {sources}")
            else:
                logger.info(f"[Research] No relevant chunks found from semantic search.")
        else:
            logger.info(f"[Research] Skipping semantic search as use_knowledge=False.")

        # Get web results if requested
        web_results = []
        if use_web:
            web_results = google_search(
                query=query,
                limit=CONFIG.web_search.limit
            )
        
        # Prepare prompt with chunks and web results
        prompt = f"Research query: {query}\n\n"
        
        if context: # Check if context was actually generated
            prompt += f"Relevant context from knowledge base:\n{context}\n\n"
        
        if chunks:
            prompt += "Relevant chunks:\n"
            for i, chunk in enumerate(chunks, 1):
                raw_metadata = chunk.get('metadata')
                metadata_dict = {}
                if isinstance(raw_metadata, str):
                    try:
                        metadata_dict = json.loads(raw_metadata)
                    except json.JSONDecodeError:
                        logger.warning(f"[Research] Failed to parse metadata JSON: {raw_metadata}")
                elif isinstance(raw_metadata, dict):
                    metadata_dict = raw_metadata
                filename = metadata_dict.get('filename', 'unknown file')
                page = metadata_dict.get('page', '?')
                prompt += f"{i}. From {filename} (page {page}):\n{chunk['text']}\n\n"
        
        if web_results:
            prompt += "Web results:\n"
            for i, result in enumerate(web_results, 1):
                prompt += f"{i}. {result['title']}\n{result['snippet']}\n\n"
        
        prompt += """Based on the above information, generate a research report in the following JSON format.
**You MUST return ONLY the raw JSON object, starting with { and ending with }, with no other text, explanations, or markdown formatting outside the JSON structure.**

```json
{
    "report": {
        "title": "A descriptive title for the research",
        "summary": "A concise summary of the findings",
        "insights": [
            "Key insight 1",
            "Key insight 2",
            "Key insight 3"
        ],
        "analysis": "A detailed analysis of the findings",
        "sources": [
            "Source 1",
            "Source 2",
            "Source 3"
        ]
    }
}
```"""
        
        # Determine model and provider to use
        actual_model = model or CONFIG.research.model
        actual_provider = provider or CONFIG.research.provider
        logger.info(f"Using {actual_provider}/{actual_model} to generate research report")

        # Generate report with LLM
        logger.info("Generating research report content...")
        report_content = ""
        try:
            async for chunk in generate_with_provider(
                messages=[
                    {"role": "system", "content": "You are a research assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=actual_model, # Pass determined model
                provider=actual_provider, # Pass determined provider
                temperature=CONFIG.research.temperature
            ):
                report_content += chunk
        except GenerationError as e:
            logger.error(f"Failed to generate research report content: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")
        except Exception as e:
            # Catch unexpected errors during generation stream
            logger.error(f"Unexpected error during research report generation stream: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected error during generation: {e}")

        if not report_content:
            logger.warning("LLM generated empty report content.")
            # Return a default structure or raise error?
            raise HTTPException(status_code=500, detail="LLM generated an empty report.")

        # Clean the response and attempt to extract JSON block
        response_text = report_content.strip()
        logger.debug(f"Raw research response received: {response_text[:500]}...")

        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                report_data = json.loads(json_str)
                # Basic validation (optional)
                if isinstance(report_data, dict) and "report" in report_data and isinstance(report_data["report"], dict):
                    logger.info("Successfully generated and parsed research report JSON.")
                    # Ensure the 'sources' field exists in the report before assigning
                    if "report" in report_data and isinstance(report_data["report"], dict):
                        # Combine KB sources (if use_knowledge was true) with LLM-generated sources (if any)
                        # For now, overwrite with KB sources if KB was used, otherwise keep LLM sources
                        if use_knowledge:
                            report_data["report"]["sources"] = sources # Overwrite with KB sources
                        # If not using knowledge, trust LLM to list web sources etc.
                        elif "sources" not in report_data["report"]:
                            report_data["report"]["sources"] = [] # Ensure key exists
                    else:
                         # Handle case where report structure is missing
                         if use_knowledge:
                             report_data["sources"] = sources
                         elif "sources" not in report_data:
                             report_data["sources"] = []
                        
                    return report_data
                else:
                    logger.error("Parsed JSON does not match expected report structure.")
                    raise ValueError("Parsed JSON structure is invalid.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON from research response: {e}")
                logger.debug(f"Extracted JSON string was: {json_str}")
                raise ValueError(f"Invalid research report format: Failed to parse JSON - {e}")
        else:
            logger.error("Could not find valid JSON block in research response.")
            logger.debug(f"Full response was: {response_text}")
            raise ValueError("Invalid research report format: No JSON block found.")

    except ValueError as ve: # Re-raise specific ValueErrors
        raise ve
    except Exception as e:
        logger.exception(f"Failed to generate research report: {e}") # Use logger.exception
        raise RuntimeError(f"Failed to generate research report due to an unexpected error: {e}")
            