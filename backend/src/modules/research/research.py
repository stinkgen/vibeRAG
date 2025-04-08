"""Research module for generating comprehensive reports.

This module handles the creation of detailed research reports using LLMs,
with support for semantic search and structured JSON output.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any

from src.modules.config.config import CONFIG
from src.modules.retrieval.search import semantic_search, google_search
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

async def create_research_report(
    query: str,
    use_web: bool = True
) -> Dict[str, Any]:
    """Create a research report from chunks and web results.
    
    Args:
        query: Research query
        use_web: Whether to include web results
    
    Returns:
        Research report with title, summary, insights, analysis, and sources
    """
    try:
        # Get relevant chunks from vector store
        chunks = semantic_search(
            query=query,
            limit=CONFIG.research.chunks_limit
        )
        
        # Get web results if requested
        web_results = []
        if use_web:
            web_results = google_search(
                query=query,
                limit=CONFIG.web_search.limit
            )
        
        # Prepare prompt with chunks and web results
        prompt = f"Research query: {query}\n\n"
        
        if chunks:
            prompt += "Relevant chunks:\n"
            for i, chunk in enumerate(chunks, 1):
                filename = chunk['metadata'].get('filename', 'unknown file')
                page = chunk['metadata'].get('page', '?')
                prompt += f"{i}. From {filename} (page {page}):\n{chunk['text']}\n\n"
        
        if web_results:
            prompt += "Web results:\n"
            for i, result in enumerate(web_results, 1):
                prompt += f"{i}. {result['title']}\n{result['snippet']}\n\n"
        
        prompt += """Based on the above information, generate a research report in the following JSON format:
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
}"""
        
        # Generate report with LLM
        response_gen = generate_with_provider(
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": prompt}
            ],
            model=CONFIG.research.model,
            provider=CONFIG.research.provider,
            temperature=CONFIG.research.temperature
        )
        
        # Collect all chunks into a single response
        response_text = ""
        async for chunk in response_gen:
            response_text += chunk

        # Clean the response and attempt to extract JSON block
        response_text = response_text.strip()
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
                    # Add detailed sources if needed (currently handled in prompt)
                    # report_data["report"]["detailed_sources"] = ... 
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
            