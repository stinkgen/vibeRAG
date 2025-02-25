"""Research module for generating comprehensive reports.

This module handles the creation of detailed research reports using LLMs,
with support for semantic search and structured JSON output.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any

from config.config import CONFIG
from retrieval.search import semantic_search, google_search
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
        response = ""
        async for chunk in response_gen:
            response += chunk
        
        # Extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                logger.info("Successfully generated research report—4090's got insights! 🧠")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse research report JSON: {str(e)}—4090's confused! 😅")
                raise ValueError(f"Invalid research report format: {str(e)}")
        else:
            logger.error("No JSON found in response—4090's lost! 😕")
            raise ValueError("No valid JSON found in response")
            
    except Exception as e:
        logger.error(f"Failed to generate research report: {str(e)}")
        raise 