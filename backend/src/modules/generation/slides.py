"""Slide generation module for creating presentations.

This module handles the creation of slide decks using LLMs, with support for
semantic search and structured JSON output.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
import re # Import regex module
import uuid

from sqlalchemy.orm import Session # Import Session for type hinting

from src.modules.config.config import CONFIG
from src.modules.auth.database import User, get_db # Import User model and db session
from fastapi import HTTPException # Import HTTPException

# --- Import Agent Service components --- 
from src.modules.agent_service.manager import (
    create_agent_definition, 
    get_agent_definitions_by_user # To find existing default agent
)
from src.modules.agent_service.models import AgentTask, AgentOutput
from src.modules.auth.database import AgentCreate # For creating default agent
# -------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into a context string for the LLM."""
    if not chunks:
        return ""
    return "\n\n".join(str(chunk.get("text", "")) for chunk in chunks if chunk.get("text") is not None)

def _parse_single_slide_json(json_string: str, slide_num: int) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON string containing a single slide object."""
    try:
        # Attempt to find JSON object within potential markdown/text
        match = re.search(r"{\s*\"title\"\s*:", json_string)
        if not match:
            logger.warning(f"Could not find start of JSON object for slide {slide_num}")
            return None
            
        # Try parsing from the found opening brace
        json_start = match.start()
        slide_data = json.loads(json_string[json_start:])
        
        # Basic validation
        if isinstance(slide_data, dict) and "title" in slide_data and "content" in slide_data and isinstance(slide_data["content"], list):
            logger.info(f"Successfully parsed JSON for slide {slide_num}")
            return slide_data
        else:
            logger.warning(f"Parsed JSON for slide {slide_num} has incorrect structure: {slide_data}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error for slide {slide_num}: {e}. Response text: {json_string[:500]}...")
        return None
    except Exception as e:
         logger.error(f"Unexpected error parsing slide {slide_num} JSON: {e}")
         return None

# --- Function to find or create the default presentation agent ---
async def _get_or_create_default_presentation_agent(db: Session, user: User) -> int:
    """Finds the default presentation agent for the user, creates if not exists."""
    agent_name = CONFIG.agent.default_presentation_agent_name
    
    # Check if agent exists
    existing_agents = get_agent_definitions_by_user(db, user_id=user.id, limit=1000)
    found_agent = next((agent for agent in existing_agents if agent.name == agent_name), None)
    
    if found_agent:
        logger.info(f"Found existing default presentation agent (ID: {found_agent.id}) for user {user.username}")
        return found_agent.id
    else:
        # Create the default agent
        logger.info(f"Creating default presentation agent for user {user.username}")
        agent_data = AgentCreate(
            owner_user_id=user.id,
            name=agent_name,
            persona=CONFIG.agent.default_presentation_agent_persona,
            goals=CONFIG.agent.default_presentation_agent_goals,
            is_active=True
        )
        new_agent = create_agent_definition(db, agent=agent_data, owner=user)
        logger.info(f"Created default presentation agent with ID: {new_agent.id}")
        return new_agent.id

# --- Main function refactored to use Agent Service --- 
async def create_presentation(
    prompt: str,
    user: User,
    filename: Optional[str] = None, # Context filename
    n_slides: int = 5,
    model: Optional[str] = None, # Allow overriding model/provider for agent?
    provider: Optional[str] = None
) -> Dict[str, Any]: # Return type includes slides list and sources list
    """Generates a presentation outline by triggering the default presentation agent."""
    log_prefix = f"[Slides {user.username}]"
    logger.info(f"{log_prefix} Starting presentation generation task for: '{prompt[:50]}...'. Target slides: {n_slides}")

    db = next(get_db())
    try:
        # 1. Get Agent ID
        agent_id = await _get_or_create_default_presentation_agent(db, user)

        # 2. Prepare Agent Task
        # Agent persona defines the goal structure. Pass user inputs as part of the goal/input_data.
        # Combine prompt and context info for the agent goal.
        agent_goal = f"Create a {n_slides}-slide presentation outline about: {prompt}."
        input_data = {
            "n_slides": n_slides,
            "prompt": prompt,
        }
        if filename:
            agent_goal += f" Use the document '{filename}' as primary context."
            input_data["context_filename"] = filename # Pass filename for potential tool use

        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_id=agent_id,
            goal=agent_goal,
            input_data=input_data
            # Can potentially pass model/provider overrides here if run_agent_task supports it
        )

        # 3. Run Agent Task - THIS IS NOW HANDLED ASYNCHRONOUSLY
        # This function should now likely just return the agent_id and the prepared goal/input_data
        # The API endpoint will handle the actual Celery task dispatch.
        # For now, raise an error indicating it needs full refactor OR return dummy data.
        # Let's return the info needed by the API endpoint:
        logger.info(f"{log_prefix} Prepared task for agent {agent_id}. Returning info to dispatcher.")
        return {
            "agent_id": agent_id,
            "goal": agent_goal,
            "input_data": input_data,
            "status": "prepared" # Indicate preparation is done
        }

    except Exception as e:
        logger.exception(f"{log_prefix} Unexpected error during presentation generation: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error generating presentation: {str(e)}")
    finally:
        db.close()

# --- Old implementation (keep for reference or remove) ---
async def create_presentation_old(
    prompt: str, 
    user: User, 
    filename: Optional[str] = None, 
    n_slides: int = 5, 
    model: Optional[str] = None, 
    provider: Optional[str] = None
) -> Dict[str, Any]:
    logger.warning("Old create_presentation_old function called (should be refactored).")
    # Return dummy structure to satisfy type hint
    return {"slides": [], "sources": []}
    # pass 