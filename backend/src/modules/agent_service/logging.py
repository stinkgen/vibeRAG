"""Logging utilities for the Agent Service."""

import logging
import json
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from src.modules.auth.database import AgentLog # Import the DB model

logger = logging.getLogger(__name__)


def log_agent_activity(
    db: Session,
    agent_id: int,
    level: str, # e.g., INFO, PLAN, ACTION, TOOL_CALL, ERROR
    message: str,
    details: Optional[Dict[str, Any]] = None,
    # task_id: Optional[str] = None # Add if task_id is implemented in model
):
    """Logs a structured message about agent activity to the database."""
    
    details_str = None
    if details:
        try:
            details_str = json.dumps(details)
        except TypeError as e:
            logger.error(f"Failed to serialize agent log details for agent {agent_id}: {e}")
            details_str = json.dumps({"error": "Serialization failed", "original_details": str(details)})
            
    log_entry = AgentLog(
        agent_id=agent_id,
        # task_id=task_id,
        level=level.upper(),
        message=message,
        details=details_str
    )
    
    try:
        db.add(log_entry)
        db.commit()
        # db.refresh(log_entry) # Probably not needed unless we use the ID immediately
    except Exception as e:
        logger.error(f"Failed to write agent log to database for agent {agent_id}: {e}")
        db.rollback() # Rollback the session in case of DB error

# Example usage within agent runtime:
# from .logging import log_agent_activity
# from src.modules.auth.database import get_db
# db = next(get_db())
# log_agent_activity(db, agent_id=1, level="PLAN", message="Generated initial plan.", details={"plan_steps": ["step1", "step2"]})
# log_agent_activity(db, agent_id=1, level="ERROR", message="Tool execution failed.", details={"tool_name": "web_search", "error": "API timeout"}) 