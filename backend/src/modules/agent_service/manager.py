"""Placeholder for Agent Management logic.

This module will handle CRUD operations for agent definitions,
loading agent configurations, and potentially managing agent lifecycles
(if not handled directly by the runtime or an orchestrator).
"""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import desc # For ordering logs
from datetime import datetime # For time filtering

from src.modules.auth.database import Agent, AgentCapability, AgentLog, AgentCreate, AgentUpdate, User, get_db, AgentTask
# Import AgentResponse if needed from auth.database or agent_service.models
from src.modules.auth.database import AgentResponse

logger = logging.getLogger(__name__)

# --- Agent Definition CRUD --- 

def create_agent_definition(db: Session, agent: AgentCreate, owner: User) -> Agent:
    """Creates a new agent definition in the database."""
    # Basic validation or default setting could happen here
    db_agent = Agent(
        **agent.model_dump(exclude_unset=True), # Use Pydantic v2 model_dump
        owner_user_id=owner.id # Ensure owner is set correctly
    )
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    logger.info(f"Created agent definition '{db_agent.name}' (ID: {db_agent.id}) for user {owner.username}.")
    return db_agent

def get_agent_definition(db: Session, agent_id: int, user_id: int) -> Optional[Agent]:
    """Gets a specific agent definition by ID, ensuring user ownership."""
    return db.query(Agent).filter(Agent.id == agent_id, Agent.owner_user_id == user_id).first()

def get_agent_definitions_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Agent]:
    """Gets all agent definitions owned by a specific user."""
    return db.query(Agent).filter(Agent.owner_user_id == user_id).offset(skip).limit(limit).all()

def update_agent_definition(db: Session, agent_id: int, agent_update: AgentUpdate, user_id: int) -> Optional[Agent]:
    """Updates an agent definition, ensuring user ownership."""
    db_agent = get_agent_definition(db, agent_id, user_id)
    if not db_agent:
        return None
    
    update_data = agent_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_agent, key, value)
        
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    logger.info(f"Updated agent definition '{db_agent.name}' (ID: {db_agent.id}).")
    return db_agent

def delete_agent_definition(db: Session, agent_id: int, user_id: int) -> Optional[Agent]:
    """Deletes an agent definition, ensuring user ownership."""
    db_agent = get_agent_definition(db, agent_id, user_id)
    if not db_agent:
        return None
        
    agent_name = db_agent.name # For logging
    db.delete(db_agent)
    db.commit()
    logger.info(f"Deleted agent definition '{agent_name}' (ID: {agent_id}).")
    # TODO: Add cleanup logic? (e.g., associated logs, memory?)
    return db_agent

def list_all_agents(db: Session, skip: int = 0, limit: int = 100) -> List[Agent]:
    """Lists all agent definitions in the system, regardless of owner."""
    return db.query(Agent).order_by(Agent.id).offset(skip).limit(limit).all()

# --- Agent Discovery Functions ---

def get_agent_by_id(db: Session, agent_id: int) -> Optional[Agent]:
    """Gets a specific agent definition by ID, regardless of owner."""
    return db.query(Agent).filter(Agent.id == agent_id, Agent.is_active == True).first()

def get_agent_by_name(db: Session, agent_name: str, user_id: int) -> Optional[Agent]:
    """Gets a specific active agent definition by name for a specific user."""
    return db.query(Agent).filter(
        Agent.name == agent_name, 
        Agent.owner_user_id == user_id, 
        Agent.is_active == True
    ).first()

def find_active_agent_by_name(db: Session, agent_name: str) -> Optional[Agent]:
    """Finds the first active agent definition by name, regardless of owner."""
    # NOTE: Assumes names might not be unique across users. Returns the first active match.
    return db.query(Agent).filter(Agent.name == agent_name, Agent.is_active == True).first()

def list_all_active_agents(db: Session, skip: int = 0, limit: int = 100) -> List[Agent]:
    """Lists all active agent definitions in the system."""
    return db.query(Agent).filter(Agent.is_active == True).offset(skip).limit(limit).all()

# --- Agent Capability Management --- 

def get_agent_capabilities(db: Session, agent_id: int) -> List[str]:
    """Gets the list of tool names the agent is allowed to use."""
    capabilities = db.query(AgentCapability.tool_name).filter(AgentCapability.agent_id == agent_id).all()
    return [c[0] for c in capabilities] # Extract tool names from tuples

def add_agent_capability(db: Session, agent_id: int, tool_name: str, user_id: int) -> Optional[AgentCapability]:
    """Adds a tool capability to an agent, ensuring user ownership."""
    # Ensure agent exists and user owns it
    agent = get_agent_definition(db, agent_id=agent_id, user_id=user_id)
    if not agent:
        logger.warning(f"Attempt to add capability for non-existent or unauthorized agent {agent_id} by user {user_id}.")
        return None
        
    # TODO: Check if tool_name actually exists in the tool_registry?
    
    # Check if capability already exists
    existing_cap = db.query(AgentCapability).filter(
        AgentCapability.agent_id == agent_id, 
        AgentCapability.tool_name == tool_name
    ).first()
    
    if existing_cap:
        logger.debug(f"Capability '{tool_name}' already exists for agent {agent_id}.")
        return existing_cap
        
    new_cap = AgentCapability(agent_id=agent_id, tool_name=tool_name)
    try:
        db.add(new_cap)
        db.commit()
        db.refresh(new_cap)
        logger.info(f"Added capability '{tool_name}' to agent {agent_id} ('{agent.name}').")
        return new_cap
    except IntegrityError: # Catch potential race conditions or other DB errors
        db.rollback()
        logger.error(f"Failed to add capability '{tool_name}' to agent {agent_id} due to integrity error.", exc_info=True)
        # Re-check if it exists now just in case
        return db.query(AgentCapability).filter(AgentCapability.agent_id == agent_id, AgentCapability.tool_name == tool_name).first()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to add capability '{tool_name}' to agent {agent_id}: {e}", exc_info=True)
        return None

def remove_agent_capability(db: Session, agent_id: int, tool_name: str, user_id: int) -> bool:
    """Removes a tool capability from an agent, ensuring user ownership. Returns True if deleted."""
    # Ensure agent exists and user owns it
    agent = get_agent_definition(db, agent_id=agent_id, user_id=user_id)
    if not agent:
        logger.warning(f"Attempt to remove capability for non-existent or unauthorized agent {agent_id} by user {user_id}.")
        return False
        
    cap_to_delete = db.query(AgentCapability).filter(
        AgentCapability.agent_id == agent_id, 
        AgentCapability.tool_name == tool_name
    ).first()
    
    if not cap_to_delete:
        logger.warning(f"Capability '{tool_name}' not found for agent {agent_id}. Cannot remove.")
        return False
        
    try:
        db.delete(cap_to_delete)
        db.commit()
        logger.info(f"Removed capability '{tool_name}' from agent {agent_id} ('{agent.name}').")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to remove capability '{tool_name}' from agent {agent_id}: {e}", exc_info=True)
        return False

# --- Agent Log Retrieval --- 

def get_agent_logs(
    db: Session,
    agent_id: Optional[int] = None,
    user_id: Optional[int] = None, # Filter by owner if needed
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    levels: Optional[List[str]] = None,
    skip: int = 0,
    limit: int = 100
) -> List[AgentLog]:
    """Retrieves agent logs with optional filtering."""
    query = db.query(AgentLog)
    
    if agent_id is not None:
        query = query.filter(AgentLog.agent_id == agent_id)
        
    # Optional: Filter by agent owner - requires joining Agent table
    if user_id is not None:
         query = query.join(Agent, AgentLog.agent_id == Agent.id)
         query = query.filter(Agent.owner_user_id == user_id)
         
    if start_time is not None:
        query = query.filter(AgentLog.timestamp >= start_time)
        
    if end_time is not None:
        query = query.filter(AgentLog.timestamp <= end_time)
        
    if levels:
        # Ensure levels are uppercase for matching
        upper_levels = [level.upper() for level in levels]
        query = query.filter(AgentLog.level.in_(upper_levels))
        
    # Order by timestamp descending by default
    query = query.order_by(desc(AgentLog.timestamp))
    
    logs = query.offset(skip).limit(limit).all()
    logger.debug(f"Retrieved {len(logs)} agent logs with provided filters.")
    return logs

# Add other management functions here (e.g., list capabilities)

# --- Admin Agent CRUD (Bypass Ownership Checks) ---

def update_agent_definition_admin(db: Session, agent_id: int, agent_update: AgentUpdate) -> Optional[Agent]:
    """Updates any agent definition by ID (Admin only)."""
    db_agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not db_agent:
        logger.warning(f"[Admin] Agent ID {agent_id} not found for update.")
        return None
    
    update_data = agent_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_agent, key, value)
        
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    logger.info(f"[Admin] Updated agent definition '{db_agent.name}' (ID: {db_agent.id}).")
    return db_agent

def delete_agent_definition_admin(db: Session, agent_id: int) -> Optional[Agent]:
    """Deletes any agent definition by ID (Admin only)."""
    db_agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not db_agent:
        logger.warning(f"[Admin] Agent ID {agent_id} not found for deletion.")
        return None
        
    agent_name = db_agent.name # For logging
    db.delete(db_agent)
    db.commit()
    logger.info(f"[Admin] Deleted agent definition '{agent_name}' (ID: {agent_id}).")
    return db_agent

# --- Agent Task Management ---

def get_active_tasks(db: Session, user_id: Optional[int] = None) -> List[AgentTask]:
    """Retrieves tasks currently in the 'running' state.
    If user_id is provided, filters for tasks owned by that user.
    """
    query = db.query(AgentTask).filter(AgentTask.status == 'running')
    if user_id is not None:
        query = query.filter(AgentTask.user_id == user_id)
    return query.order_by(AgentTask.created_at.asc()).all()
