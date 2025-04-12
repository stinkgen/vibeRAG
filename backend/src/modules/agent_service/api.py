"""API endpoints for managing Agent definitions."""

import logging
from typing import List, Optional, Dict, Any, Set
import uuid # For generating task IDs
from datetime import datetime # Import datetime for query params
from fastapi import Query, Response, status, WebSocket, WebSocketDisconnect
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, create_model # Added create_model
import asyncio
from urllib.parse import parse_qs

# Dependency to get current user (assuming agents are user-scoped)
from src.modules.auth.auth import get_current_user, get_current_active_admin_user, decode_access_token, get_user_by_id # Use get_current_user
from src.modules.auth.database import User, get_db, AgentTask as AgentTaskModel, Agent

# Agent CRUD functions and Pydantic models
from .manager import (
    create_agent_definition,
    get_agent_definition,
    get_agent_definitions_by_user,
    update_agent_definition,
    delete_agent_definition,
    get_agent_capabilities,
    add_agent_capability,
    remove_agent_capability,
    get_agent_logs
)
# Agent Memory functions
from .memory import (
    store_memory,
    retrieve_relevant_memories,
    delete_memory_by_id
)
from src.modules.auth.database import (
    Agent, 
    AgentCreate, 
    AgentUpdate, 
    AgentResponse,
    AgentLogResponse, # Use Pydantic model for response
    AgentMemoryResponse, # Import AgentMemoryResponse
    AgentMemory, # Import AgentMemory
    AgentCapability # Import AgentCapability
)

# Import task/output models (needed for schemas?)
from .models import AgentTask, AgentOutput 

# Import manager functions
from . import manager as agent_manager 
# Import capability model for response/request if needed (using List[str] for now)
from src.modules.auth.database import AgentCapability 

# Import memory retrieval function
from . import memory as agent_memory

# Import tool registry and definition model
from .tools import tool_registry, ToolDefinition

# Import schemas (now defined in schemas.py)
from .schemas import AgentRunRequest, AgentTaskQueuedResponse
# Keep others if needed for other parts of api.py?
# from .schemas import AgentDefinitionCreate, AgentDefinitionUpdate, AgentDefinitionSchema, AgentCapabilityCreate, AgentCapabilitySchema, AgentLogSchema, AgentMemoryCreate, AgentMemorySchema, AgentOutput, AgentTaskStatusSchema, ScratchpadEntrySchema, AgentTask

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["Agents"],
)

# --- Tool Listing Endpoint --- 

# Dynamically create a Pydantic model for ToolDefinition suitable for API response
# (excluding the execute_func which is not serializable)
ToolInfo = create_model(
    'ToolInfo',
    __base__=ToolDefinition,
    # Exclude non-serializable fields or fields not needed by UI
    __exclude_fields__ = {'execute_func', 'mcp_requirements', 'expects_context'} 
)

@router.get("/tools/", response_model=List[ToolInfo], tags=["Tools"])
def list_available_tools():
    """Lists all available tools registered in the system."""
    logger.debug("Request received for listing available tools.")
    tools = tool_registry.list_tools()
    # Pydantic conversion to ToolInfo based on response_model happens automatically
    return tools

@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def handle_create_agent_definition(
    agent: AgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Creates a new agent definition owned by the current user."""
    # Ensure the owner_user_id matches the current user, or is ignored and set server-side
    if agent.owner_user_id != current_user.id:
         # Option 1: Raise error if ID mismatch
         # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Owner ID mismatch")
         # Option 2: Ignore provided owner_id and use current_user.id (Safer)
         logger.warning(f"AgentCreate owner_user_id ({agent.owner_user_id}) does not match current user ({current_user.id}). Overriding.")
         agent.owner_user_id = current_user.id 
    
    logger.info(f"User '{current_user.username}' creating agent '{agent.name}'")
    db_agent = create_agent_definition(db=db, agent=agent, owner=current_user)
    return db_agent

@router.get("/", response_model=List[AgentResponse])
def handle_get_agent_definitions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves all agent definitions owned by the current user."""
    logger.info(f"User '{current_user.username}' retrieving agent definitions.")
    agents = get_agent_definitions_by_user(db=db, user_id=current_user.id, skip=skip, limit=limit)
    return agents

@router.get("/all", 
             response_model=List[AgentResponse], 
             dependencies=[Depends(get_current_active_admin_user)],
             summary="List All Agents (Admin Only)",
             description="Retrieves all agent definitions in the system. Requires admin privileges.")
def handle_get_all_agent_definitions_admin(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_admin_user: User = Depends(get_current_active_admin_user) # Just to log who made the request
):
    """Admin endpoint to retrieve all agent definitions."""
    logger.info(f"Admin user '{current_admin_user.username}' retrieving all agent definitions.")
    try:
        agents = agent_manager.list_all_agents(db=db, skip=skip, limit=limit)
        return agents
    except Exception as e:
        logger.error(f"Failed to retrieve all agents for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve all agents.")

@router.get("/{agent_id}", response_model=AgentResponse)
def handle_get_agent_definition(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves a specific agent definition by ID, owned by the current user."""
    logger.info(f"User '{current_user.username}' retrieving agent definition ID: {agent_id}.")
    db_agent = get_agent_definition(db=db, agent_id=agent_id, user_id=current_user.id)
    if db_agent is None:
        logger.warning(f"Agent definition ID {agent_id} not found for user '{current_user.username}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    return db_agent

@router.put("/{agent_id}", response_model=AgentResponse)
def handle_update_agent_definition(
    agent_id: int,
    agent_update: AgentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Updates an agent definition owned by the current user."""
    logger.info(f"User '{current_user.username}' updating agent definition ID: {agent_id}.")
    db_agent = update_agent_definition(db=db, agent_id=agent_id, agent_update=agent_update, user_id=current_user.id)
    if db_agent is None:
        logger.warning(f"Agent definition ID {agent_id} not found for user '{current_user.username}' during update attempt.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    return db_agent

@router.delete("/{agent_id}", response_model=AgentResponse) # Or just status code 204
def handle_delete_agent_definition(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deletes an agent definition owned by the current user."""
    logger.info(f"User '{current_user.username}' deleting agent definition ID: {agent_id}.")
    db_agent = delete_agent_definition(db=db, agent_id=agent_id, user_id=current_user.id)
    if db_agent is None:
        logger.warning(f"Agent definition ID {agent_id} not found for user '{current_user.username}' during delete attempt.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    # Optionally return status.HTTP_204_NO_CONTENT instead of the deleted object
    return db_agent 

# --- NEW: Admin Agent Update/Delete Endpoints --- 

@router.put("/admin/{agent_id}", 
            response_model=AgentResponse, 
            dependencies=[Depends(get_current_active_admin_user)],
            summary="Update Any Agent (Admin Only)",
            description="Updates any agent definition by ID. Requires admin privileges.")
def handle_update_agent_definition_admin(
    agent_id: int,
    agent_update: AgentUpdate,
    db: Session = Depends(get_db),
    current_admin_user: User = Depends(get_current_active_admin_user) # Log who did it
):
    """Admin endpoint to update any agent definition."""
    logger.info(f"Admin user '{current_admin_user.username}' updating agent definition ID: {agent_id}.")
    db_agent = agent_manager.update_agent_definition_admin(db=db, agent_id=agent_id, agent_update=agent_update)
    if db_agent is None:
        logger.warning(f"[Admin] Failed update attempt for non-existent agent ID {agent_id}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    return db_agent

@router.delete("/admin/{agent_id}", 
              response_model=AgentResponse, # Or status code 204
              dependencies=[Depends(get_current_active_admin_user)],
              summary="Delete Any Agent (Admin Only)",
              description="Deletes any agent definition by ID. Requires admin privileges.")
def handle_delete_agent_definition_admin(
    agent_id: int,
    db: Session = Depends(get_db),
    current_admin_user: User = Depends(get_current_active_admin_user) # Log who did it
):
    """Admin endpoint to delete any agent definition."""
    logger.info(f"Admin user '{current_admin_user.username}' deleting agent definition ID: {agent_id}.")
    db_agent = agent_manager.delete_agent_definition_admin(db=db, agent_id=agent_id)
    if db_agent is None:
        logger.warning(f"[Admin] Failed delete attempt for non-existent agent ID {agent_id}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    # Optionally return Response(status_code=status.HTTP_204_NO_CONTENT)
    return db_agent

# --- Agent Execution Endpoint (Now Async Task Dispatcher) --- 
# Change response model to indicate queuing
@router.post("/{agent_id}/run", 
             response_model=AgentTaskQueuedResponse, 
             status_code=status.HTTP_202_ACCEPTED) # Use 202 Accepted
async def handle_run_agent_task(
    agent_id: int,
    run_request: AgentRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Queues an agent task for asynchronous execution via Celery."""
    # 1. Verify Agent Exists and is Active
    agent_def = get_agent_definition(db=db, agent_id=agent_id, user_id=current_user.id)
    if agent_def is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found or not accessible")
    if not agent_def.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Agent is not active")
        
    logger.info(f"User '{current_user.username}' queuing task for agent '{agent_def.name}' (ID: {agent_id}) with goal: {run_request.goal}")
    
    # 2. Create AgentTask DB Record
    try:
        db_task = AgentTaskModel(
            agent_id=agent_id,
            user_id=current_user.id,
            goal=run_request.goal,
            input_data=run_request.input_data,
            status="pending" # Initial status
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        task_db_id = db_task.id
        logger.info(f"Created AgentTask record with ID: {task_db_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create AgentTask record for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create task record.")

    # 3. Dispatch Celery Task
    try:
        # Pass data needed by the task worker (minimal is best, here just the DB ID)
        task_payload = {"id": task_db_id} 
        celery_result = execute_agent_task.delay(agent_task_data=task_payload)
        celery_task_id = celery_result.id
        logger.info(f"Dispatched Celery task {celery_task_id} for AgentTask DB ID {task_db_id}")
        
        # 4. Update DB record with Celery Task ID
        db_task.celery_task_id = celery_task_id
        db.commit()
        
        # 5. Return Queued Response
        return AgentTaskQueuedResponse(
            task_db_id=task_db_id,
            celery_task_id=celery_task_id
        )

    except Exception as e:
        logger.error(f"Failed to dispatch Celery task for AgentTask DB ID {task_db_id}: {e}", exc_info=True)
        # Attempt to mark DB task as failed if dispatch fails?
        db_task.status = "failed"
        db_task.error_message = f"Failed to dispatch Celery task: {str(e)}"
        db.commit() 
        # Raise error to client
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to queue agent task.")

# --- Agent Capability Endpoints --- 

@router.get("/{agent_id}/capabilities", response_model=List[str])
def get_agent_capabilities_route(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Gets the list of capabilities (tool names) for a specific agent owned by the user."""
    # Verify ownership first
    agent = get_agent_definition(db=db, agent_id=agent_id, user_id=current_user.id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found or not owned by user")
    
    capabilities = agent_manager.get_agent_capabilities(db, agent_id=agent_id)
    return capabilities

# Define a request model for adding a capability
class AddCapabilityRequest(BaseModel):
    tool_name: str

@router.post("/{agent_id}/capabilities", status_code=status.HTTP_201_CREATED) # Using 201 for creation
def add_agent_capability_route(
    agent_id: int,
    request: AddCapabilityRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Adds a capability (tool name) to an agent owned by the user."""
    # Manager function handles ownership check and creation logic
    new_cap = agent_manager.add_agent_capability(
        db=db, 
        agent_id=agent_id, 
        tool_name=request.tool_name, 
        user_id=current_user.id
    )
    if new_cap is None:
        # Handle specific errors? e.g., Agent not found vs. Capability already exists vs. DB error
        # For now, generic error if creation failed (manager logs specifics)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to add capability '{request.tool_name}'. Agent might not exist or capability already added.")
    
    # Return success message or the capability object? Let's return a message.
    return {"message": f"Capability '{request.tool_name}' added successfully to agent {agent_id}."}

@router.delete("/{agent_id}/capabilities/{tool_name}", status_code=status.HTTP_200_OK)
def remove_agent_capability_route(
    agent_id: int,
    tool_name: str, # Get tool name from path parameter
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Removes a capability (tool name) from an agent owned by the user."""
    # Manager function handles ownership check and deletion
    deleted = agent_manager.remove_agent_capability(
        db=db, 
        agent_id=agent_id, 
        tool_name=tool_name, 
        user_id=current_user.id
    )
    
    if not deleted:
        # Agent/Capability not found or deletion failed
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Capability '{tool_name}' not found for agent {agent_id} or could not be removed.")
        
    return {"message": f"Capability '{tool_name}' removed successfully from agent {agent_id}."} 

# --- Agent Log Retrieval Endpoint --- 

@router.get("/logs/", response_model=List[AgentLogResponse])
def get_agent_logs_route(
    agent_id: Optional[int] = None,
    user_id: Optional[int] = None, # Allow filtering by user (e.g., for admins)
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    levels: Optional[List[str]] = Query(None), # Use Query for list parameters
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Require authentication
):
    """Retrieves agent logs with filtering. 
    
    Requires authentication. Non-admin users can only view logs for their own agents
    unless specific agent_id is provided AND they own that agent.
    Admins can view logs across users if user_id is omitted or set.
    """
    # Security Check: If user is not admin, enforce filtering by their user_id
    effective_user_id = user_id
    if not current_user.is_admin:
        if user_id is not None and user_id != current_user.id:
             raise HTTPException(status_code=403, detail="Non-admin users cannot view logs for other users.")
        elif agent_id is not None:
             # Check if the non-admin user owns the requested agent_id
             agent = agent_manager.get_agent_definition(db, agent_id=agent_id, user_id=current_user.id)
             if not agent:
                  raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found or not owned by user.")
             # If owning the specific agent, allow seeing its logs (user_id filter in manager is redundant here)
             effective_user_id = None # Let manager filter only by agent_id
        else:
            # If no specific agent_id requested by non-admin, force filter by their ID
            effective_user_id = current_user.id 
            
    try:
        logs = agent_manager.get_agent_logs(
            db=db,
            agent_id=agent_id,
            user_id=effective_user_id, # Apply security-checked user_id filter
            start_time=start_time,
            end_time=end_time,
            levels=levels,
            skip=skip,
            limit=limit
        )
        return logs
    except Exception as e:
        logger.error(f"Failed to retrieve agent logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent logs.")

# --- Agent Memory Retrieval Endpoint --- 

@router.get("/{agent_id}/memory", response_model=List[AgentMemoryResponse])
async def get_agent_memory_route(
    agent_id: int,
    query_text: str, # Required query parameter for similarity search
    limit: int = 5,  # Optional limit with default
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves relevant memories for a specific agent owned by the user, based on query text similarity."""
    # 1. Verify ownership
    agent = agent_manager.get_agent_definition(db, agent_id=agent_id, user_id=current_user.id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found or not owned by user")
        
    # 2. Call memory retrieval function
    try:
        memories = await agent_memory.retrieve_relevant_memories(
            db=db,
            agent_id=agent_id,
            query_text=query_text,
            limit=limit
        )
        # AgentMemoryResponse should map correctly via from_attributes=True
        return memories
    except Exception as e:
        logger.error(f"Failed to retrieve memories for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent memories.")

# --- Manual Agent Memory Management Endpoints --- 

class RetrieveMemoriesRequest(BaseModel):
    query_text: str = Field(..., description="Text to search for similar memories.")
    limit: int = Field(5, gt=0, description="Maximum number of memories to retrieve.")

@router.post("/{agent_id}/memory/retrieve", 
             response_model=List[AgentMemoryResponse],
             status_code=status.HTTP_200_OK,
             dependencies=[Depends(get_current_user)],
             summary="Retrieve Agent Memories",
             description="Retrieves relevant memories for a specific agent based on a query text.")
async def handle_retrieve_agent_memories(
    agent_id: int,
    request: RetrieveMemoriesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves relevant memories for a specific agent based on a query text."""
    # Verify agent exists and belongs to user
    agent = get_agent_definition(db=db, agent_id=agent_id, user_id=current_user.id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        
    # Call the memory retrieval function
    memories = await agent_memory.retrieve_relevant_memories(
        db=db,
        agent_id=agent_id,
        query_text=request.query_text,
        limit=request.limit
    )
    return memories # Pydantic conversion

# Endpoint to manually add memory
class ManualMemoryCreate(BaseModel):
    memory_type: str = Field(default="manual", description="Type of memory (e.g., manual, observation)")
    content: str = Field(..., description="The textual content of the memory.")
    importance: Optional[float] = Field(0.5, description="Importance score (0.0-1.0)")
    # Allow optional embedding if user provides it
    embedding: Optional[List[float]] = Field(None, description="Optional pre-computed embedding vector.")

@router.post("/{agent_id}/memory", 
             response_model=AgentMemoryResponse,
             status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(get_current_user)],
             summary="Manually Add Agent Memory",
             description="Adds a new memory entry (metadata and optional embedding) to the specified agent.")
async def handle_add_agent_memory(
    agent_id: int,
    memory_data: ManualMemoryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Manually adds a memory entry (metadata and optional embedding) to an agent owned by the user."""
    # Verify agent exists and belongs to user
    db_agent = get_agent_definition(db=db, agent_id=agent_id, user_id=current_user.id)
    if not db_agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        
    # Use store_memory function from memory.py
    created_memory = await agent_memory.store_memory(
        db=db,
        agent_id=agent_id,
        memory_type=memory_data.memory_type,
        content=memory_data.content,
        importance=memory_data.importance,
        embedding=memory_data.embedding
    )
    
    if not created_memory:
        raise HTTPException(status_code=500, detail="Failed to store memory.")
        
    return created_memory # Pydantic conversion

# Endpoint to delete memory
@router.delete("/{agent_id}/memory/{memory_id}",
              status_code=status.HTTP_204_NO_CONTENT,
              dependencies=[Depends(get_current_user)],
              summary="Delete Agent Memory",
              description="Deletes a specific memory (metadata and embedding) by its ID.")
async def handle_delete_agent_memory(
    agent_id: int,
    memory_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deletes a specific memory (metadata and embedding) for an agent owned by the user."""
    # Verify agent exists and belongs to user first
    db_agent = get_agent_definition(db=db, agent_id=agent_id, user_id=current_user.id)
    if not db_agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        
    # Call the delete function from memory.py
    success = await agent_memory.delete_memory_by_id(db=db, memory_id=memory_id, agent_id=agent_id)
    if not success:
        # delete_memory_by_id logs details, maybe return 404 if memory specifically not found for agent?
        raise HTTPException(status_code=404, detail="Memory not found or deletion failed")
        
    return Response(status_code=status.HTTP_204_NO_CONTENT) # Return No Content on success

# --- Pydantic Model for Active Task Response --- 
class ActiveAgentTaskResponse(BaseModel):
    task_db_id: int
    agent_id: int
    user_id: int
    status: str
    goal: str
    created_at: datetime
    celery_task_id: Optional[str] = None
    
    class Config:
        from_attributes = True

# --- Agent Task Status/Monitoring Endpoints ---
@router.get("/tasks/active", 
             response_model=List[ActiveAgentTaskResponse], 
             dependencies=[Depends(get_current_user)], # Use regular user auth
             summary="List User's Active Agent Tasks",
             description="Retrieves agent tasks currently in the 'running' state for the authenticated user.")
def handle_get_active_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists the authenticated user's currently running agent tasks."""
    logger.info(f"User '{current_user.username}' requesting list of their active agent tasks.")
    try:
        # Filter tasks by the current user's ID
        active_tasks = agent_manager.get_active_tasks(db=db, user_id=current_user.id)
        return active_tasks # Pydantic conversion handled by response_model
    except Exception as e:
        logger.error(f"Failed to retrieve active tasks for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve active tasks.")

# TODO: Add endpoint to get status/result of a specific task ID (using DB ID or Celery ID)
# e.g., GET /tasks/{task_db_id}/status

# --- WebSocket Connection Manager --- 
class ConnectionManager:
    def __init__(self):
        # Store active connections: {user_id: {WebSocket}} - Use a set for uniqueness
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"WebSocket connected for user {user_id}. Total connections for user: {len(self.active_connections[user_id])}")

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            logger.info(f"WebSocket disconnected for user {user_id}. Remaining connections for user: {len(self.active_connections[user_id])}")
            # Clean up user entry if no connections left
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                logger.info(f"Removed user {user_id} from active connections.")
        else:
            logger.warning(f"Attempted to disconnect WebSocket for user {user_id} but user not found in active connections.")

    async def send_personal_message(self, message: str, user_id: int, websocket: WebSocket):
        """Sends a message to a specific websocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message to specific websocket for user {user_id}: {e}")

    async def broadcast_to_user(self, message: str, user_id: int):
        """Sends a message to all active websocket connections for a specific user."""
        if user_id in self.active_connections:
            # Create a list of tasks to send messages concurrently
            tasks = [conn.send_text(message) for conn in self.active_connections[user_id]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log any errors that occurred during broadcast
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    ws = list(self.active_connections[user_id])[i] # Get corresponding websocket
                    logger.error(f"Failed to broadcast to websocket {ws.client} for user {user_id}: {result}")
        else:
            logger.debug(f"No active WebSocket connections found for user {user_id} to broadcast message.")

# Instantiate the manager (global instance)
connection_manager = ConnectionManager()

# --- Add WebSocket Endpoint --- 
# Needs to be added after the router definition

# --- WebSocket Endpoint for Task Updates ---
# Helper for WebSocket Auth using Cookie OR Query Parameter
async def get_user_for_ws(websocket: WebSocket, db: Session) -> Optional[User]:
    token = None
    auth_source = "None"
    # 1. Try getting token from query parameter first
    try:
        query_params = parse_qs(websocket.url.query)
        token_list = query_params.get('token')
        if token_list:
            token = token_list[0]
            auth_source = "Query Param"
            logger.debug(f"WebSocket attempting auth via query parameter. Token found: {token[:10]}...")
    except Exception as e:
        logger.error(f"Error parsing WebSocket query parameters: {e}")
        token = None # Ensure token is None if parsing fails

    # 2. If no token from query param, try the cookie
    if not token:
        token = websocket.cookies.get("access_token")
        if token:
            auth_source = "Cookie"
            logger.debug(f"WebSocket attempting auth via cookie. Token found: {token[:10]}...")

    # 3. If still no token, fail authentication
    if not token:
        logger.warning("WebSocket connection attempt without access_token (checked query param and cookie).")
        return None
        
    logger.info(f"WebSocket attempting validation. Source: {auth_source}, Token: {token[:10]}...")
    # 4. Validate the token (same logic as before)
    try:
        token_data = decode_access_token(token)
        logger.debug(f"Decoded token data: {token_data}") # Log decoded data
        if token_data is None or token_data.user_id is None:
            logger.warning(f"WebSocket connection attempt with invalid token (decode result invalid): {token_data}")
            return None
            
        user_id_from_token = token_data.user_id
        logger.debug(f"Attempting to fetch user with ID: {user_id_from_token}")
        user = get_user_by_id(db, user_id=user_id_from_token)
        logger.debug(f"User fetch result: {user}") # Log user fetch result
        
        if user is None:
             logger.warning(f"WebSocket connection attempt: User {user_id_from_token} not found in DB.")
             return None
        elif not user.is_active:
            logger.warning(f"WebSocket connection attempt: User {user_id_from_token} is inactive.")
            return None
            
        logger.info(f"WebSocket successfully authenticated user {user.id} via {auth_source}.")
        return user
    except Exception as e:
        logger.error(f"WebSocket token validation error ({type(e).__name__}): {e}", exc_info=True)
        return None

@router.websocket("/ws/tasks")
async def websocket_task_updates(websocket: WebSocket): # Removed db: Session = Depends(get_db)
    # --- Perform Authentication using Helper --- 
    
    # !! Need a way to get DB session here if auth helper needs it !!
    # Temporary: Create a session manually (NOT RECOMMENDED for production)
    db_session_local = SessionLocal()
    try:
        user = await get_user_for_ws(websocket, db_session_local) # Pass the manually created session
    finally:
        db_session_local.close() # Ensure session is closed
        
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    authenticated_user_id = user.id
    # --- End Authentication ---
    
    await connection_manager.connect(websocket, authenticated_user_id)
    try:
        # Keep the connection alive
        while True:
            await asyncio.sleep(60) 
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, authenticated_user_id)
        logger.info(f"WebSocket connection closed for user {authenticated_user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {authenticated_user_id}: {e}", exc_info=True)
        connection_manager.disconnect(websocket, authenticated_user_id)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except RuntimeError:
            pass # Already closed

# Mount this router in the main application 