from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# --- Model for triggering agent task ---
class AgentRunRequest(BaseModel):
    goal: str = Field(..., description="The specific goal or query for the agent task.")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Optional input data for the task.")
    # Add other relevant parameters like specific model/provider overrides later if needed

# --- Response model for task queuing ---
class AgentTaskQueuedResponse(BaseModel):
    message: str = "Task queued successfully."
    task_db_id: int
    celery_task_id: str

# --- Pydantic Model for Active Task Response --- 
# (Could also go here, but currently defined in api.py and seems only used there)
# class ActiveAgentTaskResponse(BaseModel):
#    task_db_id: int
#    agent_id: int
#    user_id: int
#    status: str
#    goal: str
#    created_at: datetime
#    celery_task_id: Optional[str] = None
#    
#    class Config:
#        from_attributes = True

# --- Model for Manual Memory Creation Request ---
# (Currently defined in api.py, seems specific to that endpoint)
# class ManualMemoryCreate(BaseModel):
#    memory_type: str = Field(default="manual", description="Type of memory (e.g., manual, observation)")
#    content: str = Field(..., description="The textual content of the memory.")
#    importance: Optional[float] = Field(0.5, description="Importance score (0.0-1.0)")
#    embedding: Optional[List[float]] = Field(None, description="Optional pre-computed embedding vector.")

# --- Model for Memory Retrieval Request ---
# (Currently defined in api.py, seems specific to that endpoint)
# class RetrieveMemoriesRequest(BaseModel):
#    query_text: str = Field(..., description="Text to search for similar memories.")
#    limit: int = Field(5, gt=0, description="Maximum number of memories to retrieve.")

# --- Agent Capability Request Model ---
# (Currently defined in api.py)
# class AddCapabilityRequest(BaseModel):
#    tool_name: str 

# --- Pydantic Model for Scratchpad Entries ---+
class ScratchpadEntrySchema(BaseModel):
    role: str # e.g., 'system', 'user', 'assistant' (thought/action), 'tool' (observation)
    content: str

# (Could also go here, but currently defined in api.py and seems only used there)
# class ActiveAgentTaskResponse(BaseModel):
#    task_db_id: int 

# --- Schemas for WebSocket Task Updates --- #

class AgentFinalOutput(BaseModel):
    """Structure for the final payload within a task update."""
    result: Optional[str] = Field(None, description="Final result string if task completed successfully.")
    error: Optional[str] = Field(None, description="Error message string if task failed.")

class AgentTaskUpdatePayload(BaseModel):
    """Structure for messages sent over WebSocket to update task status."""
    type: str = Field("task_update", description="Message type identifier.")
    task_db_id: int = Field(..., description="The database ID of the agent task being updated.")
    status: str = Field(..., description="The current status of the task (e.g., 'running', 'completed', 'failed').")
    payload: AgentFinalOutput = Field(..., description="The final result or error details.")


# --- Agent Memory Schemas --- #
# ... (existing memory schemas) ... 