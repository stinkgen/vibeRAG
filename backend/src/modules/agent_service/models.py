"""Placeholder for Agent Pydantic models specific to the agent service.
May duplicate or refine models from auth.database if needed for service logic.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class AgentTask(BaseModel):
    task_id: str
    agent_id: int
    goal: str
    input_data: Optional[Dict[str, Any]] = None
    # ... other task parameters

class AgentOutput(BaseModel):
    task_id: str
    agent_id: int
    output: Any
    status: str # e.g., completed, failed
    error_message: Optional[str] = None
