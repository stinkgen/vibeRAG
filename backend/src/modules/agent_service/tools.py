"""Tool Registry and definitions for the Agent Service."""

import logging
from typing import Dict, Callable, Any, List, Optional, Union
from pydantic import BaseModel, Field
import asyncio

# Import MCP connector function
from .mcp.connectors import provide_context
# Import actual tool implementation functions
from src.modules.retrieval.search import semantic_search as actual_semantic_search
from src.modules.retrieval.search import google_search as actual_google_search
from src.modules.auth.database import User # Needed for semantic_search
from src.modules.auth.auth import get_user_by_id # To get user object from ID
from src.modules.auth.database import get_db # To get DB session for user lookup
# Import agent manager and runtime for delegation
from . import manager as agent_manager
from . import runtime as agent_runtime
from src.modules.agent_service.schemas import AgentRunRequest # Import from schemas.py
from .logging import log_agent_activity # Import the database logging function

logger = logging.getLogger(__name__)

# --- Tool Definition --- 

class ToolParameter(BaseModel):
    name: str
    type: str # e.g., 'string', 'integer', 'boolean', 'array[string]'
    description: str
    required: bool = True

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    execute_func: Callable[..., Any] # Allow flexible signature for context injection
    mcp_requirements: Optional[List[str]] = None 
    expects_context: Optional[List[str]] = None # Explicit keys expected by execute_func beyond params
    
    class Config:
        arbitrary_types_allowed = True

# --- Tool Registry --- 

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        logger.info("Tool Registry initialized.")

    def register_tool(self, tool_def: ToolDefinition):
        if tool_def.name in self._tools:
            logger.warning(f"Tool '{tool_def.name}' is already registered. Overwriting.")
        self._tools[tool_def.name] = tool_def
        logger.info(f"Registered tool: '{tool_def.name}'")

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())
        
    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def execute_tool(self,
                           name: str,
                           params: Dict[str, Any],
                           mcp_context_identifiers: Dict[str, Any] # e.g., {"user_id": 1}
                           ) -> Any:
        """Executes a registered tool by name, fetching context via MCP."""
        tool_def = self.get_tool(name)
        if not tool_def:
            logger.error(f"Attempted to execute unregistered tool: '{name}'")
            raise ValueError(f"Tool '{name}' not found.")
            
        # --- MCP Context Fetching --- 
        fetched_context = {}
        if tool_def.mcp_requirements:
            log_prefix_mcp = f"[Tool: {name} | MCP]"
            logger.debug(f"{log_prefix_mcp} Requirements: {tool_def.mcp_requirements}")
            try:
                fetched_context = await provide_context(
                    required_context_keys=tool_def.mcp_requirements,
                    **mcp_context_identifiers # Pass user_id etc.
                )
                logger.debug(f"{log_prefix_mcp} Fetched context keys: {fetched_context.keys()}")
            except Exception as mcp_err:
                 logger.error(f"{log_prefix_mcp} Failed to fetch context: {mcp_err}", exc_info=True)
                 raise ValueError(f"Failed to fetch required context for tool '{name}': {mcp_err}")
        # --------------------------
        
        # TODO: Add parameter validation against tool_def.parameters
        log_prefix_exec = f"[Tool: {name} | Execute]"
        logger.info(f"{log_prefix_exec} Attempting execution with params: {params}")
        try:
            # Prepare arguments for the actual tool function
            # This might involve merging params and fetched_context based on function signature
            # Or injecting context separately
            tool_args = params.copy() # Start with parameters provided by agent
            
            # Inject specific context needed by the function if defined
            if tool_def.expects_context:
                 logger.debug(f"{log_prefix_exec} Injecting expected context: {tool_def.expects_context}")
                 for key in tool_def.expects_context:
                     if key in fetched_context:
                         tool_args[key] = fetched_context[key]
                     elif key in mcp_context_identifiers:
                         tool_args[key] = mcp_context_identifiers[key]
                     else:
                         logger.warning(f"{log_prefix_exec} Expected context '{key}' not found in fetched context or identifiers.")
                         # Maybe add default value or raise error based on tool needs?
            
            # Inject the full context dict if the function expects it? (Less clean)
            # tool_args["mcp_context"] = fetched_context 
            
            # Execute the function
            if asyncio.iscoroutinefunction(tool_def.execute_func):
                logger.debug(f"{log_prefix_exec} Awaiting async execute_func...")
                result = await tool_def.execute_func(**tool_args) # Pass args as keywords
            else:
                logger.debug(f"{log_prefix_exec} Running sync execute_func in executor...")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: tool_def.execute_func(**tool_args))
                
            logger.info(f"{log_prefix_exec} Execution successful.")
            logger.debug(f"{log_prefix_exec} Result preview: {str(result)[:200]}...")
            return result
        except Exception as e:
            logger.exception(f"{log_prefix_exec} Error during execution: {e}")
            raise # Re-raise the exception after logging

# --- Global Registry Instance --- 
# Can be instantiated here or managed via FastAPI dependency injection
tool_registry = ToolRegistry()

# --- Tool Implementation Wrappers --- 

# Wrapper for semantic_search to match expected signature and context injection
async def _execute_semantic_search_wrapper(query: str, user_id: int) -> Any:
    """Wrapper to call actual semantic_search with necessary context."""
    logger.info(f"Executing semantic_search for query: '{query}', User ID: {user_id}")
    db = next(get_db())
    try:
        # Fetch User object needed by semantic_search
        user = get_user_by_id(db, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found for semantic search.")
            
        # Call actual implementation (assuming CONFIG provides defaults for limit/min_score)
        results = await actual_semantic_search(
            query=query,
            user=user,
            # limit=CONFIG.search.default_limit, # Let semantic_search handle defaults
            # min_score=CONFIG.search.min_score,
            filters=None # TODO: Allow passing filters from agent params later
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in semantic_search wrapper: {e}", exc_info=True)
        return {"error": str(e)} # Return error info instead of raising?
    finally:
        db.close()

# Wrapper for google_search
async def _execute_web_search_wrapper(query: str) -> Any:
    """Wrapper to call actual google_search."""
    logger.info(f"Executing google_search for query: '{query}'")
    try:
        # Call actual implementation (assuming CONFIG provides default limit)
        # Needs to be async if google_search becomes async
        # results = actual_google_search(query=query)
        # TEMP: Assuming google_search is sync
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, actual_google_search, query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in google_search wrapper: {e}", exc_info=True)
        return {"error": str(e)} 

# --- Delegate Task Tool --- 

async def _execute_delegate_task_wrapper(
    target_agent_identifier: Union[int, str], 
    task_description: str, 
    calling_agent_id: int, 
    user_id: int, # Injected from context
    agent_context: Optional[str] = None) -> Any:
    """Wrapper to delegate a task to another agent."""
    # Use calling_agent_id for logging context
    log_prefix = f"[Tool: delegate_task | Caller: {calling_agent_id} | User: {user_id}]"
    logger.info(f"{log_prefix} Attempting delegation to '{target_agent_identifier}' for task: '{task_description[:50]}...'")
    
    db = next(get_db())
    target_agent = None
    try:
        # Log delegation attempt
        log_agent_activity(db, agent_id=calling_agent_id, level="ACTION", message="Initiating delegate_task tool.",
                           details={"target": target_agent_identifier, "task_preview": task_description[:100]})
                           
        # 1. Find Target Agent
        if isinstance(target_agent_identifier, int):
            logger.debug(f"{log_prefix} Finding target agent by ID: {target_agent_identifier}")
            target_agent = agent_manager.get_agent_by_id(db, target_agent_identifier)
        elif isinstance(target_agent_identifier, str):
            logger.debug(f"{log_prefix} Finding target agent by Name: '{target_agent_identifier}'")
            target_agent = agent_manager.find_active_agent_by_name(db, target_agent_identifier)
        else:
            err_msg = "target_agent_identifier must be an integer ID or string name."
            log_agent_activity(db, agent_id=calling_agent_id, level="ERROR", message="[Delegate] Invalid target identifier type.", details={"identifier": target_agent_identifier, "error": err_msg})
            raise ValueError(err_msg)

        if not target_agent:
            err_msg = f"Target agent '{target_agent_identifier}' not found or inactive."
            logger.error(f"{log_prefix} {err_msg}")
            log_agent_activity(db, agent_id=calling_agent_id, level="ERROR", message="[Delegate] Target agent not found or inactive.", details={"target": target_agent_identifier, "error": err_msg})
            raise ValueError(err_msg)
        
        logger.info(f"{log_prefix} Found target agent '{target_agent.name}' (ID: {target_agent.id}).")
        log_agent_activity(db, agent_id=calling_agent_id, level="DEBUG", message="[Delegate] Found target agent.", 
                           details={"target_name": target_agent.name, "target_id": target_agent.id})

        # 2. Prevent Self-Delegation
        if target_agent.id == calling_agent_id:
            err_msg = "Agent cannot delegate task to itself."
            logger.warning(f"{log_prefix} Agent attempted to delegate task to itself. Aborting delegation.")
            log_agent_activity(db, agent_id=calling_agent_id, level="WARN", message="[Delegate] Self-delegation attempt blocked.", details={"error": err_msg})
            raise ValueError(err_msg)
            
        # 3. Prepare Request for Target Agent
        delegate_request = AgentRunRequest(
            agent_id=target_agent.id,
            user_id=user_id, # Run delegated task as the original user
            prompt=task_description,
            agent_context=agent_context, # Pass optional context
        )
        logger.debug(f"{log_prefix} Prepared AgentRunRequest for agent {target_agent.id}")

        # 4. Execute Task on Target Agent
        logger.info(f"{log_prefix} Calling run_agent_task for agent {target_agent.id}...")
        log_agent_activity(db, agent_id=calling_agent_id, level="INFO", message=f"[Delegate] Calling run_agent_task on target agent '{target_agent.name}' (ID: {target_agent.id}).", 
                           details={"target_id": target_agent.id, "task_preview": task_description[:100]})
        try:
            result = await agent_runtime.run_agent_task(request=delegate_request, db=db)
            logger.info(f"{log_prefix} Delegation to agent {target_agent.id} completed.")
            result_dump = result.model_dump() # Return Pydantic model as dict
            log_agent_activity(db, agent_id=calling_agent_id, level="OBSERVATION", message=f"[Delegate] Received result from target agent {target_agent.id}.", 
                               details={"target_id": target_agent.id, "result_status": result.status, "result_preview": str(result_dump)[:200]})
            return result_dump
        except Exception as run_err:
            err_msg = f"Error during delegated run_agent_task for agent {target_agent.id}: {run_err}"
            logger.error(f"{log_prefix} {err_msg}", exc_info=True)
            log_agent_activity(db, agent_id=calling_agent_id, level="ERROR", message=f"[Delegate] Error during delegated task execution for target {target_agent.id}.", 
                               details={"target_id": target_agent.id, "error": str(run_err)})
            return {"error": f"Delegation failed: {str(run_err)}"}

    except Exception as e:
        err_msg = f"Error during delegation setup: {e}"
        logger.error(f"{log_prefix} {err_msg}", exc_info=True)
        log_agent_activity(db, agent_id=calling_agent_id, level="ERROR", message="[Delegate] Error during delegation setup.", 
                           details={"target": target_agent_identifier, "error": str(e)})
        return {"error": str(e)} # Return error info
    finally:
        db.close()


delegate_task_tool = ToolDefinition(
    name="delegate_task",
    description="Delegates a specific task to another specialized agent and returns the result. Provide the target agent's unique name or ID.",
    parameters=[
        ToolParameter(name="target_agent_identifier", type="Union[int, str]", description="The unique ID (integer) or name (string) of the *active* agent to delegate the task to.", required=True),
        ToolParameter(name="task_description", type="string", description="A clear and specific description of the task to be performed by the target agent.", required=True),
        ToolParameter(name="calling_agent_id", type="integer", description="Internal Use: ID of the agent initiating the delegation. This is injected automatically.", required=True),
        ToolParameter(name="agent_context", type="string", description="Optional JSON string providing necessary context, data, or input parameters for the task.", required=False),
    ],
    execute_func=_execute_delegate_task_wrapper,
    mcp_requirements=["user_id"], # Needs user_id for context
    expects_context=["user_id"]   # Wrapper function expects user_id
)

# --- Register Actual Tools --- 

semantic_search_tool = ToolDefinition(
    name="semantic_search",
    description="Performs semantic similarity search over ingested documents in the user's knowledge base.",
    parameters=[
        ToolParameter(name="query", type="string", description="The natural language query to search for.", required=True),
    ],
    execute_func=_execute_semantic_search_wrapper, # Use the wrapper
    mcp_requirements=["user_id"], # Indicate user_id is needed via MCP
    expects_context=["user_id"] # Explicitly state the wrapper needs user_id
)

web_search_tool = ToolDefinition(
    name="web_search",
    description="Performs a web search using Google Custom Search engine.",
    parameters=[
        ToolParameter(name="query", type="string", description="The search query.", required=True),
    ],
    execute_func=_execute_web_search_wrapper, # Use the wrapper
    # google_search fetches keys from env/config internally, no MCP needed for keys
    mcp_requirements=[], 
    expects_context=[] 
)

# Register the tools
tool_registry.register_tool(semantic_search_tool)
tool_registry.register_tool(web_search_tool) 
tool_registry.register_tool(delegate_task_tool) # Register the new tool 