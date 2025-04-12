"""Placeholder for the Agent Runtime logic.

This module will contain the core agent execution loop (Observe-Plan-Act),
state management during execution, and interaction with tools/memory.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional
import json
import re
import asyncio
import uuid # For task IDs

from sqlalchemy.orm import Session

from src.modules.agent_service.models import AgentTask, AgentOutput
from src.modules.agent_service.manager import get_agent_definition, get_agent_capabilities
from src.modules.agent_service.tools import tool_registry, ToolDefinition
from src.modules.agent_service.logging import log_agent_activity
from src.modules.generation.generate import generate_with_provider, GenerationError # Assuming non-streaming call is ok, or need adaptation
from src.modules.auth.database import Agent, get_db, User, AgentTask as AgentTaskModel
from src.modules.config.config import CONFIG
from src.modules.agent_service.memory import summarize_and_store_memory, retrieve_relevant_memories, AgentMemory

logger = logging.getLogger(__name__)


def _construct_react_prompt(
    agent: Agent,
    tools: List[ToolDefinition],
    task_goal: str,
    retrieved_memories: List[AgentMemory] = None,
    previous_steps: List[str] = None
) -> List[Dict[str, str]]:
    """Constructs the prompt messages for the ReAct planning step, including retrieved memories."""
    
    # Base system prompt incorporating persona and goals
    system_prompt = f"You are {agent.name}.\n"
    if agent.persona:
        system_prompt += f"Your persona: {agent.persona}\n"
    if agent.goals:
        system_prompt += f"Your overall goals: {agent.goals}\n"
    if agent.base_prompt:
         system_prompt += f"{agent.base_prompt}\n"
         
    # Add retrieved memories if available
    if retrieved_memories:
        system_prompt += "\n\nRelevant Past Memories (Use these for context):\n"
        for mem in retrieved_memories:
            # Format memory for prompt (consider type, importance, timestamp?)
            system_prompt += f"- [{mem.memory_type.upper()}] {mem.content}\n"
            
    system_prompt += ("\nYou must respond using the ReAct (Reasoning + Action) format. "
                      "First, provide your reasoning (Thought: ...). "
                      "Then, specify the action to take (Action: ...). "
                      "The ONLY valid actions are using one of the available tools or providing a final answer. "
                      "Format tool actions as: Action: tool_name(param1=\"value1\", param2=value2). "
                      "Format final answers as: Action: final_answer(answer=\"Your final response here.\")"
                      "\nAvailable Tools:")

    # Add tool descriptions
    for tool in tools:
        params_str = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters])
        system_prompt += f"\n- {tool.name}({params_str}): {tool.description}"
        
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add previous steps if any (for future multi-step implementation)
    if previous_steps:
        for step in previous_steps:
            # Assuming steps alternate between user/assistant or action/observation
            # This part needs refinement for multi-step
            messages.append({"role": "assistant", "content": step}) 
            
    # Add the current task goal
    messages.append({"role": "user", "content": f"Current task: {task_goal}"})
    
    return messages

def _parse_action(llm_response: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parses the Action: line from the LLM response."""
    action_match = re.search(r"Action: (\w+)\((.*)\)", llm_response, re.DOTALL)
    if action_match:
        tool_name = action_match.group(1).strip()
        params_str = action_match.group(2).strip()
        
        # Basic parsing of key=value pairs (handle strings carefully)
        # This is fragile and needs improvement for robust parsing
        params = {}
        try:
            # Attempt to parse as keyword arguments (might fail with complex strings)
            # A more robust parser (like ast.literal_eval on a dict string) might be needed
            # For now, simple regex for key="value" or key=value
            for match in re.finditer(r"(\w+)\s*=\s*(?:\"(.*?)\"|([^,\)]+))", params_str):
                key = match.group(1)
                value = match.group(2) if match.group(2) is not None else match.group(3)
                # Basic type conversion attempt (needs refinement based on tool schema)
                try:
                    # Attempt JSON decoding for potential lists/dicts/bools/numbers
                    params[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Fallback to string if not valid JSON
                    params[key] = value.strip() # Remove leading/trailing whitespace
                    
            return tool_name, params
        except Exception as e:
            logger.error(f"Failed to parse action parameters '{params_str}': {e}")
            return tool_name, None # Return tool name but indicate param failure
            
    return None, None # No valid action found

async def run_agent_task_logic(task: AgentTaskModel, owner: User, db: Session) -> AgentOutput:
    """Core logic for running an agent task, potentially over multiple ReAct steps.
    Accepts DB models and session, performs logic, returns AgentOutput Pydantic model.
    Does NOT commit changes to the database.
    """
    task_log_id = task.celery_task_id or str(uuid.uuid4()) # Use celery ID if available for logging context
    max_steps = CONFIG.agent.max_steps
    agent_id = task.agent_id
    
    # Logging now happens within the Celery task or API endpoint before calling this
    # log_agent_activity(db, agent_id=agent_id, level="INFO", message=f"Task {task_log_id} started...")
    
    scratchpad: List[Dict[str, Any]] = [] # Initialize scratchpad for this execution
    step_counter = 0
    done = False
    final_output_data = None # Store the final result data (dict)
    final_status = "failed" # Default to failed unless explicitly completed
    llm_response_text = "" 
    error_message = None # Store potential error message

    try:
        # --- Load Agent Definition --- 
        agent_def = db.query(Agent).filter(Agent.id == agent_id).first() # Use passed DB session
        if not agent_def:
            raise ValueError(f"Agent definition {agent_id} not found.") # Raise error
        if not agent_def.is_active:
             raise ValueError(f"Agent {agent_id} is not active.")

        # --- Get Agent Capabilities --- 
        try:
            allowed_tool_names = agent_manager.get_agent_capabilities(db, agent_id=agent_def.id)
            # Log this maybe? But logging should happen outside this pure logic function
        except Exception as cap_err:
            logger.error(f"Failed to get capabilities for agent {agent_def.id}: {cap_err}", exc_info=True)
            # Don't log to DB here
            allowed_tool_names = None # Fallback to allowing all tools
            
        # --- Filter Available Tools --- 
        all_tools = tool_registry.list_tools()
        if allowed_tool_names is not None:
            if not allowed_tool_names:
                available_tools = []
            else:
                available_tools = [tool for tool in all_tools if tool.name in allowed_tool_names]
        else:
            available_tools = all_tools
            
        # --- ReAct Loop --- 
        while not done and step_counter < max_steps:
            step_counter += 1
            # Log step start? Outside function.
            
            # --- Retrieve Relevant Memories --- 
            current_goal_for_memory = task.goal # Use goal from task model
            if scratchpad:
                current_goal_for_memory += "\nContext: " + scratchpad[-1]['content']
            retrieved_memories = []
            try:
                retrieved_memories = await retrieve_relevant_memories(
                    db=db,
                    agent_id=agent_def.id,
                    query_text=current_goal_for_memory,
                    limit=CONFIG.agent.memory_retrieval_limit
                )
            except Exception as mem_err:
                logger.error(f"Failed to retrieve memories during task {task.id}: {mem_err}")
                # Log DB outside
                 
            # --- Construct Prompt --- 
            current_goal = task.goal # Use task goal
            # Use input_data if needed? How was agent_context passed before?
            # Let's assume input_data holds the original context if any.
            if task.input_data and isinstance(task.input_data, dict) and task.input_data.get('agent_context') and step_counter == 1:
                 current_goal += f"\n\nAdditional Context Provided:\n{task.input_data.get('agent_context')}"
                 
            scratchpad_content_for_prompt = [item['content'] for item in scratchpad]
            prompt_messages = _construct_react_prompt(
                agent_def, 
                available_tools,
                current_goal, 
                retrieved_memories=retrieved_memories, 
                previous_steps=scratchpad_content_for_prompt
            )

            # --- Call LLM for Planning --- 
            llm_response_text = ""
            try:
                provider = agent_def.llm_provider if agent_def.llm_provider else CONFIG.chat.provider
                model = agent_def.llm_model if agent_def.llm_model else CONFIG.chat.model
                temperature = CONFIG.chat.temperature
                
                generation_args = {
                    "messages": prompt_messages,
                    "model": model,
                    "provider": provider,
                    "temperature": temperature
                }
                response_gen = generate_with_provider(**generation_args)
                async for chunk in response_gen:
                    if chunk: llm_response_text += chunk

            except Exception as llm_err:
                 logger.error(f"LLM planning call failed for task {task.id}: {llm_err}", exc_info=True)
                 # Raise a specific exception to be caught by the task runner
                 raise GenerationError(f"LLM Planning failed at step {step_counter}: {llm_err}") from llm_err
                 
            if not llm_response_text:
                 logger.error(f"LLM planning response was empty for task {task.id} at step {step_counter}.")
                 raise ValueError(f"LLM returned empty response at step {step_counter}")

            scratchpad.append({"role": "assistant", "content": llm_response_text})

            # --- Parse Action --- 
            tool_name, tool_params = _parse_action(llm_response_text)
            # Log planning results outside

            if tool_name == "final_answer":
                final_answer = tool_params.get('answer', "No final answer provided.") if tool_params else "Invalid final_answer format."
                # Log final answer outside
                final_output_data = {"result": final_answer} # Structure result
                final_status = "completed"
                done = True
            elif tool_name:
                # Log tool call outside
                if tool_name not in [t.name for t in available_tools]:
                     observation_content = f"Error: Tool '{tool_name}' is not available or not allowed for this agent."
                     # Log error outside
                     raise ValueError(observation_content)
                else:
                    try:
                        # Execute tool
                        tool_result = await tool_registry.execute_tool(
                            name=tool_name,
                            params=tool_params or {},
                            mcp_context_identifiers={
                                "user_id": owner.id, 
                                "agent_id": agent_def.id # Pass agent ID for context
                            }
                        )
                        # Log tool result outside
                        # Format result for observation
                        if isinstance(tool_result, dict) and 'error' in tool_result:
                             observation_content = f"Error executing tool '{tool_name}': {tool_result['error']}"
                        else:
                             observation_content = json.dumps(tool_result, indent=2) # Pretty print result
                             
                    except Exception as tool_err:
                        logger.error(f"Error executing tool '{tool_name}' for task {task.id}: {tool_err}", exc_info=True)
                        observation_content = f"Error executing tool '{tool_name}': {tool_err}"
                        # Log error outside
                        # Optionally raise to fail the whole task? Or just add error observation?
                        # Let's add error observation for now.
                        
                scratchpad.append({"role": "system", "content": f"Observation: {observation_content}"})
                # Log observation outside
            else:
                # No valid action found, treat as failure or maybe ask for clarification?
                logger.error(f"Could not parse valid action from LLM response for task {task.id} at step {step_counter}. Response: {llm_response_text}")
                # Log error outside
                raise ValueError(f"Could not parse valid action at step {step_counter}. Last LLM response: {llm_response_text}")

        # --- End of Loop --- 

        if not done:
            logger.warning(f"Task {task.id} reached max steps ({max_steps}) without completion.")
            final_status = "failed_max_steps"
            error_message = f"Agent reached maximum steps ({max_steps}) without providing a final answer."
            final_output_data = {"error": error_message}

    except Exception as e:
        # Catch any unexpected errors during the loop or setup
        logger.error(f"Unhandled exception during agent task logic for task {task.id}: {e}", exc_info=True)
        final_status = "failed_unexpected"
        error_message = f"Unexpected error during execution: {str(e)}"
        final_output_data = {"error": error_message}
        # We need to return an AgentOutput even on failure

    # --- Memory Processing --- 
    # Call memory summarization/storage outside this function if possible? 
    # Or keep it here but ensure it doesn't commit?
    # For now, keep it here and pass the db session.
    try:
        await summarize_and_store_memory(db=db, agent_id=agent_id, scratchpad=scratchpad)
    except Exception as mem_e:
         logger.error(f"Error during memory processing for task {task.id} (task result still preserved): {mem_e}", exc_info=True)
         # Don't fail the task just because memory storage failed

    # --- Construct Final Output --- 
    output = AgentOutput(
        agent_id=agent_id,
        task_id=str(task.id), # Use DB task ID
        status=final_status,
        output=final_output_data or {}, # Ensure output is a dict
        scratchpad=scratchpad,
        error_message=error_message
    )
    return output

# Remove original run_agent_task or keep as a thin wrapper for testing?
# async def run_agent_task(task: AgentTask, owner: User) -> AgentOutput:
#    db = next(get_db())
#    try:
#        # Need to fetch/create the DB task record first?
#        # This old signature is incompatible now.
#        pass 
#    finally:
#        db.close()


# Add other runtime helper functions/classes here
# import asyncio # Already imported
import traceback # For logging errors
