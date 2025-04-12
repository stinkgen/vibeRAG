# --- tasks.py ---
import logging
import json
import traceback # For detailed error logging
from typing import Dict, Any # Import necessary types

from src.celery_app import celery_app # Import the celery app instance
from src.modules.auth.database import SessionLocal, AgentTask as AgentTaskModel, User, Agent
from src.modules.agent_service.models import AgentTask as AgentTaskSchema, AgentOutput # Pydantic schemas
from src.modules.agent_service import runtime # Import the original runtime logic
from src.modules.auth.auth import get_user_by_id # Correct import path
# Import the connection manager
from src.modules.agent_service.api import connection_manager 
import asyncio # For running async broadcast in sync task if needed

logger = logging.getLogger(__name__)

# Helper function to run async broadcast from sync task
# (Celery tasks can be async now, so maybe not needed if task is async)
# async def broadcast_async(user_id: int, message: str):
#    await connection_manager.broadcast_to_user(message, user_id)

@celery_app.task(bind=True, name='agent_service.execute_agent_task')
async def execute_agent_task(self, agent_task_data: Dict[str, Any]):
    """Celery task to execute the agent run logic asynchronously."""
    
    task_db_id = agent_task_data.get('id') # Get the DB ID of the AgentTask
    if not task_db_id:
        logger.error("Cannot execute task: AgentTask database ID missing from task data.")
        # How to report failure back? Celery state?
        self.update_state(state='FAILURE', meta={'error': 'Missing AgentTask ID'})
        return # Or raise Ignore()?

    logger.info(f"Worker received task execution request for AgentTask ID: {task_db_id}")
    db = SessionLocal()
    try:
        # --- Fetch Task and User from DB --- 
        task = db.query(AgentTaskModel).filter(AgentTaskModel.id == task_db_id).first()
        if not task:
            logger.error(f"AgentTask ID {task_db_id} not found in database.")
            self.update_state(state='FAILURE', meta={'error': 'AgentTask not found'})
            return
            
        owner_user_id = task.user_id # Store user ID for broadcast
        owner_user = db.query(User).filter(User.id == owner_user_id).first()
        if not owner_user:
            logger.error(f"Owner User ID {task.user_id} for AgentTask ID {task_db_id} not found.")
            task.status = "failed"
            task.error_message = "Owner user not found"
            db.commit()
            self.update_state(state='FAILURE', meta={'error': 'Owner user not found'})
            return

        # --- Update Task Status to Running --- 
        task.status = "running"
        db.commit()
        logger.info(f"AgentTask {task_db_id} status set to running.")
        # Celery task state update (optional)
        self.update_state(state='PROGRESS')

        # --- Prepare Schema for Runtime --- 
        # Recreate the Pydantic schema from the DB model data if needed
        # Note: runtime.run_agent_task now needs to accept the DB model or Pydantic schema?
        # Let's assume runtime is refactored to accept the DB model directly.
        
        # --- Execute Original Runtime Logic --- 
        logger.info(f"Executing runtime logic for AgentTask ID: {task_db_id}...")
        # IMPORTANT: Refactor runtime.run_agent_task to:
        # 1. Accept the AgentTask DB model and User DB model.
        # 2. Perform its logic.
        # 3. Return the final AgentOutput (Pydantic model) OR raise specific exceptions.
        # 4. NOT commit DB changes itself (handled here).
        try:
            # We pass the DB session from the task context
            final_output: AgentOutput = await runtime.run_agent_task_logic(task=task, owner=owner_user, db=db)
            
            # --- Update Task Status to Completed --- 
            task.status = "completed"
            task.result_data = final_output.model_dump() # Store result as JSON
            task.error_message = None
            db.commit()
            logger.info(f"AgentTask {task_db_id} completed successfully.")
            # Return result for Celery backend (optional)
            final_status = "completed"
            result_payload = task.result_data
            return task.result_data

        except Exception as e:
            # --- Update Task Status to Failed --- 
            logger.error(f"AgentTask {task_db_id} failed during execution: {e}", exc_info=True)
            task.status = "failed"
            task.error_message = str(e)
            # Optionally store more details like traceback?
            task.result_data = None # Clear any partial results
            db.commit()
            # Update Celery state
            self.update_state(state='FAILURE', meta={'error': str(e), 'traceback': traceback.format_exc()})
            # Propagate exception for Celery retry/error handling? 
            # raise # If we want Celery to record it as failed beyond DB status
            final_status = "failed"
            result_payload = {"error": task.error_message}
            return { "error": str(e) } # Return error info

    except Exception as outer_e:
        # Catch errors during setup (DB fetch, status update)
        logger.error(f"Critical error processing AgentTask ID {task_db_id}: {outer_e}", exc_info=True)
        # Attempt to mark DB task as failed if possible
        try:
            task = db.query(AgentTaskModel).filter(AgentTaskModel.id == task_db_id).first()
            if task and task.status != "failed":
                task.status = "failed"
                task.error_message = f"Worker setup error: {str(outer_e)}"
                db.commit()
        except Exception as db_err:
             logger.error(f"Failed to update task status to failed after worker error: {db_err}")
        # Update Celery state
        self.update_state(state='FAILURE', meta={'error': f"Worker setup error: {str(outer_e)}", 'traceback': traceback.format_exc()})
        # raise # Propagate critical error

    finally:
        db.close()
        logger.debug(f"Database session closed for AgentTask ID: {task_db_id}.") 
        
        # --- Broadcast Result via WebSocket --- 
        if owner_user_id is not None:
            logger.info(f"Broadcasting final status ({final_status}) for task {task_db_id} to user {owner_user_id}.")
            # Construct message payload
            broadcast_message = {
                "type": "task_update",
                "task_db_id": task_db_id,
                "status": final_status,
                "payload": result_payload # Contains result_data or error
            }
            try:
                # Since this task is async, we can directly await the broadcast
                await connection_manager.broadcast_to_user(json.dumps(broadcast_message), owner_user_id)
                logger.debug(f"Broadcast message sent for task {task_db_id}.")
            except Exception as ws_err:
                logger.error(f"Failed to broadcast task update via WebSocket for task {task_db_id}: {ws_err}", exc_info=True)
        else:
             logger.warning(f"Cannot broadcast task {task_db_id} result: owner_user_id not found.") 