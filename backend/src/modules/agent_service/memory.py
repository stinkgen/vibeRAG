"""Agent Memory Management.

Handles storing, retrieving, and summarizing agent memories using Postgres for metadata
and Milvus for vector embeddings.
"""
import logging
import json
from typing import Optional, List, Dict, Any
import re # Import re for summarize_and_store_memory parsing
import asyncio # Added asyncio for generate_with_provider

from sqlalchemy.orm import Session
from sqlalchemy import select, delete # Needed for retrieval/deletion
from sentence_transformers import SentenceTransformer # Import sentence-transformers
from src.modules.config.config import CONFIG # Import config

from src.modules.auth.database import AgentMemory, AgentMemoryCreate # Import DB/Pydantic models
# Import Milvus ops
from src.modules.vector_store import milvus_ops

# Import generation function
from src.modules.generation.generate import generate_with_provider, GenerationError

logger = logging.getLogger(__name__)

# --- Embedding Model Loading (Simple global instance) --- 
# Consider more robust loading/caching if needed (e.g., FastAPI lifespan events)
embedding_model = None
embedding_model_name = None
try:
    if CONFIG.embedding.model_name:
        embedding_model_name = CONFIG.embedding.model_name
        logger.info(f"Loading embedding model: {embedding_model_name} onto device: {CONFIG.embedding.device}")
        embedding_model = SentenceTransformer(embedding_model_name, device=CONFIG.embedding.device)
        # Quick check
        if embedding_model.get_sentence_embedding_dimension() != CONFIG.embedding.embedding_dim:
            logger.warning(f"Model '{embedding_model_name}' dim ({embedding_model.get_sentence_embedding_dimension()}) != CONFIG dim ({CONFIG.embedding.embedding_dim}). Using model dim.")
            # Potentially update config or raise error? For now, just log.
            # CONFIG.embedding.embedding_dim = embedding_model.get_sentence_embedding_dimension()
    else:
        logger.warning("No embedding model specified in config. Memory retrieval based on similarity will not work.")
except Exception as e:
    logger.error(f"Failed to load embedding model '{embedding_model_name}': {e}", exc_info=True)
    embedding_model = None # Ensure it's None if loading failed

# --- Helper to initialize Milvus collection (call during startup?) ---
# async def initialize_agent_memory_store():
#     """Ensures the Milvus collection for agent memories exists.""" <-- REMOVE THIS FUNCTION
#     try:
#         logger.info("Initializing agent memory Milvus collection...")
#         await milvus_ops.init_agent_memory_collection()
#         logger.info("Agent memory Milvus collection initialization complete.")
#     except Exception as e:
#         logger.error(f"Failed to initialize agent memory Milvus collection: {e}", exc_info=True)
#         # Decide if this should prevent startup?

def generate_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """Generates embeddings for a list of texts using the configured model."""
    if not embedding_model:
        logger.warning("Embedding model not loaded. Cannot generate embeddings.")
        return None
    if not texts:
        return []
        
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        # Use model's encode method
        embeddings = embedding_model.encode(
            texts, 
            batch_size=CONFIG.embedding.batch_size, 
            show_progress_bar=False # Or True for debugging
        )
        logger.info(f"Embedding generation complete.")
        return embeddings.tolist() # Convert numpy arrays to lists
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        return None

# --- Long-Term Memory Storage --- 

async def store_memory(
    db: Session,
    agent_id: int,
    memory_type: str,
    content: str,
    importance: Optional[float] = 0.5,
    related_ids: Optional[List[int]] = None,
    embedding: Optional[List[float]] = None
) -> Optional[AgentMemory]:
    """Stores memory metadata in Postgres and the embedding in Milvus.
    Returns the created AgentMemory object on success, None on failure.
    """
    logger.info(f"Storing {memory_type} memory for agent {agent_id}. Importance: {importance}. Embedding provided: {embedding is not None}")
    related_ids_str = json.dumps(related_ids) if related_ids else None
    
    db_memory = None
    try:
        # 1. Create object, add to session, flush to get ID
        db_memory = AgentMemory(
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            related_memory_ids=related_ids_str
        )
        db.add(db_memory)
        db.flush() 
        postgres_id = db_memory.id 
        if postgres_id is None: # Should not happen after flush if PK is sequence
             raise Exception("Failed to get Postgres ID after flushing memory object.")
        logger.debug(f"Flushed memory metadata to Postgres for agent {agent_id}, got ID {postgres_id}.")
        
        # 2. Store embedding in Milvus if provided
        if embedding is not None:
            memory_to_insert = {
                'postgres_id': postgres_id,
                'agent_id': agent_id,
                'embedding': embedding,
                'memory_type': memory_type,
                'importance': importance
            }
            # Pass agent_id to insert function
            milvus_pks = await milvus_ops.insert_agent_memories(agent_id=agent_id, memories=[memory_to_insert])
            if not milvus_pks:
                 logger.error(f"Milvus insert failed for Postgres memory ID {postgres_id}. Rolling back Postgres insert.")
                 db.rollback()
                 return None # Indicate failure
            else:
                 logger.debug(f"Successfully inserted embedding into Milvus for Postgres memory ID {postgres_id}. Milvus PK: {milvus_pks[0]}")
        
        # 3. Commit Postgres transaction 
        db.commit()
        logger.debug(f"Committed memory metadata to Postgres ID {postgres_id} for agent {agent_id}.")
        
        # 4. Refresh object to get committed state (optional but good practice)
        # db.refresh(db_memory) 
        
        return db_memory # Return the committed object

    except Exception as e:
        logger.error(f"Failed to store memory (Postgres or Milvus) for agent {agent_id}: {e}", exc_info=True)
        db.rollback() # Rollback on any error
        return None # Indicate failure

async def summarize_and_store_memory(
    db: Session,
    agent_id: int,
    scratchpad: List[Dict[str, Any]] # List of {'role': ..., 'content': ...} dicts
):
    """Extracts memories from scratchpad, generates embeddings, stores them, and optionally generates/stores a summary."""
    logger.info(f"Processing memories for agent {agent_id} task completion.")
    if not scratchpad:
        logger.warning(f"Scratchpad is empty for agent {agent_id}, nothing to process.")
        return
        
    # --- Store Episodic Memories (Thoughts/Actions/Observations) --- 
    texts_to_embed = []
    memory_entries_data = [] # Store dicts ready for store_memory
    scratchpad_text_for_summary = ""
    
    for step in scratchpad:
        role = step.get('role')
        content = step.get('content')
        if not content:
            continue
        
        # Add raw step to text for summary prompt
        scratchpad_text_for_summary += f"{role.upper()}:\n{content}\n\n"
            
        # Extract thought and action separately from assistant response
        if role == 'assistant': 
            thought_match = re.search(r"Thought: (.*?)(?:Action:|$)", content, re.DOTALL)
            action_match = re.search(r"Action: (.*)", content, re.DOTALL)
            
            if thought_match:
                thought_text = thought_match.group(1).strip()
                if thought_text:
                    texts_to_embed.append(thought_text)
                    memory_entries_data.append({"type": "thought", "content": thought_text, "importance": 0.6, "agent_id": agent_id})
            
            if action_match:
                action_text = action_match.group(1).strip()
                if action_text:
                    texts_to_embed.append(action_text)
                    memory_entries_data.append({"type": "action", "content": action_text, "importance": 0.7, "agent_id": agent_id})
            # If no clear thought/action, store raw assistant response? (Maybe not useful as separate memory)
            # elif not thought_match:
            #     texts_to_embed.append(content)
            #     memory_entries_data.append({"type": "assistant_raw", "content": content, "importance": 0.5, "agent_id": agent_id})
                
        # Extract observation
        elif role == 'system' and content.startswith("Observation:"): 
            observation_text = content.replace("Observation:", "", 1).strip()
            texts_to_embed.append(observation_text)
            is_error = "\"error\":" in observation_text.lower()
            memory_entries_data.append({"type": "observation", "content": observation_text, "importance": 0.4 if is_error else 0.6, "agent_id": agent_id})

    # Generate embeddings in batch for episodic memories
    episodic_embeddings = generate_embeddings(texts_to_embed)
    
    if episodic_embeddings is None:
        logger.error(f"Failed to generate embeddings for agent {agent_id} episodic memories. Storing without embeddings.")
        episodic_embeddings = [None] * len(memory_entries_data)
    elif len(episodic_embeddings) != len(memory_entries_data):
         logger.error(f"Mismatch between number of embeddings ({len(episodic_embeddings)}) and memory entries ({len(memory_entries_data)}) for agent {agent_id}. Skipping Milvus insert for episodic.")
         episodic_embeddings = [None] * len(memory_entries_data)
        
    # Store each episodic memory entry
    for i, entry_data in enumerate(memory_entries_data):
        embedding = episodic_embeddings[i]
        await store_memory(
            db=db, 
            agent_id=entry_data["agent_id"], 
            memory_type=entry_data["type"],
            content=entry_data["content"], 
            importance=entry_data["importance"],
            embedding=embedding
        )
        
    # --- Generate and Store Summary Memory (Optional) --- 
    if CONFIG.agent.summarization_enabled and scratchpad_text_for_summary:
        logger.info(f"Generating summary memory for agent {agent_id}.")
        summary_content = ""
        try:
            # Format prompt
            summary_prompt_text = CONFIG.agent.summarization_prompt_template.format(
                scratchpad_content=scratchpad_text_for_summary
            )
            summary_prompt_messages = [{"role": "user", "content": summary_prompt_text}]
            
            # Determine LLM config for summarization
            provider = CONFIG.agent.summarization_llm_provider or CONFIG.chat.provider
            model = CONFIG.agent.summarization_llm_model or CONFIG.chat.model
            temperature = CONFIG.chat.temperature # Use default chat temp for summary
            
            logger.debug(f"Calling LLM ({provider}/{model}) for summarization for agent {agent_id}.")
            # Generate summary (using async generate_with_provider)
            response_gen = generate_with_provider(
                messages=summary_prompt_messages,
                model=model,
                provider=provider,
                temperature=temperature
            )
            async for chunk in response_gen:
                if chunk: summary_content += chunk
            
            summary_content = summary_content.strip()
            logger.info(f"Generated summary for agent {agent_id}: {summary_content[:100]}...")
            
            if summary_content:
                # Generate embedding for summary if enabled
                summary_embedding = None
                if CONFIG.agent.embed_summaries:
                    summary_embeddings = generate_embeddings([summary_content])
                    if summary_embeddings and len(summary_embeddings) > 0:
                        summary_embedding = summary_embeddings[0]
                    else:
                         logger.warning(f"Failed to generate embedding for summary for agent {agent_id}. Storing without.")
                
                # Store the summary memory
                await store_memory(
                    db=db,
                    agent_id=agent_id,
                    memory_type="summary", # Or reflection?
                    content=summary_content,
                    importance=0.8, # Summaries are generally important
                    embedding=summary_embedding
                )
            else:
                logger.warning(f"LLM returned empty summary for agent {agent_id}.")

        except GenerationError as gen_err:
             logger.error(f"LLM generation failed during summarization for agent {agent_id}: {gen_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during summarization for agent {agent_id}: {e}", exc_info=True)
            # Don't let summarization failure prevent episodic storage (already done)

# --- Memory Retrieval --- 

async def retrieve_relevant_memories(
    db: Session,
    agent_id: int,
    query_text: str,
    limit: int = 5
) -> List[AgentMemory]:
    """Retrieves relevant memories by searching Milvus and fetching from Postgres."""
    
    if not embedding_model:
        logger.warning(f"Embedding model not loaded. Cannot retrieve memories for agent {agent_id}.")
        return []
        
    # 1. Generate query embedding
    query_embedding = generate_embeddings([query_text])
    if query_embedding is None or not query_embedding:
        logger.error(f"Failed to generate query embedding. Cannot retrieve memories for agent {agent_id}.")
        return []
    query_vector = query_embedding[0]
    
    try:
        # 2. Search Milvus for relevant Postgres IDs
        logger.info(f"Searching Milvus memories for agent {agent_id}...")
        milvus_results = await milvus_ops.search_agent_memories(
            agent_id=agent_id,
            query_vector=query_vector,
            limit=limit
            # Add min_importance filter later if needed
        )
        
        postgres_ids = [res['postgres_id'] for res in milvus_results if res.get('postgres_id') is not None]
        
        if not postgres_ids:
            logger.info(f"No relevant memories found in Milvus for agent {agent_id}.")
            return []
            
        logger.info(f"Found {len(postgres_ids)} relevant memory IDs in Milvus: {postgres_ids}. Fetching from Postgres...")
        
        # 3. Fetch full memory objects from Postgres using the IDs
        # Use select statement for clarity with IN clause
        stmt = select(AgentMemory).where(AgentMemory.id.in_(postgres_ids))
        # Maintain order from Milvus results? Requires mapping distances back.
        # For now, just fetch and let DB order (or lack thereof)
        results = db.execute(stmt).scalars().all()
        
        # Optional: Re-order results based on Milvus score? Requires more complex joining/mapping.
        # score_map = {res['postgres_id']: res['score'] for res in milvus_results}
        # sorted_results = sorted(results, key=lambda mem: score_map.get(mem.id, float('inf')))
        
        logger.info(f"Retrieved {len(results)} full memory objects from Postgres for agent {agent_id}.")
        return results
        
    except Exception as e:
        logger.error(f"Failed to retrieve memories (Milvus search or PG fetch) for agent {agent_id}: {e}", exc_info=True)
        # Don't rollback here as it's a read operation
        return []

# --- Manual Memory Management --- 

async def delete_memory_by_id(db: Session, memory_id: int, agent_id: int) -> bool:
    """Deletes a specific memory from Postgres and its corresponding embedding from Milvus."""
    logger.warning(f"Attempting to delete memory ID {memory_id} for agent {agent_id}.")
    
    # 1. Delete from Postgres
    try:
        stmt = delete(AgentMemory).where(AgentMemory.id == memory_id, AgentMemory.agent_id == agent_id)
        result = db.execute(stmt)
        db.flush() # Apply deletion
        
        if result.rowcount == 0:
            logger.warning(f"Memory ID {memory_id} not found in Postgres for agent {agent_id}. Nothing to delete.")
            db.rollback() # Rollback flush
            return False
            
        logger.info(f"Deleted memory metadata ID {memory_id} from Postgres.")
        
        # 2. Delete from Milvus using the Postgres ID
        # Pass agent_id to delete function
        milvus_deleted = await milvus_ops.delete_agent_memory_by_postgres_id(agent_id=agent_id, postgres_ids=[memory_id])
        if not milvus_deleted:
            # Log error, but should we rollback Postgres? 
            # Data is inconsistent, but PG delete succeeded. Risky to rollback.
            logger.error(f"Failed to delete corresponding embedding from Milvus for Postgres memory ID {memory_id}. Data may be inconsistent.")
            # Proceed with commit for PG delete? Yes, let's commit PG.
        
        # 3. Commit Postgres deletion
        db.commit()
        logger.info(f"Committed deletion for memory ID {memory_id}.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete memory ID {memory_id} for agent {agent_id}: {e}", exc_info=True)
        db.rollback()
        return False 