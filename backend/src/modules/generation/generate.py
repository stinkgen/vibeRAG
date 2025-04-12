"""Core generation module for interacting with LLMs.

This module provides the base functionality for generating text using different
LLM providers and configurations.
"""

import logging
from typing import Dict, List, Any, TypedDict, Literal, Optional, Union, Generator, AsyncGenerator, Iterator
import requests
import json
import aiohttp
import asyncio
import os
from openai import AsyncOpenAI
import openai
import tiktoken # Import tiktoken
from fastapi import WebSocket, WebSocketDisconnect, status as fastapi_status, APIRouter, Depends, HTTPException, Request
from starlette.websockets import WebSocketState
from sqlalchemy.orm import Session
import uuid

from src.modules.config.config import CONFIG
from src.modules.generation.exceptions import GenerationError
from src.modules.auth.database import get_db, SessionLocal, User
from src.modules.auth.auth import decode_access_token, get_user_by_id
from src.modules.chat.history import (
    create_chat_session,
    get_chat_session,
    add_chat_message,
    get_session_messages,
    get_user_chat_sessions
)
from src.modules.retrieval.search import semantic_search # Import semantic_search
from src.modules.retrieval.search import google_search # Import google_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client lazily
client = None

def get_openai_client() -> AsyncOpenAI:
    """Get or create OpenAI client with proper configuration."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY") or CONFIG.openai.api_key
        if not api_key:
            # Allow skipping if only using Ollama
            logger.warning("OpenAI API key not found in environment or config. OpenAI provider will not work.")
            # Return a dummy or raise specific error if needed later
            return None # Or raise ValueError("OpenAI API key not found...")
        
        # Basic check if it looks like an OpenAI key 
        if api_key.startswith("sk-") and len(api_key) > 20: 
            try:
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=os.getenv("OPENAI_API_BASE", CONFIG.openai.base_url)
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                client = None # Ensure client remains None if init fails
                raise ValueError(f"Failed to initialize OpenAI client: {e}")
        # Handle other key types (like Azure) if necessary in the future
        # else:
            # logger.warning("API key does not look like a standard OpenAI key. Assuming other configuration (e.g., Azure).")
            # client = AsyncOpenAI(...) # Add Azure or other config logic here
            
    # Return client only if successfully initialized
    if client is None:
         raise ValueError("OpenAI client could not be initialized. Check API key and configuration.")
    return client

# Type definitions
class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class WebResult(TypedDict):
    title: str
    link: str
    snippet: str

class ChunkMetadata(TypedDict):
    filename: str
    page: Union[int, str]

class Chunk(TypedDict):
    text: str
    metadata: ChunkMetadata

ProviderType = Literal["openai", "ollama"]

# Model context limits (add more as needed)
MODEL_CONTEXT_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385, # Often alias for the 16k version
    "gpt-4.5-preview": 128000 # Add assumed limit for this model
}

def get_tokenizer_for_model(model_name: str) -> tiktoken.Encoding:
    """Get the appropriate tokenizer for a given OpenAI model name."""
    try:
        # Default to cl100k_base for most modern models (GPT-3.5+, GPT-4)
        return tiktoken.get_encoding("cl100k_base")
    except KeyError:
        logger.warning(f"Tokenizer cl100k_base not found. Falling back to p50k_base.")
        try:
            return tiktoken.get_encoding("p50k_base") # Older models
        except KeyError:
             logger.error("Could not load any tiktoken tokenizer.")
             raise

def count_message_tokens(messages: List[Dict[str, str]], model: str) -> int:
    """Counts the number of tokens required for a list of messages for a specific model."""
    tokenizer = get_tokenizer_for_model(model)
    num_tokens = 0
    # Adapted from OpenAI cookbook:
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    if model in (
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        ):
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model: 
        # Assuming future gpt-3.5 counts like 0613
        logger.warning("count_message_tokens: gpt-3.5-turbo model version not found. Using cl100k_base tokens_per_message/name settings.")
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-4" in model:
        # Assuming future gpt-4 counts like 0613
        logger.warning("count_message_tokens: gpt-4 model version not found. Using cl100k_base tokens_per_message/name settings.")
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""count_message_tokens() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value: # Ensure value is not None or empty string before encoding
                num_tokens += len(tokenizer.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

async def _call_openai_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float
) -> AsyncGenerator[str, None]:
    """Calls OpenAI Chat Completions API and streams the response, dynamically adjusting max_tokens."""
    log_prefix = "[_call_openai_chat_completion]"
    try:
        openai_client = get_openai_client()
        if not openai_client:
             logger.error(f"{log_prefix} OpenAI client not available. Cannot make API call.")
             raise GenerationError("OpenAI client not available. Cannot make API call.")

        # --- Dynamic max_tokens calculation ---
        model_context_limit = MODEL_CONTEXT_LIMITS.get(model, 8192) # Default to 8k if model unknown
        if model not in MODEL_CONTEXT_LIMITS:
            logger.warning(f"{log_prefix} Context limit for model '{model}' unknown, defaulting to {model_context_limit}. Accuracy not guaranteed.")
        
        try:
            input_tokens = count_message_tokens(messages, model)
        except Exception as e_count:
             logger.error(f"{log_prefix} Failed to count input tokens for model '{model}': {e_count}. Proceeding without dynamic max_tokens.", exc_info=True)
             # Fallback: use the configured max_tokens directly, risking context length error
             requested_max_tokens = CONFIG.openai.max_tokens 
             input_tokens = -1 # Indicate failure
        
        calculated_max_completion_tokens = -1 # Initialize
        if input_tokens != -1: # Only calculate if token counting succeeded
            buffer = 100 # Small buffer for safety
            available_for_completion = model_context_limit - input_tokens - buffer
            
            if available_for_completion <= 0:
                logger.error(f"{log_prefix} Input tokens ({input_tokens}) exceed model context limit ({model_context_limit}) even with buffer. Cannot generate.")
                raise GenerationError(f"Input ({input_tokens} tokens) too large for model '{model}' ({model_context_limit} limit). Please shorten your query or context.")
            
            # Use the smaller of available space or the configured desired max_tokens
            calculated_max_completion_tokens = min(available_for_completion, CONFIG.openai.max_tokens)
            requested_max_tokens = calculated_max_completion_tokens
            logger.info(f"{log_prefix} Model: {model}, Context Limit: {model_context_limit}, Input Tokens: {input_tokens}, Config Max: {CONFIG.openai.max_tokens}, Available: {available_for_completion}, Requested Max Tokens: {requested_max_tokens}")
        else:
             # Fallback if token counting failed
             requested_max_tokens = CONFIG.openai.max_tokens 
             logger.warning(f"{log_prefix} Using configured max_tokens ({requested_max_tokens}) due to token counting failure.")
        # --- End dynamic calculation ---
             
        logger.info(f"{log_prefix} Attempting OpenAI call. Model: {model}, Temp: {temperature}, MaxTokens: {requested_max_tokens}")
        logger.debug(f"{log_prefix} Messages head: {messages[:2]}... Tail: {messages[-1:] if len(messages) > 2 else ''}")

        logger.info(f"{log_prefix} Creating OpenAI stream...")
        stream = await openai_client.chat.completions.create(
            model=model, # Use the determined model
            messages=messages, # type: ignore 
            temperature=temperature,
            stream=True,
            max_tokens=requested_max_tokens # Use dynamically calculated value
        )
        logger.info(f"{log_prefix} Successfully created OpenAI stream object for model: {model}")

        chunk_count = 0
        logger.info(f"{log_prefix} Starting to iterate through OpenAI stream...")
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                chunk_count += 1
                logger.debug(f"{log_prefix} Received chunk {chunk_count} from OpenAI. Content: '{content[:50]}...'")
                yield content
            # Handle potential finish reason if necessary
            if chunk.choices[0].finish_reason:
                logger.info(f"{log_prefix} Generation finished with reason: {chunk.choices[0].finish_reason}")
        logger.info(f"{log_prefix} Finished iterating OpenAI stream. Total chunks with content: {chunk_count}")

    except Exception as e:
        logger.exception(f"{log_prefix} Error DURING OpenAI stream processing for model '{model}': {e}")
        raise GenerationError(f"Failed to generate response from OpenAI: {e}") from e

async def _call_ollama_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float
) -> AsyncGenerator[str, None]:
    """Calls Ollama /api/chat endpoint and streams the response."""
    logger.info(f"[_call_ollama_chat_completion] Attempting Ollama call. Model: {model}, Temp: {temperature}")
    # logger.debug(f"[_call_ollama_chat_completion] Messages: {messages}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONFIG.ollama.host}{CONFIG.ollama.chat_endpoint}",
                json={
                    "model": model or CONFIG.ollama.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature}
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API request failed with status {response.status}: {error_text}")
                    raise GenerationError(f"Ollama API request failed: {response.status} - {error_text}")

                chunk_count = 0
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            content = chunk.get("message", {}).get("content")
                            if content:
                                chunk_count += 1
                                yield content
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode Ollama JSON chunk: {line}")
                            continue
                        except Exception as e_inner:
                             logger.error(f"Error processing Ollama chunk: {e_inner}")
                logger.info(f"[_call_ollama_chat_completion] Finished streaming {chunk_count} chunks from Ollama for model: {model}")

    except aiohttp.ClientError as e:
        logger.exception(f"[_call_ollama_chat_completion] Connection error during Ollama call for model '{model}': {e}")
        raise GenerationError(f"Failed to connect to Ollama: {e}") from e
    except Exception as e:
        logger.exception(f"[_call_ollama_chat_completion] Error during Ollama stream for model '{model}': {e}")
        raise GenerationError(f"Failed to generate response from Ollama: {e}") from e

# --- Dispatcher --- 

async def generate_with_provider(
    messages: List[Dict[str, str]],
    model: str,
    provider: ProviderType = CONFIG.chat.provider,
    temperature: float = CONFIG.chat.temperature
) -> AsyncGenerator[str, None]:
    """Generate text using the specified provider by dispatching to an adapter."""
    logger.info(f"[generate_with_provider] Dispatching to provider: {provider}, model: {model}")
    actual_model = model or (CONFIG.openai.default_model if provider == "openai" else CONFIG.ollama.model)
    try:
        if provider == "openai":
            async for chunk in _call_openai_chat_completion(messages, actual_model, temperature):
                yield chunk
        elif provider == "ollama":
            # Ollama calls don't need the dynamic token calc for now
            async for chunk in _call_ollama_chat_completion(messages, actual_model, temperature):
                yield chunk
        else:
            logger.error(f"Unsupported provider specified: {provider}")
            raise GenerationError(f"Unsupported provider: {provider}")

    except GenerationError as e: # Catch errors from adapters
        logger.error(f"[generate_with_provider] Generation failed for provider {provider}: {e}")
        raise e # Re-raise the specific error
    except Exception as e:
        logger.exception(f"[generate_with_provider] Unexpected error during generation dispatch: {e}")
        raise GenerationError(f"Unexpected error during generation: {e}") from e

# --- WebSocket Authentication Helper ---
async def get_user_from_token_ws(token: Optional[str]) -> Optional[User]:
    """Validates token and fetches user for WebSocket connection."""
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        if payload is None:
            return None
        user_id = payload.user_id
        if user_id is None:
            return None
        # Use a synchronous session within an async context correctly
        with SessionLocal() as db:
             user = get_user_by_id(db, user_id)
        return user
    except Exception as e: # Catch decoding errors, token expiration etc.
        logger.warning(f"WebSocket token validation failed: {e}")
        return None

# --- Main WebSocket Logic --- 
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for handling chat connections."""
    # Accept connection first
    await websocket.accept()
    
    # --- Implement robust WebSocket Authentication --- #
    user: Optional[User] = None
    db_session_local: Optional[Session] = None
    try:
        # Get token from cookie
        token = websocket.cookies.get("access_token")
        if not token:
            await websocket.close(code=fastapi_status.WS_1008_POLICY_VIOLATION, reason="Missing auth token cookie")
            logger.warning("Chat WS connection failed: Missing auth token cookie.")
            return
        
        # Decode token (doesn't need DB yet)
        token_data = decode_access_token(token)
        if token_data is None or token_data.user_id is None:
            await websocket.close(code=fastapi_status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            logger.warning(f"Chat WS connection failed: Invalid token data {token_data}")
            return

        # Get DB session to fetch user
        db_session_local = SessionLocal()
        user = get_user_by_id(db_session_local, user_id=token_data.user_id)
        
        if user is None or not user.is_active:
            await websocket.close(code=fastapi_status.WS_1008_POLICY_VIOLATION, reason="User not found or inactive")
            logger.warning(f"Chat WS connection failed: User {token_data.user_id} not found or inactive.")
            return
            
        logger.info(f"Chat WS connection authenticated for user {user.id} ({user.username})")
            
    except Exception as auth_err:
        logger.error(f"Chat WS authentication error: {auth_err}", exc_info=True)
        await websocket.close(code=fastapi_status.WS_1011_INTERNAL_ERROR, reason="Authentication error")
        if db_session_local: # Close session if opened
             db_session_local.close()
        return
    # --- End Authentication --- #

    # Authentication successful, proceed with chat logic
    # Use the authenticated user and the created db_session
    # Need to get or create a DB chat session ID here
    db_chat_session_id: Optional[int] = None 
    try:
        # Try to find the most recent session for this user, or create one
        # This logic might need refinement based on UI session handling
        sessions = get_user_chat_sessions(db=db_session_local, user_id=user.id, limit=1)
        if sessions:
            db_chat_session_id = sessions[0].id
            logger.info(f"Chat WS using existing session {db_chat_session_id} for user {user.id}")
        else:
            new_session = create_chat_session(db=db_session_local, user_id=user.id)
            db_chat_session_id = new_session.id
            logger.info(f"Chat WS created new session {db_chat_session_id} for user {user.id}")

        if not db_chat_session_id:
             raise Exception("Failed to get or create a DB chat session ID.")

        # Keep connection open and handle messages
        while True:
            # Ensure DB session is available for message handling
            if not db_session_local or not db_session_local.is_active:
                 db_session_local = SessionLocal()
                 logger.info("Re-established DB session for ongoing chat WS.")
                 
            data = await websocket.receive_text()
            # Parse incoming message (assuming JSON like {"query": "...", "stream": true})
            try:
                request_data = json.loads(data)
                query = request_data.get("query")
                stream = request_data.get("stream", True) # Default to streaming
                if not query:
                     raise ValueError("Missing 'query' in message payload")
                 # TODO: Get knowledge_only, use_web flags from request_data if needed
            except (json.JSONDecodeError, ValueError) as parse_error:
                 logger.error(f"Chat WS failed to parse message: {data}, Error: {parse_error}")
                 await websocket.send_text(json.dumps({"error": "Invalid message format"}))
                 continue
                 
            logger.info(f"Chat WS received query: '{query}' for session {db_chat_session_id}")

            try:
                # Call the core logic function (ensure it accepts db session)
                await chat_with_knowledge_core(
                    websocket=websocket,
                    db=db_session_local,
                    user=user,
                    session_id=db_chat_session_id,
                    query=query,
                    stream=stream,
                    # Pass other parameters like knowledge_only, use_web if implemented
                )
                 # chat_with_knowledge_core handles sending messages/streams back
            except Exception as core_error:
                 logger.error(f"Error during chat core logic for session {db_chat_session_id}: {core_error}", exc_info=True)
                 try:
                     await websocket.send_text(json.dumps({"error": f"Processing error: {core_error}"}))
                 except Exception as send_err:
                     logger.error(f"Chat WS failed to send error message: {send_err}")
                     # Connection might be dead, break loop?
                     break

    except WebSocketDisconnect:
        logger.info(f"Chat WebSocket disconnected for user {user.id if user else 'unknown'}, session {db_chat_session_id if db_chat_session_id else 'unknown'}.")
    except Exception as e:
        logger.error(f"Unexpected error in Chat WebSocket handler for user {user.id if user else 'unknown'}, session {db_chat_session_id if db_chat_session_id else 'unknown'}: {e}", exc_info=True)
        # Attempt to close gracefully on unexpected error
        if websocket.client_state != WebSocketState.DISCONNECTED:
             try:
                 await websocket.close(code=fastapi_status.WS_1011_INTERNAL_ERROR)
             except RuntimeError: pass
    finally:
        # Ensure DB session is closed on disconnect/error
        if db_session_local:
             db_session_local.close()
             logger.info(f"Closed DB session for chat WS user {user.id if user else 'unknown'}.")

# --- Helper Functions for Prompt Construction ---

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a single context string."""
    if not chunks:
        return ""
    # Ensure text exists and handle potential non-string types gracefully
    return "\n\n".join(str(chunk.get("text", "")) for chunk in chunks if chunk.get("text") is not None)

def format_web_results(results: List[Dict[str, str]]) -> str:
    """Formats web search results into a single context string."""
    if not results:
        return ""
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No Title')
        snippet = result.get('snippet', 'No Snippet')
        link = result.get('link', 'No Link')
        formatted.append(f"Result {i}:\nTitle: {title}\nLink: {link}\nSnippet: {snippet}")
    return "\n\n".join(formatted)

async def chat_with_knowledge_core(
    websocket: WebSocket,
    db: Session, # Added DB session
    user: User,
    session_id: int, # Added session ID
    query: str,
    knowledge_only: bool = True,
    use_web: bool = False,
    model: str = CONFIG.chat.model,
    provider: ProviderType = CONFIG.chat.provider,
    temperature: float = CONFIG.chat.temperature,
    filters: Optional[List[Dict]] = None,
):
    """Core logic for handling a query, retrieving context, generating, and streaming.
    
    Args:
        websocket: The WebSocket connection.
        db: SQLAlchemy DB session.
        user: The authenticated user.
        session_id: The active chat session ID.
        query: The user's query.
        knowledge_only: If True, only use vector store, no web search or general LLM.
        use_web: If True, perform web search.
        model: LLM model name.
        provider: LLM provider.
        temperature: LLM temperature.
        filters: Optional filters for semantic search.
    """
    log_prefix = f"[Core WS {user.username} Session {session_id}]"
    logger.info(f"{log_prefix} Starting chat logic. Query: '{query[:50]}...'")
    full_response = ""
    retrieved_chunks = []
    web_results = []
    sources = [] # Initialize sources list
    
    try:
        # --- Load History --- 
        try:
             existing_messages = get_session_messages(db, session_id, limit=CONFIG.chat.history_limit)
             logger.info(f"{log_prefix} Loaded {len(existing_messages)} messages from history.")
        except Exception as hist_err:
             logger.error(f"{log_prefix} Failed to load message history: {hist_err}", exc_info=True)
             existing_messages = []
             await websocket.send_json({"type": "error", "data": "Failed to load message history"})
             # Decide if we should return here or continue without history?
             # For now, continue without history
             
        # --- Retrieve Context --- 
        try:
            if knowledge_only or use_web: # Search KB if knowledge_only or if general mode allows KB
                logger.info(f"{log_prefix} Performing semantic search (knowledge_only={knowledge_only})...")
                search_start_time = asyncio.get_event_loop().time()
                try:
                    retrieved_chunks = await semantic_search(
                        query=query,
                        user=user,
                        limit=CONFIG.chat.chunks_limit,
                        filters=filters # Pass filters if any
                    )
                    search_duration = asyncio.get_event_loop().time() - search_start_time
                    logger.info(f"{log_prefix} Semantic search completed in {search_duration:.2f}s. Found {len(retrieved_chunks)} chunks.")
                    if retrieved_chunks:
                        sources = list(set(c.get('metadata', {}).get('filename') for c in retrieved_chunks if c.get('metadata', {}).get('filename')))
                    else:
                        logger.info(f"{log_prefix} No relevant chunks found from semantic search.")
                except Exception as search_err:
                     logger.error(f"{log_prefix} Error during semantic search: {search_err}", exc_info=True)
                     # Decide how to handle: send error to client? Continue without KB context?
                     await websocket.send_json({"type": "error", "data": f"Error during knowledge base search: {search_err}"})
                     # Continue without KB context if knowledge_only is false
                     if knowledge_only:
                         return # Cannot proceed if knowledge only is required and search failed
                     retrieved_chunks = [] # Ensure chunks list is empty
            else:
                logger.info(f"{log_prefix} Skipping semantic search as knowledge_only=False and use_knowledge=False (or just use_knowledge=False)...")

            if use_web:
                logger.info(f"{log_prefix} Performing web search...")
                web_search_start_time = asyncio.get_event_loop().time()
                try:
                    web_results = google_search(
                        query=query,
                        limit=CONFIG.web_search.limit
                    )
                    web_search_duration = asyncio.get_event_loop().time() - web_search_start_time
                    logger.info(f"{log_prefix} Web search completed in {web_search_duration:.2f}s. Found {len(web_results)} results.")
                    # Send web results to client immediately? Or just use in prompt?
                    await websocket.send_json({"type": "web_results", "data": web_results})
                except Exception as web_err:
                    logger.error(f"{log_prefix} Error during web search: {web_err}", exc_info=True)
                    await websocket.send_json({"type": "error", "data": f"Error during web search: {web_err}"})
                    # Continue without web context
                    web_results = []

        except Exception as retrieval_err: # Catch-all for unexpected errors during the retrieval phase
             logger.error(f"{log_prefix} Unexpected error during retrieval phase: {retrieval_err}", exc_info=True)
             await websocket.send_json({"type": "error", "data": f"Unexpected error preparing context: {retrieval_err}"})
             return # Stop processing if context preparation fails catastrophically

        # --- Prepare Messages for LLM ---
        messages: List[Dict[str, str]] = []

        # Determine System Prompt based on context availability when knowledge_only is false
        system_prompt = CONFIG.chat.system_prompt # Default system prompt
        if not knowledge_only and not retrieved_chunks:
            # Override system prompt for general chat when no KB context found
            system_prompt = "You are a helpful AI assistant. Answer the user's query generally."
            logger.info(f"{log_prefix} No KB context found and knowledge_only=False. Using general system prompt.")

        messages.append({"role": "system", "content": system_prompt})

        # Add context chunks if available and relevant (based on knowledge_only flag)
        if retrieved_chunks: # Check if list is not empty
            context = format_context(retrieved_chunks)
            messages.append({"role": "system", "content": f"Use the following context to answer the query:\\n\\n{context}"})
            logger.info(f"{log_prefix} Context prepared. Length: {len(context)} chars.")
        else:
            # Log if knowledge_only was true but no chunks were found
            if knowledge_only:
                 logger.warning(f"{log_prefix} Knowledge Only mode is enabled, but no relevant chunks were retrieved from semantic search.")
                 # Maybe add a message indicating no context found? Or rely on default system prompt behavior.
                 # messages.append({"role": "system", "content": "No relevant context was found in the knowledge base."})


        # Add web results if available and relevant
        if web_results:
            web_context = format_web_results(web_results)
            messages.append({"role": "system", "content": f"Use the following web search results to answer the query:\\n\\n{web_context}"})

        # Add chat history (limited)
        # Ensure history messages don't exceed token limits (simplistic for now)
        history_limit = CONFIG.chat.history_limit
        start_index = max(0, len(existing_messages) - history_limit)
        for msg in existing_messages[start_index:]:
            messages.append({"role": msg.sender, "content": msg.content})

        # Add the current user query
        messages.append({"role": "user", "content": query})

        # --- Generate Response ---
        logger.info(f"{log_prefix} Starting generation with {provider}/{model}...")
        full_response = ""
        response_stream = generate_with_provider(messages, model, provider, temperature)
        async for chunk in response_stream:
            full_response += chunk
            try:
                 logger.debug(f"{log_prefix} Sending chunk via WebSocket...")
                 await websocket.send_json({"type": "chunk", "data": chunk})
                 logger.debug(f"{log_prefix} Successfully sent chunk via WebSocket.")
            except Exception as ws_send_err:
                  logger.error(f"{log_prefix} Failed to send chunk via WebSocket: {ws_send_err}", exc_info=True)
                  # Decide whether to break or continue if WS fails
                  break # Stop sending if websocket fails
                  
        logger.info(f"{log_prefix} Exited generate_with_provider loop. Total chunks received: {len(full_response)}")
            
    except GenerationError as e:
        logger.error(f"{log_prefix} LLM generation failed: {e}")
        # Send error to client without saving assistant message
        await websocket.send_json({"type": "error", "data": f"LLM generation failed: {str(e)}"})
        return # Stop processing here
    except Exception as e:
        logger.exception(f"{log_prefix} Unexpected error during generation stream: {e}")
        # Send error to client without saving assistant message
        await websocket.send_json({"type": "error", "data": f"Unexpected error during generation: {str(e)}"})
        return # Stop processing here
        
    logger.info(f"{log_prefix} Finished generation stream processing logic.")

    # --- Save Assistant Response & Finish --- 
    if full_response:
        try:
            # Ensure we save the message in the context of the original DB session
            add_chat_message(db, session_id, "assistant", full_response)
            logger.info(f"{log_prefix} Saved assistant response to DB.")
        except Exception as db_err:
            logger.error(f"{log_prefix} Failed to save assistant message to DB: {db_err}", exc_info=True)
            # Send an error, but maybe don't halt everything if response was already sent?
            await websocket.send_json({"type": "error", "data": "Failed to save assistant response history"})
            
    # Send end signal
    await websocket.send_json({"type": "end"})
    logger.info(f"{log_prefix} Sent end signal.")
    logger.info(f"{log_prefix} Chat logic execution finished.")
    
    # Close the passed DB session
    db.close() 