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
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.modules.config.config import CONFIG
from src.modules.generation.exceptions import GenerationError

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
            raise ValueError("OpenAI API key not found in environment or config")
        if api_key.startswith("sk-") and len(api_key) > 20:  # Simple validation for OpenAI key format
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_API_BASE", CONFIG.openai.base_url)
            )
            logger.info("OpenAI client initialized successfully")
        else:
            raise ValueError("Invalid OpenAI API key format - should start with 'sk-'")
    return client

# Type definitions—4090's precision-tuned! 🔥
class Message(TypedDict):
    """Message for LLM chat—keeping it tight! 💪"""
    role: Literal["system", "user", "assistant"]
    content: str

class WebResult(TypedDict):
    """Web search result—structured AF! 🎯"""
    title: str
    link: str
    snippet: str

class ChunkMetadata(TypedDict):
    """Chunk metadata—clean types! 🚀"""
    filename: str
    page: Union[int, str]

class Chunk(TypedDict):
    """Document chunk with metadata—precision data! 💪"""
    text: str
    metadata: ChunkMetadata

ProviderType = Literal["openai", "anthropic", "ollama"]

async def generate_with_provider(
    messages: List[Dict[str, str]],
    model: str,
    provider: str = "ollama",
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    """Generate text using the specified provider.

    Args:
        messages: List of message dicts with role and content
        model: Model to use for generation
        provider: Provider to use (ollama or openai)
        temperature: Temperature for generation

    Yields:
        Generated text chunks
    """
    try:
        if provider == "ollama":
            # Format messages for ollama
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            # Call ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CONFIG.ollama.host}{CONFIG.ollama.generate_endpoint}",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {"temperature": temperature}
                    }
                ) as response:
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                            except json.JSONDecodeError:
                                continue
                    
        elif provider == "openai":
            openai_client = get_openai_client()
            logger.info(f"[generate_with_provider] Attempting OpenAI call. Model: {model}, Temp: {temperature}")
            logger.debug(f"[generate_with_provider] Messages: {messages}") # Log the messages being sent
            
            try:
                logger.info(f"[generate_with_provider] Attempting chat completions with model: {model}")
                stream = await openai_client.chat.completions.create(
                    model=model or CONFIG.openai.default_model,
                    messages=messages,
                    temperature=temperature,
                    stream=True
                )
                logger.info(f"[generate_with_provider] Successfully created OpenAI stream for model: {model}")
                
                chunk_count = 0
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        chunk_count += 1
                        # logger.debug(f"[generate_with_provider] Yielding chunk {chunk_count}: {content}") # Very verbose, uncomment if needed
                        yield content
                logger.info(f"[generate_with_provider] Finished streaming {chunk_count} chunks from chat completions for model: {model}")

            except openai.NotFoundError as e:
                # Check if the error is specifically the 'model not supported for chat' error
                error_details = e.response.json()
                if "error" in error_details and "message" in error_details["error"] and "not supported in the v1/chat/completions endpoint" in error_details["error"]["message"]:
                    logger.warning(f"[generate_with_provider] Model '{model}' not supported by chat completions. Error: {e}")
                    
                    # Fallback to the v1/completions endpoint
                    # Need to format messages into a single prompt
                    prompt = "\\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    
                    try:
                        stream = await openai_client.completions.create(
                            model=model, # Use the selected model
                            prompt=prompt,
                            temperature=temperature,
                            stream=True,
                            max_tokens=1500  # Adjust max_tokens as needed for completions
                        )
                        async for chunk in stream:
                            if chunk.choices[0].text:
                                yield chunk.choices[0].text
                        logger.info(f"Successfully streamed from completions endpoint for model: {model}")

                    except Exception as fallback_e:
                        logger.error(f"Fallback to completions endpoint failed for model '{model}': {fallback_e}")
                        raise GenerationError(f"Failed to generate response using fallback completions endpoint: {fallback_e}") from fallback_e
                else:
                    # Re-raise if it's a different NotFoundError (e.g., model truly doesn't exist)
                    logger.error(f"OpenAI API Not Found error (non-chat compatibility): {e}")
                    raise GenerationError(f"OpenAI API request failed: {e}") from e
            except Exception as e:
                logger.exception(f"[generate_with_provider] Error during OpenAI chat completions stream for model '{model}': {e}") # Use logger.exception
                raise GenerationError(f"Failed to generate response: {e}") from e
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        # Catch exceptions outside the provider blocks (e.g., client init)
        logging.error(f"Generation failed: {str(e)}")
        # Avoid wrapping GenerationError in another GenerationError
        if isinstance(e, GenerationError):
            raise e
        else:
            raise GenerationError(f"Failed to generate response: {str(e)}") from e

async def ollama(messages: List[Dict[str, str]], model: str = "llama2", stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generate chat response using Ollama.
    
    Args:
        messages: List of message dicts with role and content
        model: Model name to use
        stream: Whether to stream the response
        
    Returns:
        Generated response text or async generator yielding response chunks
    """
    try:
        # Format messages for Ollama API
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        async with aiohttp.ClientSession() as session:
            # Call Ollama API
            async with session.post(
                f"{CONFIG.chat.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": formatted_messages,
                    "stream": stream
                }
            ) as response:
                response.raise_for_status()
                
                if stream:
                    async def generate():
                        async for line in response.content:
                            if line:
                                chunk = json.loads(line)
                                if chunk.get("message", {}).get("content"):
                                    yield chunk["message"]["content"]
                    return generate()
                
                # Extract response text for non-streaming
                result = await response.json()
                return result["message"]["content"]
        
    except Exception as e:
        logger.error(f"Ollama generation failed: {str(e)}")
        raise

# --- Import the in-memory store ---
from src.api.app import chat_histories_store
# ---------------------------------

async def chat_with_knowledge_ws(
    websocket: WebSocket,
    query: str,
    filename: Optional[str] = None,
    knowledge_only: bool = True,
    use_web: bool = False,
    model: str = CONFIG.chat.model,
    provider: ProviderType = CONFIG.chat.provider,
    temperature: float = CONFIG.chat.temperature,
    filters: Optional[List[Dict]] = None,
    chat_history_id: Optional[str] = None
):
    """Chat with an LLM using knowledge, sending events over WebSocket."""
    logger.info(f"Starting chat_with_knowledge_ws: query='{query[:30]}...', filename={filename}, knowledge_only={knowledge_only}, use_web={use_web}, history_id={chat_history_id}")
    context_text = ""
    sources = []
    web_results = []

    # --- Existing logic for conditional search --- 
    perform_search = knowledge_only or filename or use_web or (filters is not None and len(filters) > 0)
    if perform_search:
        logger.info("Search required based on parameters (knowledge_only, filename, filters or use_web)")
        # 1. Semantic Search
        if (filename or knowledge_only or (filters is not None and len(filters) > 0)) and not (use_web and not knowledge_only):
            logger.info(f"Performing semantic search. Filters: {filters if filters else 'N/A'}")
            try:
                # Import moved inside to avoid circular dependency if search uses generate
                from src.modules.retrieval.search import semantic_search
                search_results = semantic_search(
                    query,
                    filename=filename,
                    limit=CONFIG.search.default_limit
                )
                logger.info(f"Semantic search returned {len(search_results)} results.")
                
                # --- MOVE RESULT PROCESSING INSIDE TRY BLOCK --- 
                if search_results:
                    sources_data = []
                    for r in search_results:
                        raw_metadata = r.get('metadata') # Get raw metadata field
                        metadata_dict = {} # Default to empty dict
                        if isinstance(raw_metadata, str):
                            try:
                                metadata_dict = json.loads(raw_metadata)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse metadata JSON string: {raw_metadata}")
                        elif isinstance(raw_metadata, dict):
                            metadata_dict = raw_metadata
                        else:
                            logger.warning(f"Unexpected metadata type: {type(raw_metadata)}, value: {raw_metadata}")

                        src_filename = metadata_dict.get('filename', 'unknown')
                        src_page = metadata_dict.get('page', '?')
                        sources_data.append({"filename": src_filename, "page": src_page})
                    
                    unique_sources = []
                    seen = set()
                    for item in sources_data:
                        identifier = (item['filename'], item['page'])
                        if identifier not in seen:
                            unique_sources.append(item)
                            seen.add(identifier)
                    
                    context_text += "Relevant Documents:\n" + "\n".join([r['text'] for r in search_results])
                    await websocket.send_json({"type": "sources", "data": unique_sources})
                # --- END MOVED BLOCK ---

            except Exception as e:
                # This will now catch errors during search AND processing
                logger.exception(f"Semantic search or result processing failed: {e}") 
                await websocket.send_json({"type": "error", "data": f"Failed to retrieve information: {e}"}) 
                return
        # 2. Web Search
        if use_web:
            logger.info("Performing web search...")
            try:
                # Import moved inside
                from src.modules.retrieval.search import web_search
                web_results = await web_search(query)
                if web_results:
                    context_text += "\n\nWeb Search Results:\n" + "\n".join([f"Title: {r['title']}\nSnippet: {r['snippet']}" for r in web_results])
                    # Optionally send web sources over WebSocket
                    # await websocket.send_json({"type": "web_sources", "data": web_results})
            except Exception as e:
                logger.exception(f"Web search failed: {e}")
                # Optionally send warning/error over WebSocket
                # await websocket.send_json({"type": "warning", "data": f"Web search failed: {e}"})
        # 3. Check if context is sufficient
        if knowledge_only and not search_results:
            logger.info("Knowledge only mode, but no relevant documents found.")
            # Send message and close or return? For now, send and return.
            await websocket.send_json({"type": "response", "data": "I don\'t have any relevant information in my knowledge base to answer that question."})
            await websocket.send_json({"type": "end", "data": "No relevant knowledge found."}) # Send end event
            return
    else:
        logger.info("Skipping knowledge/web search based on parameters.")

    # --- Retrieve and Prepare History ---
    retrieved_history: List[Dict[str, str]] = []
    if chat_history_id:
        retrieved_history = chat_histories_store.get(chat_history_id, [])
        # Limit history (e.g., last 10 messages)
        history_limit = 10 
        if len(retrieved_history) > history_limit:
            retrieved_history = retrieved_history[-history_limit:]
            logger.info(f"Truncated history for {chat_history_id} to last {history_limit} messages.")
    else:
         logger.info("No chat_history_id provided, starting fresh conversation.")
         # If no ID, we don't store history for this turn

    # --- Add current user query to store BEFORE generation ---
    if chat_history_id:
        if chat_history_id not in chat_histories_store:
             chat_histories_store[chat_history_id] = []
        # Store the user message CONTENT as it's being sent to the LLM (with context)
        
        # REVERT: Store only the raw user query in history for clarity
        # Construct the user message content with context for storage <-- Remove this section
        # context_for_prompt = context_text if context_text else "No specific context was provided for this query."
        # user_content_for_history = f"CONTEXT:\n{context_for_prompt}\n\nQUESTION: {query}"
        # chat_histories_store[chat_history_id].append({"role": "user", "content": user_content_for_history})
        
        # Correct line to store raw query:
        chat_histories_store[chat_history_id].append({"role": "user", "content": query})
        
        # logger.info(f"Added user query (with context) to history store for ID: {chat_history_id}") # OLD LOG
        logger.info(f"Added raw user query to history store for ID: {chat_history_id}") # Correct log

    # --- Prepare messages for LLM --- 
    system_prompt_content = (
        "You are a helpful AI assistant. Use the provided CONTEXT from the user's current query "
        "to answer the user's question accurately and concisely. Consider the CHAT HISTORY provided "
        "for conversational context. If the context or history doesn't contain relevant information, state that clearly.\n\n" 
        "**Formatting Instructions:**\n"
        "- Use Markdown for formatting (e.g., lists, bold, italics, code blocks) to enhance readability.\n"
        "- Structure responses logically, using headings or bullet points where appropriate."
    )
    system_prompt_message = {"role": "system", "content": system_prompt_content}

    # This calculation is now duplicated, but keep it for clarity in constructing the LLM message
    context_for_prompt_llm = context_text if context_text else "No specific context was provided for this query."
    current_user_query_content_llm = f"CONTEXT:\n{context_for_prompt_llm}\n\nQUESTION: {query}"
    current_user_message = {"role": "user", "content": current_user_query_content_llm}

    # Combine system prompt, retrieved history, and current user message
    messages_for_llm = [system_prompt_message] + retrieved_history + [current_user_message]
    logger.debug(f"Messages prepared for LLM (History length: {len(retrieved_history)}): {messages_for_llm}")

    # --- Generate response and send over WebSocket --- 
    full_assistant_response = "" # Accumulate the full response
    try:
        response_generator = generate_with_provider(
            messages_for_llm, # Use the combined list
            model,
            provider,
            temperature
        )
        
        async for response_chunk in response_generator:
            if response_chunk: # Ensure not sending empty strings
                full_assistant_response += response_chunk # Accumulate
                try:
                    # Check connection state before sending
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"type": "response", "data": response_chunk})
                    else:
                        logger.warning("WebSocket disconnected before sending response chunk.")
                        break # Exit loop if disconnected
                except WebSocketDisconnect:
                    logger.warning("WebSocket disconnected during send_json for response chunk.")
                    break # Exit loop if disconnected
        
        # --- Store assistant response AFTER successful generation ---
        if chat_history_id:
             # Check if the ID is still valid (might have been cleared elsewhere?) - unlikely but safe
             if chat_history_id in chat_histories_store:
                  chat_histories_store[chat_history_id].append({"role": "assistant", "content": full_assistant_response})
                  logger.info(f"Added assistant response to history store for ID: {chat_history_id}")
             else:
                  logger.warning(f"Chat history ID {chat_history_id} disappeared from store before assistant response could be saved.")
        
        # Send end message only if still connected
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "end", "data": "Generation complete."}) 
                logger.info("LLM generation finished, sent end event.")
            else:
                 logger.warning("WebSocket disconnected before sending end event.")
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected during send_json for end event.")

    except GenerationError as e:
        logger.error(f"Generation failed during streaming: {e}")
        # Do NOT store assistant response if generation failed.
        try:
            # Check connection state before sending error
            if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.send_json({"type": "error", "data": f"LLM generation failed: {e}"}) 
            else:
                 logger.warning("WebSocket disconnected before sending GenerationError.")
        except WebSocketDisconnect:
             logger.warning("WebSocket disconnected during send_json for GenerationError.")
    except WebSocketDisconnect: 
        # Explicitly catch disconnects that might happen *during* generate_with_provider iteration
        logger.warning("WebSocket disconnected during generate_with_provider iteration.")
    except Exception as e:
        logger.exception(f"Unexpected error during generation streaming: {e}")
        try:
            # Check connection state before sending error
            if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.send_json({"type": "error", "data": f"An unexpected error occurred during generation: {e}"})
            else:
                 logger.warning("WebSocket disconnected before sending generic Exception error.")
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected during send_json for generic Exception error.")

# Types locked—code's sharp as fuck! 🔥 