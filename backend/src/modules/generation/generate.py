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
            # Get OpenAI client
            openai_client = get_openai_client()
                
            # Call OpenAI API using new interface
            stream = await openai_client.chat.completions.create(
                model=model or CONFIG.openai.default_model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        logging.error(f"Generation failed: {str(e)}")
        raise GenerationError(f"Failed to generate response: {str(e)}")

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

async def chat_with_knowledge(
    query: str,
    filename: Optional[str] = None,
    knowledge_only: bool = True,
    use_web: bool = False,
    model: str = CONFIG.chat.model,
    provider: ProviderType = CONFIG.chat.provider,
    temperature: float = CONFIG.chat.temperature
) -> AsyncGenerator[Dict[str, Any], None]:
    """Chat with an LLM using knowledge from your docs, yielding structured events.

    Args:
        query: The user's question or prompt
        filename: Optional filename to filter chunks
        knowledge_only: If True, only respond based on found knowledge
        use_web: Whether to include web search results
        model: Name of the model to use
        provider: LLM provider
        temperature: Generation temperature

    Yields:
        Dicts representing events: 
        {'type': 'sources', 'data': [...]}
        {'type': 'response', 'data': '...'}
        {'type': 'error', 'data': '...'}
    """
    from src.modules.retrieval.search import semantic_search, google_search
    
    logger.info(f"Starting chat_with_knowledge: query='{query[:50]}...', filename={filename}, use_web={use_web}")

    try:
        # 1. Get relevant chunks
        try:
            chunks = semantic_search(query, top_k=CONFIG.chat.chunks_limit, filename=filename)
            logger.info(f"Found {len(chunks)} relevant chunks")
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            yield {"type": "error", "data": f"Failed to retrieve information: {e}"}
            return

        # 2. Get web results if requested
        web_results = []
        if use_web:
            try:
                web_results = google_search(query)
                logger.info(f"Found {len(web_results)} web results")
            except Exception as e:
                logger.warning(f"Google search failed: {e}") # Log as warning, proceed without web results
                # Optionally yield a warning to the client?
                # yield {"type": "warning", "data": "Web search failed, proceeding without it."} 

        # 3. Handle no context found
        if not chunks and not web_results:
            if knowledge_only:
                logger.warning("No knowledge found and knowledge_only=True")
                yield {"type": "error", "data": "I don't have any relevant information to answer your question based on the available documents."}
                return
            else:
                logger.info("No context found, proceeding with query only.")
                # Proceed without specific context

        # 4. Extract and yield sources
        sources_list = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            source_filename = metadata.get('filename', 'unknown')
            source_page = metadata.get('page', '?')
            sources_list.append({"type": "document", "filename": source_filename, "page": source_page})

        for result in web_results:
             sources_list.append({"type": "web", "title": result.get('title', ''), "link": result.get('link', '')})
        
        if sources_list:
            yield {"type": "sources", "data": sources_list}

        # 5. Format context for LLM
        context_parts = []
        for i, chunk in enumerate(chunks):
            text = chunk['text'].strip()
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, str): # Redundant parsing, but keep for context string
                try: metadata = json.loads(metadata)
                except: metadata = {}
            filename_ctx = metadata.get('filename', 'unknown file')
            page_ctx = metadata.get('page', '?')
            context_parts.append(f"Source {i+1} ({filename_ctx}, page {page_ctx}):\n{text}")

        if web_results:
            context_parts.append("\n--- Web Results ---")
            for i, result in enumerate(web_results):
                context_parts.append(f"Web Result {i+1} ({result['title']}):\n{result['snippet']}")

        context = "\n\n".join(context_parts)
        if not context:
             context = "No specific context was found."
        
        # 6. Build messages for the LLM
        system_prompt = (
            "You are a helpful AI assistant. Use the provided context (documents and web results) "
            "to answer the user's question accurately and concisely. "
            "Cite the sources used in your answer using the format [Source X] for documents or [Web Result X] for web results, corresponding to the numbering in the context. "
            "If the context doesn't contain relevant information to answer the question, explicitly state that."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 7. Generate response and yield chunks
        try:
            response_generator = generate_with_provider(
                messages,
                model,
                provider,
                temperature
            )
            
            async for response_chunk in response_generator:
                if response_chunk: # Ensure not yielding empty strings
                    yield {"type": "response", "data": response_chunk}
        except GenerationError as e:
            logger.error(f"Generation failed during streaming: {e}")
            yield {"type": "error", "data": f"LLM generation failed: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error during generation streaming: {e}")
            yield {"type": "error", "data": f"An unexpected error occurred during generation: {e}"}

    except Exception as e:
        logger.exception(f"Unhandled error in chat_with_knowledge: {e}") # Use logger.exception to include traceback
        yield {"type": "error", "data": f"An unexpected error occurred: {e}"} 

# Types locked—code's sharp as fuck! 🔥 