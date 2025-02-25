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

from config.config import CONFIG
from .exceptions import GenerationError

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
) -> AsyncGenerator[str, None]:
    """Chat with an LLM using knowledge from your docs.

    Args:
        query: The user's question or prompt
        filename: Optional filename to filter chunks
        knowledge_only: If True, only respond based on found knowledge
        use_web: Whether to include web search results
        model: Name of the model to use
        provider: LLM provider
        temperature: Generation temperature

    Returns:
        AsyncGenerator yielding response chunks
    """
    from retrieval.search import semantic_search, google_search

    # Get relevant chunks
    chunks = semantic_search(query, top_k=CONFIG.chat.chunks_limit, filename=filename)
    logger.info(f"Found {len(chunks)} relevant chunks")

    # Add web results if requested
    web_results = []
    if use_web:
        web_results = google_search(query)
        logger.info(f"Found {len(web_results)} web results")

    if not chunks and not web_results:
        if knowledge_only:
            logger.warning("No knowledge found and knowledge_only=True")
            yield "I don't have any relevant information to answer your question."
            return

    # Format context
    context_parts = []

    # Add document chunks
    for chunk in chunks:
        text = chunk['text'].strip()
        # Process metadata - it might be a string or a dict
        metadata = chunk['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"filename": "unknown file", "page": "?"}
        
        filename = metadata.get('filename', 'unknown file')
        page = metadata.get('page', '?')
        context_parts.append(f"From {filename} (page {page}):\n{text}")

    # Add web results
    if web_results:
        for result in web_results:
            context_parts.append(f"From {result['title']}:\n{result['snippet']}")

    context = "\n\n".join(context_parts)

    # Build messages for the LLM
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant. Use the provided context to answer
            questions accurately and concisely. If you don't know something or the context
            doesn't contain relevant information, say so."""
        },
        {
            "role": "user",
            "content": f"""Based on this context:
{context}

Answer this question: {query}"""
        }
    ]

    # Generate response
    async for chunk in generate_with_provider(
        messages=messages,
        model=model,
        provider=provider,
        temperature=temperature
    ):
        yield chunk

# Types locked—code's sharp as fuck! 🔥 