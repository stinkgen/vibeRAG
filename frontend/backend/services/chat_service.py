"""Chat service module for VibeRAG—handling knowledge bombs with style! 🎯

This module provides a clean service layer for chat functionality, keeping
the routes lean and the business logic mean.
"""

from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from config.config import CONFIG
from retrieval.search import semantic_search
from vector_store.milvus_ops import search_by_tags, search_by_metadata
from generation.generate import generate_with_provider
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Base RAG fuckup—something's off! 🚨"""
    pass

class SearchError(RAGError):
    """Search went sideways—Milvus ain't happy! 💀"""
    pass

class GenerationError(RAGError):
    """LLM shat the bed—generation's busted! 🔥"""
    pass

class ChatService:
    """Runs chat like a goddamn boss—streaming vibes locked in! 🔥"""
    
    def __init__(self, model: str = "llama2", provider: str = "ollama", temperature: float = 0.7):
        """Initializes the chat service with spicy defaults! 🌶️"""
        self.model = model
        self.provider = provider
        self.temperature = temperature
        logging.info(f"Chat service initialized—{model} on {provider} ready to shred! 🔥")

    async def chat_with_knowledge(self, 
            query: str, 
            filename: Optional[str] = None,
            knowledge_only: bool = True,
            use_web: bool = False,
            model: Optional[str] = None,
            provider: Optional[str] = None
        ) -> AsyncGenerator[str, None]:
        """Streams chat responses with relevant knowledge—brain's firing! 🧠
        
        Args:
            query: User's question
            filename: Optional file to search in
            knowledge_only: If True, only respond based on found knowledge
            use_web: Whether to use web search for additional context
            model: Optional model to override the default
            provider: Optional provider to override the default
            
        Returns:
            AsyncGenerator yielding response chunks as JSON strings
            
        Raises:
            SearchError: If search fails
            GenerationError: If LLM generation fails
        """
        try:
            # Use provided values or fall back to defaults
            model_to_use = model or self.model
            provider_to_use = provider or self.provider
            
            # Validate provider/model compatibility
            if provider_to_use == "openai" and not model_to_use.startswith(("gpt-", "ft:")):
                logging.warning(f"Incompatible model {model_to_use} for OpenAI, falling back to gpt-3.5-turbo")
                model_to_use = "gpt-3.5-turbo"
            elif provider_to_use == "ollama" and model_to_use.startswith("gpt-"):
                logging.warning(f"Incompatible model {model_to_use} for Ollama, falling back to llama3")
                model_to_use = "llama3"
            
            logging.info(f"Starting chat with model={model_to_use}, provider={provider_to_use}")
            
            # Search for relevant chunks
            results = semantic_search(query, filename)
            logging.info(f"Found {len(results)} relevant chunks")
            
            # Handle case with no results
            if not results and knowledge_only:
                logging.warning("No hits found and knowledge_only=True—returning error message")
                yield json.dumps({"error": "No relevant information found in the knowledge base"})
                return
                
            # Get web results if requested
            web_results = []
            if use_web:
                try:
                    from retrieval.search import google_search
                    web_results = google_search(query)
                    logging.info(f"Found {len(web_results)} web results")
                except Exception as e:
                    logging.error(f"Web search failed: {str(e)}")
            
            # Build context from results
            context_parts = []
            
            # Add document chunks
            for result in results:
                context_parts.append(result["text"])
                
            # Add web results
            for result in web_results:
                context_parts.append(f"Web: {result['title']} - {result['snippet']}")
                
            # Combine context
            context = "\n\n".join(context_parts) if context_parts else "No specific context found."

            # Collect source information to return
            sources = []
            for result in results:
                # Extract metadata
                metadata = result.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                filename = metadata.get("filename", "unknown")
                page = metadata.get("page", "?")
                sources.append(f"Page {page} of {filename}")
                
            # Add web sources
            for result in web_results:
                sources.append(f"Web: {result['title']} ({result['link']})")
                
            # First yield sources as JSON
            if sources:
                yield json.dumps({"sources": sources})
                logging.info(f"Sent {len(sources)} sources to client")

            # Build messages for LLM
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely. If you don't know something or the context doesn't contain relevant information, say so."},
                {"role": "user", "content": f"Based on this context:\n{context}\n\nAnswer this question: {query}"}
            ]

            # Stream that sweet response
            try:
                response_generator = generate_with_provider(
                    messages, 
                    model_to_use, 
                    provider_to_use, 
                    self.temperature
                )
                
                async for chunk in response_generator:
                    if chunk:  # Skip empty chunks
                        response_json = json.dumps({"response": chunk})
                        yield response_json
                        logging.debug(f"Streamed response chunk: {chunk[:20]}...")
            except Exception as e:
                logging.error(f"Error during response generation: {str(e)}")
                yield json.dumps({"error": f"Generation error: {str(e)}"})

        except Exception as e:
            error_msg = f"Chat service error: {str(e)}"
            logging.error(f"{error_msg}")
            yield json.dumps({"error": error_msg})

    # Search and bypass fixed—Phase 2's flowing! 🔥

# Tags and metadata—4090's filtering like a champ! 🚀 