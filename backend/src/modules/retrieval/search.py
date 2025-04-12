"""Retrieval module for semantic and keyword search in Milvus.

This module brings the heat with semantic search using sentence-transformers,
keyword filtering, and a hybrid approach that combines both like a DJ mixing tracks.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union, TypedDict, Tuple
import numpy as np
import requests
from dotenv import load_dotenv
from pymilvus import Collection, connections, utility
from src.modules.vector_store.milvus_ops import search_by_tags, search_by_metadata, search_collection
from src.modules.config.config import CONFIG  # Config's in the house! 🏠
from src.modules.embedding.service import get_embedding_model # Import the service function
from src.modules.vector_store.milvus_ops import connect_milvus, init_collection
from src.modules.config.config import CONFIG, SearchConfig
from src.modules.vector_store.milvus_ops import (
    get_user_collection_name, 
    get_admin_collection_name, 
    get_global_collection_name
)
from src.modules.auth.database import User # Import User model

# Load environment variables
load_dotenv()

# Configure logging with style
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants that slap
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "vibe_chunks"
TOP_K = 10  # Default number of results to return

# Initialize the embedding model
# _model = None

def get_embeddings(text: str) -> np.ndarray:
    """Get embeddings for a SINGLE text string using the shared embedding service."""
    try:
        model = get_embedding_model()
        embeddings = model.encode([text], convert_to_numpy=True)[0]
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate single embedding using service: {str(e)}")
        raise

# Type definitions—4090's precision-tuned! 🔥
class SearchMetadata(TypedDict):
    """Metadata for search results—keeping it tight! 💪"""
    source: str
    filename: str
    page: Optional[int]

class SearchResult(TypedDict):
    """Search result with all the goods—structured AF! 🎯"""
    chunk_id: Optional[int]
    text: str
    metadata: SearchMetadata
    tags: List[str]
    score: float

def get_document(filename: str) -> str:
    """Retrieve and reconstruct a document from its chunks.
    
    Args:
        filename: Name of the file to retrieve
        
    Returns:
        Complete document text
    """
    collection = Collection(CONFIG.milvus.collection_name)
    collection.load()
    
    # Query for all chunks from this file
    expr = f'filename == "{filename}"'
    results = collection.query(
        expr=expr,
        output_fields=["chunk_id", "text", "metadata"],
        limit=1000  # Get all chunks
    )
    
    if not results:
        logger.warning(f"No chunks found for {filename}! 🤔")
        return ""
    
    # Sort chunks by their position in the original document
    # This assumes chunks were created in order
    sorted_chunks = sorted(results, key=lambda x: x['chunk_id'])
    
    # Combine chunks
    document = "\n\n".join(chunk['text'] for chunk in sorted_chunks)
    logger.info(f"Pulled {filename}—source locked! 📄")
    
    return document

# --- Helper to build Milvus filter expression --- 
def _build_milvus_filter_expression(filters: Optional[List[Dict[str, str]]]) -> Optional[str]:
    """Builds a Milvus filter expression from frontend filter structure."""
    if not filters:
        return None
    
    # Assuming filters is a list like: [{'type': 'filename', 'value': 'doc1.pdf'}, {'type': 'tag', 'value': 'urgent'}]
    # We only support filename filters for now based on the chat_with_knowledge_core call
    # Example: Convert [{'type': 'filename', 'value': 'doc1.pdf'}, {'type': 'filename', 'value': 'doc2.md'}] to "filename in ['doc1.pdf', 'doc2.md']"
    
    filename_filters = [f['value'] for f in filters if f.get('type') == 'filename' and f.get('value')]
    
    if not filename_filters:
        return None
        
    # Escape single quotes within filenames if necessary (though unlikely)
    safe_filenames = [name.replace("'", "\'") for name in filename_filters]
    
    # Format for Milvus 'in' operator
    expression = f"filename in ['{'', ''.join(safe_filenames)}']"
    logger.info(f"Constructed Milvus filter expression: {expression}")
    return expression

# --- Search Functions ---

async def semantic_search(
    query: str, 
    user: User, # Add user object
    limit: int = CONFIG.search.default_limit, 
    min_score: float = CONFIG.search.min_score, 
    filters: Optional[List[Dict[str, str]]] = None 
) -> List[Dict[str, Any]]:
    """Performs semantic search across relevant collections for the user."""
    logger.info(f"Performing semantic search for user '{user.username}', query: '{query[:50]}...'")
    
    # Determine collections to search and ensure user collection exists
    collections_to_search = []
    user_collection_name = None # Initialize
    
    if user.role == 'admin':
        admin_collection_name = get_admin_collection_name()
        if admin_collection_name: # Check if admin collection name is configured
             collections_to_search.append(admin_collection_name)
             # Ensure admin collection exists (might be redundant if lifespan works, but safe)
             if not utility.has_collection(admin_collection_name):
                 logger.warning(f"Admin collection '{admin_collection_name}' not found, attempting to create.")
                 try:
                    init_collection(admin_collection_name)
                    logger.info(f"Dynamically created admin collection '{admin_collection_name}'.")
                 except Exception as e_create:
                    logger.error(f"Failed to dynamically create admin collection '{admin_collection_name}': {e_create}", exc_info=True)
                    # Continue without admin collection if creation fails
    else:
        user_collection_name = get_user_collection_name(user.id)
        if user_collection_name:
            # ----> DYNAMICALLY CREATE USER COLLECTION IF IT DOES NOT EXIST <----
            if not utility.has_collection(user_collection_name):
                logger.warning(f"User collection '{user_collection_name}' for user {user.id} not found. Creating now.")
                try:
                    init_collection(user_collection_name) # Create it
                    logger.info(f"Successfully created dynamic collection '{user_collection_name}' for user {user.id}.")
                except Exception as e_create:
                    logger.error(f"Failed to dynamically create collection '{user_collection_name}' for user {user.id}: {e_create}", exc_info=True)
                    # Decide how to handle this: maybe raise error or return empty results?
                    # For now, let's log and proceed without this collection.
                    user_collection_name = None # Prevent searching a non-existent collection
            
            if user_collection_name: # Re-check if creation was successful
                collections_to_search.append(user_collection_name)

    # Always add global collection if configured
    global_collection_name = get_global_collection_name()
    if global_collection_name: # Check if global collection name is configured
        collections_to_search.append(global_collection_name)
        # Ensure global collection exists (might be redundant if lifespan works, but safe)
        if not utility.has_collection(global_collection_name):
            logger.warning(f"Global collection '{global_collection_name}' not found, attempting to create.")
            try:
                init_collection(global_collection_name)
                logger.info(f"Dynamically created global collection '{global_collection_name}'.")
            except Exception as e_create:
                logger.error(f"Failed to dynamically create global collection '{global_collection_name}': {e_create}", exc_info=True)
                # Continue without global collection if creation fails
                if global_collection_name in collections_to_search:
                     collections_to_search.remove(global_collection_name)

    # Log the final list of collections we will actually search
    logger.info(f"Final collections to search: {collections_to_search}")
    
    # If no valid collections to search, return empty results
    if not collections_to_search:
        logger.warning("No valid Milvus collections found to search for this user.")
        return []
        
    # Generate query embedding
    try:
        # Ensure get_embeddings is synchronous or awaited if it becomes async
        # If get_embeddings uses a thread pool implicitly, this is okay.
        # If it's truly async, use await asyncio.to_thread(get_embeddings, query)
        query_vector = get_embeddings(query) 
    except Exception as e:
         logger.warning(f"Failed to generate query embedding: {e}", exc_info=True)
         return []
    
    # Build filter expression from frontend filters (currently only supports filename)
    filter_expression = _build_milvus_filter_expression(filters)
    
    try:
        # Call the updated search_collection with the list of names
        search_results = search_collection(
            query_vector=query_vector, 
            collection_names=collections_to_search,
            limit=limit,               
            expr=filter_expression     
            # Pass search_params, output_fields if needed, defaults are in search_collection
        )
        
        # Filter results by min_score (search_collection returns sorted results)
        # Note: Score interpretation depends on metric (L2 lower is better, IP higher is better)
        metric_type = CONFIG.milvus.search_params.get("metric_type", "L2").upper()
        if metric_type == "IP":
             final_results = [res for res in search_results if res.get("score", 0) >= min_score]
        else: # Default to L2 or other distance metrics where lower is better
             final_results = [res for res in search_results if res.get("score", float('inf')) <= min_score]
        
        logger.info(f"Semantic search completed. Found {len(final_results)} results meeting score threshold across {len(collections_to_search)} collections.")
        # !!! PARANOID LOGGING !!!
        logger.info(f"Type of final_results before returning: {type(final_results)}")
        if final_results:
            logger.info(f"Type of first item in final_results: {type(final_results[0])}")
            logger.info(f"Content of first item: {final_results[0]}")
        # !!! END PARANOID LOGGING !!!
        return final_results
        
    except Exception as e:
        logger.exception(f"Error during semantic search for user '{user.username}': {e}")
        return []

def keyword_search(
    query: str,
    top_k: int = TOP_K,
    filename: Optional[str] = None
) -> List[SearchResult]:
    """Search for chunks containing specific keywords.
    
    Args:
        query: The keyword query to search for
        top_k: Maximum number of results to return
        filename: Optional filename to filter results
        
    Returns:
        List of SearchResult dicts with text and metadata
    """
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # Split query into words and search for each
    words = query.lower().split()
    if not words:
        return []
    
    # Add filename filter if provided
    expr = ""
    if filename:
        expr = f'metadata like "%\\"filename\\": \\"{filename}\\"%"'
        logger.info(f"Narrowing to {filename}—precision shot! 🎯")
    
    # Get chunks and filter
    results = collection.query(
        expr=expr,
        output_fields=["text", "metadata"],
        limit=1000  # Get enough chunks to filter
    )
    
    # Format results and calculate scores based on word matches
    hits: List[SearchResult] = []
    for hit in results:
        text = hit.get('text', '').lower()
        # Score based on how many query words appear in the text
        matching_words = sum(1 for word in words if word in text)
        if matching_words > 0:  # Only include if at least one word matches
            # Parse metadata from JSON string back to dict
            try:
                metadata = json.loads(hit.get('metadata', '{}'))
            except (json.JSONDecodeError, TypeError):
                metadata = {'source': 'unknown source', 'filename': 'unknown'}
            
            score = matching_words / len(words)  # Normalize score to [0,1]
            hits.append({
                'chunk_id': None,  # Keyword search doesn't track chunk IDs
                'text': hit.get('text', ''),
                'metadata': metadata,
                'tags': [],  # Keyword search doesn't track tags
                'score': score
            })
    
    # Sort by score and take top_k
    hits.sort(key=lambda x: x['score'], reverse=True)
    hits = hits[:top_k]
    
    logger.info(f"Keyword search nailed it with {len(hits)} matches! 🎯")
    return hits

def hybrid_search(
    query: str,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: int = TOP_K,
    filename: Optional[str] = None
) -> List[SearchResult]:
    """Combine semantic and keyword search results like a master mixologist.
    
    Args:
        query: Search query
        semantic_weight: Weight for semantic search scores (default: 0.7)
        keyword_weight: Weight for keyword search scores (default: 0.3)
        top_k: Number of final results to return
        filename: Optional filename to filter results
        
    Returns:
        List of SearchResult dicts with text, metadata, and combined score
    """
    # Get results from both approaches
    semantic_results = semantic_search(query, top_k=top_k, filename=filename)
    keyword_results = keyword_search(query, top_k=top_k, filename=filename)
    
    # Combine results, using text as key to avoid duplicates
    combined_results: Dict[str, SearchResult] = {}
    
    # Add semantic results
    for hit in semantic_results:
        combined_results[hit['text']] = {
            'text': hit['text'],
            'metadata': hit['metadata'],
            'score': hit['score'] * semantic_weight
        }
    
    # Add or update with keyword results
    for hit in keyword_results:
        if hit['text'] in combined_results:
            # Add keyword score to existing entry
            combined_results[hit['text']]['score'] += hit['score'] * keyword_weight
        else:
            combined_results[hit['text']] = {
                'text': hit['text'],
                'metadata': hit['metadata'],
                'score': hit['score'] * keyword_weight
            }
    
    # Convert to list and sort by score
    results = list(combined_results.values())
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Take top_k results
    results = results[:top_k]
    
    logger.info(f"Hybrid search dropped {len(results)} fire results! 🔥")
    return results

def google_search(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Search the web using Google Custom Search API.
    
    Args:
        query: Search query
        limit: Number of results to return (default: 3)
        
    Returns:
        List of dicts with title, link, and snippet
    """
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key or not engine_id:
        logger.error("Google Search API credentials not found in .env.local! 🚫")
        return []
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": engine_id,
        "q": query,
        "num": limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("items", [])[:limit]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        
        logger.info(f"Google's dropping {len(results)} web hits—fresh intel! 🌐")
        return results
        
    except Exception as e:
        logger.error(f"Google search failed: {str(e)} 😅")
        return []

def search_by_tag_list(tags: List[str]) -> List[Dict[str, Any]]:
    """Search for documents with specific tags.
    
    Args:
        tags: List of tags to search for
        
    Returns:
        List of matching chunks with metadata
    """
    # Build expression for tag search
    expr = " || ".join([f"array_contains(tags, '{tag}')" for tag in tags])
    
    # Execute search
    try:
        # Create a dummy query vector for tag-only search
        dummy_vector = [0.0] * CONFIG.milvus.embedding_dim
        
        results = search_collection(
            query_vector=dummy_vector,
            expr=expr,
            output_fields=["text", "metadata", "tags"],
            limit=CONFIG.search.default_limit
        )
        return results
    except Exception as e:
        logger.error(f"Tag search failed: {str(e)}")
        raise

def search_by_metadata_field(field: str, value: str) -> List[Dict[str, Any]]:
    """Search documents by metadata field value.

    Args:
        field: Metadata field name
        value: Field value to match

    Returns:
        List of matching documents
    """
    expr = f"{field} == '{value}'"

    # Use a dummy query vector since we only care about filtering
    dummy_vector = [0.0] * CONFIG.milvus.dim

    results = search_collection(
        query_vector=dummy_vector,
        collection_name=CONFIG.milvus.collection_name,
        expr=expr
    )

    return results 