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
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
from vector_store.milvus_ops import search_by_tags, search_by_metadata, search_collection
from config.config import CONFIG  # Config's in the house! 🏠

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
_model = None

def get_embeddings(text: str) -> np.ndarray:
    """Get embeddings for text using sentence-transformers.
    
    Args:
        text: Text to embed
        
    Returns:
        Numpy array of embeddings
    """
    global _model
    
    try:
        # Initialize model if not already done
        if _model is None:
            _model = SentenceTransformer(CONFIG.embedding.model_name)
            logger.info(f"Initialized {CONFIG.embedding.model_name} for embeddings 🧠")
        
        # Generate embeddings
        embeddings = _model.encode([text], convert_to_numpy=True)[0]
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
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

def semantic_search(
    query: str,
    filename: Optional[str] = None,
    limit: Optional[int] = None,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Searches Milvus with a query—4090's shredding! 🔥
    
    Args:
        query: Search query
        filename: Optional filename to filter results
        limit: Maximum number of results (alias for top_k)
        top_k: Maximum number of results (deprecated, use limit instead)
        
    Returns:
        List of matching chunks with metadata and scores
    """
    # Handle both limit and top_k parameters
    result_limit = limit or top_k or CONFIG.search.default_limit
    
    # Get embeddings for query
    query_embedding = get_embeddings(query)
    
    # Prepare search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    
    # Add filename filter if provided
    expr = f"{CONFIG.milvus.filename_field} == '{filename}'" if filename else None
    
    # Execute search
    results = search_collection(
        query_vector=query_embedding,
        limit=result_limit,
        expr=expr,
        search_params=search_params
    )
    
    return results

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