"""Milvus vector store operations for managing embeddings.

This module handles all the juicy vector operations with Milvus,
making your embeddings searchable faster than you can say 'semantic similarity'.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional
import time
import uuid
import numpy as np

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)

from src.modules.config.config import CONFIG  # Config's in the house! 🏠

# Configure logging with style
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_milvus(host: str = None, port: int = None) -> None:
    """Connect to Milvus server like a boss.
    
    Args:
        host: Milvus server host (default: from config)
        port: Milvus server port (default: from config)
    """
    try:
        host = host or CONFIG.milvus.host
        port = port or CONFIG.milvus.port
        
        # Check if already connected
        try:
            if connections.has_connection("default"):
                logger.info("Already connected to Milvus")
                return
        except:
            pass
            
        connections.connect(host=host, port=port)
        logger.info("Milvus is live, baby! 🚀")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        raise

def ensure_connection():
    """Ensure Milvus connection is established."""
    try:
        if not connections.has_connection("default"):
            connect_milvus()
    except Exception as e:
        logger.error(f"Failed to ensure Milvus connection: {str(e)}")
        raise

def init_collection(recreate: bool = False) -> Collection:
    """Initialize the chunks collection with all the bells and whistles.
    
    Args:
        recreate: If True, drop existing collection and create new one
        
    Returns:
        Collection: The initialized Milvus collection
    """
    # Ensure connection first
    ensure_connection()
    
    if utility.has_collection(CONFIG.milvus.collection_name):
        if recreate:
            utility.drop_collection(CONFIG.milvus.collection_name)
            logger.info(f"Dropped existing collection: {CONFIG.milvus.collection_name}")
        else:
            logger.info(f"Collection {CONFIG.milvus.collection_name} already exists, loading it up")
            collection = Collection(CONFIG.milvus.collection_name)
            
            # Check if index exists, create if not
            try:
                index_info = collection.index().params
                logger.info("Index already exists")
            except Exception:
                logger.info("Creating index for existing collection")
                index = {
                    "index_type": CONFIG.milvus.index_params["index_type"],
                    "metric_type": CONFIG.milvus.index_params["metric_type"],
                    "params": CONFIG.milvus.index_params["params"]
                }
                collection.create_index(CONFIG.milvus.embedding_field, index)
                logger.info(f"Created index on {CONFIG.milvus.embedding_field} field")
            
            collection.load()
            return collection
    
    # Create fields
    fields = [
        FieldSchema(
            name=CONFIG.milvus.chunk_id_field,
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),
        FieldSchema(
            name=CONFIG.milvus.doc_id_field,
            dtype=DataType.VARCHAR,
            max_length=CONFIG.milvus.field_params[CONFIG.milvus.doc_id_field]['max_length']
        ),
        FieldSchema(
            name=CONFIG.milvus.embedding_field,
            dtype=DataType.FLOAT_VECTOR,
            dim=CONFIG.milvus.embedding_dim
        ),
        FieldSchema(
            name=CONFIG.milvus.text_field,
            dtype=DataType.VARCHAR,
            max_length=CONFIG.milvus.field_params[CONFIG.milvus.text_field]['max_length']
        ),
        FieldSchema(
            name=CONFIG.milvus.metadata_field,
            dtype=DataType.JSON
        ),
        FieldSchema(
            name=CONFIG.milvus.tags_field,
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=CONFIG.milvus.field_params[CONFIG.milvus.tags_field]['max_capacity'],
            max_length=CONFIG.milvus.field_params[CONFIG.milvus.tags_field]['max_length']
        ),
        FieldSchema(
            name=CONFIG.milvus.filename_field,
            dtype=DataType.VARCHAR,
            max_length=CONFIG.milvus.field_params[CONFIG.milvus.filename_field]['max_length']
        ),
        FieldSchema(
            name='category',
            dtype=DataType.VARCHAR,
            max_length=CONFIG.milvus.field_params['category']['max_length']
        )
    ]
    
    # Create schema
    schema = CollectionSchema(fields=fields)
    
    # Create collection
    collection = Collection(CONFIG.milvus.collection_name, schema=schema)
    
    # Create index on the embedding field
    index = {
        "index_type": CONFIG.milvus.index_params["index_type"],
        "metric_type": CONFIG.milvus.index_params["metric_type"],
        "params": CONFIG.milvus.index_params["params"]
    }
    collection.create_index(CONFIG.milvus.embedding_field, index)
    logger.info(f"Created index on {CONFIG.milvus.embedding_field} field")
    
    # Load the collection into memory
    collection.load()
    logger.info(f"Collection {CONFIG.milvus.collection_name} loaded into memory")
    
    return collection

def store_with_metadata(chunks: List[Dict], tags: List[str] = None, metadata: Dict = None, filename: str = None) -> List[str]:
    """Store chunks with metadata in Milvus.
    
    Args:
        chunks: List of dicts with text, embedding, and metadata
        tags: List of tags to apply to all chunks
        metadata: Additional metadata to apply to all chunks
        filename: Filename to apply to all chunks
        
    Returns:
        List of document IDs for the stored chunks
    """
    collection = init_collection()
    
    # Generate a unique doc_id for this batch
    doc_id = str(uuid.uuid4())
    
    # Prepare data in Milvus format - order must match schema
    chunk_ids = []  # Will be auto-generated
    doc_ids = []  # doc_id field
    embeddings = []  # embedding field
    texts = []  # text field
    chunk_metadata = []  # metadata field
    chunk_tags = []  # tags field
    filenames = []  # filename field
    categories = []  # category field
    
    for chunk in chunks:
        # Get filename from chunk metadata first, then document metadata, then parameter
        chunk_metadata_dict = chunk.get('metadata', {})
        chunk_filename = chunk_metadata_dict.get('filename', metadata.get('filename', '') if metadata else '')
        if filename:  # Override with parameter if provided
            chunk_filename = filename
            
        category = chunk_metadata_dict.get('category', metadata.get('category', '') if metadata else '')
        
        # Combine chunk and document metadata
        combined_metadata = {
            **(metadata or {}),
            **(chunk_metadata_dict or {})
        }
        
        # Append all fields in schema order
        doc_ids.append(doc_id)
        embeddings.append(chunk['embedding'])
        texts.append(chunk['text'])
        chunk_metadata.append(json.dumps(combined_metadata))
        chunk_tags.append(tags or [])
        filenames.append(chunk_filename)
        categories.append(category)
    
    # Insert into collection - order must match schema exactly
    insert_data = [
        doc_ids,  # doc_id field
        embeddings,  # embedding field
        texts,  # text field
        chunk_metadata,  # metadata field
        chunk_tags,  # tags field
        filenames,  # filename field
        categories  # category field
    ]
    
    collection.insert(insert_data)
    logger.info(f"Inserted {len(chunks)} chunks into Milvus")
    
    # Create index if it doesn't exist
    try:
        index_info = collection.index().params
        logger.info("Index already exists")
    except Exception:
        logger.info("Creating index")
        index = {
            "index_type": CONFIG.milvus.index_params["index_type"],
            "metric_type": CONFIG.milvus.index_params["metric_type"],
            "params": CONFIG.milvus.index_params["params"]
        }
        collection.create_index(CONFIG.milvus.embedding_field, index)
        logger.info(f"Created index on {CONFIG.milvus.embedding_field} field")
    
    # Flush to ensure data is persisted
    collection.flush()
    logger.info("Flushed data to disk")
    
    return doc_ids

def delete_document(doc_id: str) -> bool:
    """Delete all chunks associated with a document.

    Args:
        doc_id: Document ID or filename to delete

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Check if collection exists
        if not utility.has_collection(CONFIG.milvus.collection_name):
            logger.warning(f"Collection {CONFIG.milvus.collection_name} does not exist")
            return True  # Return True since there's nothing to delete

        # Get collection with schema and load it
        collection = init_collection()
        collection.load()  # Ensure collection is loaded before querying

        # Try to delete by doc_id first
        expr = f'{CONFIG.milvus.doc_id_field} == "{doc_id}"'
        results = collection.query(
            expr=expr,
            output_fields=[CONFIG.milvus.chunk_id_field],
            limit=1
        )

        # If no results found, try by filename
        if len(results) == 0:
            expr = f'{CONFIG.milvus.filename_field} == "{doc_id}"'
            results = collection.query(
                expr=expr,
                output_fields=[CONFIG.milvus.chunk_id_field],
                limit=1
            )

        # If still no results, document doesn't exist
        if len(results) == 0:
            logger.info(f"No chunks found for document: {doc_id}")
            return True

        # Delete all chunks for this document
        collection.delete(expr)
        collection.flush()  # Ensure deletion is persisted
        logger.info(f"Deleted all chunks for document: {doc_id}")

        return True

    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {str(e)}")
        return False

def clean_collection() -> bool:
    """Clean up the entire Milvus collection.
    
    Returns:
        True if cleanup was successful
    """
    try:
        if utility.has_collection(CONFIG.milvus.collection_name):
            utility.drop_collection(CONFIG.milvus.collection_name)
            logger.info(f"Dropped collection {CONFIG.milvus.collection_name}—fresh start! 🧹")
            # Reinitialize empty collection
            init_collection()
            return True
        else:
            logger.info(f"No collection {CONFIG.milvus.collection_name} found to clean")
            return False
            
    except Exception as e:
        logger.error(f"Failed to clean collection: {str(e)}")
        return False

def search_by_tags(
    tags: List[str],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for chunks by tags.
    
    Args:
        tags: List of tags to search for
        limit: Maximum number of results
        
    Returns:
        List of matching chunks with metadata
    """
    collection = Collection(CONFIG.milvus.collection_name)
    collection.load()
    
    # Build tag search expression using array_contains
    tag_conditions = [f"array_contains({CONFIG.milvus.tags_field}, '{tag}')" for tag in tags]
    expr = " && ".join(tag_conditions)
    
    output_fields = [
        CONFIG.milvus.chunk_id_field,
        CONFIG.milvus.doc_id_field,
        CONFIG.milvus.text_field,
        CONFIG.milvus.metadata_field,
        CONFIG.milvus.tags_field,
        CONFIG.milvus.filename_field
    ]
    
    results = collection.query(
        expr=expr,
        output_fields=output_fields,
        limit=limit
    )
    
    chunks = []
    for hit in results:
        try:
            metadata = hit.get(CONFIG.milvus.metadata_field, {})  # Already JSON object in Milvus
            tags = hit.get(CONFIG.milvus.tags_field, [])  # Already array in Milvus
        except Exception:
            metadata = {}
            tags = []
        
        chunks.append({
            'chunk_id': hit.get(CONFIG.milvus.chunk_id_field),
            'doc_id': hit.get(CONFIG.milvus.doc_id_field),
            'text': hit.get(CONFIG.milvus.text_field),
            'metadata': metadata,
            'tags': tags,
            'filename': hit.get(CONFIG.milvus.filename_field)
        })
    
    logger.info(f"Found {len(chunks)} chunks with tags {tags}—tagged content locked! 🏷️")
    return chunks

def search_by_metadata(
    key: str,
    value: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for chunks by metadata field.
    
    Args:
        key: Metadata field name
        value: Value to search for
        limit: Maximum number of results
        
    Returns:
        List of matching chunks with metadata
    """
    collection = Collection(CONFIG.milvus.collection_name)
    collection.load()
    
    # Search in metadata JSON using JSON field access
    expr = f'metadata["{key}"] == "{value}"'
    
    output_fields = [
        CONFIG.milvus.chunk_id_field,
        CONFIG.milvus.doc_id_field,
        CONFIG.milvus.text_field,
        CONFIG.milvus.metadata_field,
        CONFIG.milvus.tags_field,
        CONFIG.milvus.filename_field
    ]
    
    results = collection.query(
        expr=expr,
        output_fields=output_fields,
        limit=limit
    )
    
    chunks = []
    for hit in results:
        try:
            metadata = hit.get(CONFIG.milvus.metadata_field, {})  # Already JSON object in Milvus
            tags = hit.get(CONFIG.milvus.tags_field, [])  # Already array in Milvus
        except Exception:
            metadata = {}
            tags = []
        
        chunks.append({
            'chunk_id': hit.get(CONFIG.milvus.chunk_id_field),
            'doc_id': hit.get(CONFIG.milvus.doc_id_field),
            'text': hit.get(CONFIG.milvus.text_field),
            'metadata': metadata,
            'tags': tags,
            'filename': hit.get(CONFIG.milvus.filename_field)
        })
    
    logger.info(f"Found {len(chunks)} chunks with {key}={value}—metadata matched! 🔍")
    return chunks

def search_collection(
    query_vector: List[float],
    collection_name: str = None,
    expr: str = None,
    limit: int = 10,
    search_params: Optional[Dict[str, Any]] = None,
    output_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Search the Milvus collection.

    Args:
        query_vector: Query vector to search with
        collection_name: Name of collection to search
        expr: Optional filter expression
        limit: Maximum number of results to return
        search_params: Optional search parameters
        output_fields: Optional list of fields to return in results

    Returns:
        List of search results with distances
    """
    try:
        # Ensure connection first
        ensure_connection()
        
        # Use default collection name if not specified
        if collection_name is None:
            collection_name = CONFIG.milvus.collection_name

        # Get collection
        collection = Collection(collection_name)
        collection.load()

        # Set default search params if not provided
        if search_params is None:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }

        # Set default output fields if not provided
        if output_fields is None:
            output_fields = ["text", "metadata", "tags"]

        # Execute search
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "text": hit.fields.get("text", ""),
                    "metadata": hit.fields.get("metadata", {}),
                    "score": 1.0 / (1.0 + hit.distance)  # Convert distance to similarity score
                }
                if "tags" in hit.fields:
                    result["tags"] = hit.fields["tags"]
                formatted_results.append(result)

        return formatted_results

    except Exception as e:
        logging.error(f"Search failed: {str(e)}")
        raise

async def disconnect_milvus():
    """Disconnect from Milvus server."""
    try:
        await connections.disconnect("default")
        logger.info("Disconnected from Milvus")
    except Exception as e:
        logger.error(f"Failed to disconnect from Milvus: {str(e)}")
        raise