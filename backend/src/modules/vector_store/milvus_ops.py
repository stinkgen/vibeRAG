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

def get_collection_schema() -> CollectionSchema:
    """Creates the collection schema based on config."""
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
        # Add category field if defined in config, otherwise skip
        *([] if 'category' not in CONFIG.milvus.field_params else [
            FieldSchema(
                name='category',
                dtype=DataType.VARCHAR,
                max_length=CONFIG.milvus.field_params['category']['max_length']
            )
        ])
    ]
    return CollectionSchema(fields=fields, description="Document Chunks Collection")

def create_collection_index(collection: Collection):
    """Creates the HNSW index on the embedding field."""
    try:
        if not collection.has_index():
            logger.info(f"Creating index for collection '{collection.name}'...")
            index_params = {
                "index_type": CONFIG.milvus.index_params["index_type"],
                "metric_type": CONFIG.milvus.index_params["metric_type"],
                "params": CONFIG.milvus.index_params["params"]
            }
            collection.create_index(CONFIG.milvus.embedding_field, index_params)
            logger.info(f"Created index on field '{CONFIG.milvus.embedding_field}' for collection '{collection.name}'.")
        else:
            logger.info(f"Index already exists for collection '{collection.name}'.")
    except Exception as e:
        logger.error(f"Failed to create or check index for collection '{collection.name}': {e}", exc_info=True)
        # Decide if this should raise an error

def init_collection(collection_name: str, recreate: bool = False) -> Collection:
    """Initialize or load a specific Milvus collection.

    Args:
        collection_name: The name of the collection to initialize/load.
        recreate: If True, drop existing collection and create new one.

    Returns:
        Collection: The initialized Milvus collection.
    """
    ensure_connection()
    
    collection_exists = utility.has_collection(collection_name)

    if collection_exists:
        if recreate:
            logger.warning(f"Dropping existing collection: {collection_name}")
            utility.drop_collection(collection_name)
            collection_exists = False # Reset flag as it's dropped
        else:
            logger.info(f"Collection '{collection_name}' already exists, loading it.")
            collection = Collection(collection_name)
            create_collection_index(collection) # Ensure index exists
            collection.load() # Load into memory
            logger.info(f"Collection '{collection_name}' loaded.")
            return collection

    # If collection doesn't exist or was dropped
    if not collection_exists:
        logger.info(f"Creating new collection: '{collection_name}'")
        schema = get_collection_schema()
        collection = Collection(name=collection_name, schema=schema, consistency_level=CONFIG.milvus.consistency_level)
        logger.info(f"Collection '{collection_name}' created.")
        create_collection_index(collection) # Create index
        collection.load() # Load into memory
        logger.info(f"Collection '{collection_name}' loaded.")
        return collection

    # This part should ideally not be reached, but return collection if it exists somehow
    return Collection(collection_name)

# --- Add Collection Management Functions ---

def list_all_collections() -> List[str]:
    """Lists all collections in Milvus."""
    ensure_connection()
    try:
        return utility.list_collections()
    except Exception as e:
        logger.error(f"Failed to list Milvus collections: {e}")
        return []

def drop_collection(collection_name: str) -> bool:
    """Drops a specific collection."""
    ensure_connection()
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"Successfully dropped collection: {collection_name}")
            return True
        else:
            logger.warning(f"Collection '{collection_name}' does not exist, cannot drop.")
            return False
    except Exception as e:
        logger.error(f"Failed to drop collection '{collection_name}': {e}")
        return False

def get_user_collection_name(user_id: int) -> str:
    """Generates the collection name for a given user ID."""
    # Use a consistent prefix and the user's ID
    return f"user_{user_id}"

def get_admin_collection_name() -> str:
    """Gets the name for the admin's personal collection."""
    return "user_admin" # Or use admin user ID if preferred
    
def get_global_collection_name() -> str:
    """Gets the name for the global shared collection."""
    return "global_kb"

# --- Modify existing functions to use collection_name ---

def store_with_metadata(collection_name: str, chunks: List[Dict], tags: List[str] = None, metadata: Dict = None, filename: str = None) -> List[str]:
    """Store chunks with metadata in a specific Milvus collection.
    
    Args:
        collection_name: The name of the collection to store data in.
        chunks: List of dicts with text, embedding, and metadata
        tags: List of tags to apply to all chunks
        metadata: Additional metadata to apply to all chunks
        filename: Filename to apply to all chunks
        
    Returns:
        List of document IDs for the stored chunks
    """
    collection = init_collection(collection_name) # Get the specific collection
    
    # Generate a unique doc_id for this batch within this collection context
    doc_id = str(uuid.uuid4())
    
    # Prepare data fields
    doc_ids = []
    embeddings = []
    texts = []
    chunk_metadata_list = [] # Renamed to avoid confusion
    chunk_tags = []
    filenames = []
    categories = [] # Ensure category is handled

    has_category_field = 'category' in CONFIG.milvus.field_params

    for chunk in chunks:
        chunk_metadata_dict = chunk.get('metadata', {})
        # Determine filename
        chunk_filename = chunk_metadata_dict.get('filename', metadata.get('filename', '') if metadata else '')
        if filename:
            chunk_filename = filename

        # Combine metadata
        combined_metadata = {**(metadata or {}), **(chunk_metadata_dict or {})}

        doc_ids.append(doc_id)
        embeddings.append(chunk['embedding'])
        texts.append(chunk['text'])
        chunk_metadata_list.append(json.dumps(combined_metadata))
        chunk_tags.append(tags or chunk_metadata_dict.get('tags', [])) # Use chunk tags if doc tags not provided
        filenames.append(chunk_filename)
        if has_category_field:
             category = chunk_metadata_dict.get('category', metadata.get('category', '') if metadata else '')
             categories.append(category)

    # Prepare insert_data based on actual schema fields defined in get_collection_schema
    # The order MUST match the fields returned by get_collection_schema()
    insert_data = [
        doc_ids, 
        embeddings, 
        texts, 
        chunk_metadata_list,
        chunk_tags,
        filenames
    ]
    if has_category_field:
        insert_data.append(categories)

    # Insert into the specific collection
    collection.insert(insert_data)
    logger.info(f"Inserted {len(chunks)} chunks into Milvus collection '{collection_name}'.")
    
    # Flush data
    collection.flush()
    logger.info(f"Flushed collection '{collection_name}'.")

    # Index creation is handled by init_collection now
    
    return [doc_id] * len(chunks) # Return the single doc_id used for this batch

def delete_document(collection_name: str, filename: str) -> bool:
    """Delete document chunks associated with a filename from a specific collection."""
    try:
        collection = init_collection(collection_name) # Ensure collection exists and is loaded
        
        expr = f"{CONFIG.milvus.filename_field} == \"{filename}\""
        logger.info(f"Attempting deletion from '{collection_name}' with expression: {expr}")
        
        # Search for matching entities before deleting to log count
        search_res = collection.query(expr=expr, output_fields=[CONFIG.milvus.chunk_id_field])
        delete_count = len(search_res)
        logger.info(f"Found {delete_count} entities to delete for filename '{filename}' in collection '{collection_name}'.")
        
        if delete_count > 0:
            delete_result = collection.delete(expr=expr)
            logger.info(f"Deletion result for {filename} in {collection_name}: {delete_result}")
            collection.flush() # Ensure deletion is persisted
            return True
        else:
            logger.warning(f"No documents found matching filename '{filename}' in collection '{collection_name}' for deletion.")
            return False # Indicate nothing was deleted
            
    except Exception as e:
        logger.error(f"Error deleting document {filename} from {collection_name}: {str(e)}", exc_info=True)
        return False

def update_metadata_in_vector_store(
    collection_name: str,
    filename: str, 
    tags: Optional[List[str]] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Update tags and metadata for all chunks associated with a specific filename
    within a given collection.
    
    Note: Uses a Delete-then-Insert approach.
    
    Args:
        collection_name: The name of the collection to update.
        filename: The filename whose chunks' metadata should be updated.
        tags: The new list of tags. If None, tags are not updated.
        metadata: The new metadata dictionary. If None, metadata is not updated.
        
    Returns:
        bool: True if the update process (delete + insert) succeeded, False otherwise.
    """
    if tags is None and metadata is None:
        logger.warning("Update called with no changes specified for tags or metadata.")
        return True # No changes needed

    collection = init_collection(collection_name) # Use the specified collection
    
    # 1. Query for all existing chunks matching the filename
    query_expr = f"{CONFIG.milvus.filename_field} == \"{filename}\""
    
    # Determine output fields based on schema
    schema_fields = [field.name for field in get_collection_schema().fields]
    output_fields = [
        field for field in schema_fields 
        if field != CONFIG.milvus.chunk_id_field # Exclude PK for re-insert
    ]
    # Ensure embedding is included if not already
    if CONFIG.milvus.embedding_field not in output_fields:
        output_fields.append(CONFIG.milvus.embedding_field)
    # Ensure PK is included for deletion reference
    if CONFIG.milvus.chunk_id_field not in output_fields:
         output_fields.append(CONFIG.milvus.chunk_id_field)
    
    try:
        logger.info(f"Querying collection '{collection_name}' for chunks with filename '{filename}' to update metadata...")
        results = collection.query(
            expr=query_expr,
            output_fields=output_fields,
            limit=16384 # Max limit, adjust if needed
        )
        logger.info(f"Found {len(results)} chunks for filename '{filename}' in collection '{collection_name}'.")

        if not results:
            logger.warning(f"No chunks found for filename '{filename}' in collection '{collection_name}', cannot update metadata.")
            return True # No chunks exist, so "update" is trivially successful

        # Prepare data for re-insertion
        insert_data_lists = {field: [] for field in schema_fields if field != CONFIG.milvus.chunk_id_field}
        original_pks = [res[CONFIG.milvus.chunk_id_field] for res in results] # Store original PKs for deletion

        for chunk_data in results:
            # Update tags if provided, otherwise keep original
            updated_tags = chunk_data.get(CONFIG.milvus.tags_field, [])
            if tags is not None:
                updated_tags = tags
            
            # Update metadata if provided
            try:
                existing_metadata_dict = json.loads(chunk_data.get(CONFIG.milvus.metadata_field, '{}'))
            except (json.JSONDecodeError, TypeError):
                existing_metadata_dict = {}
            
            updated_metadata_dict = existing_metadata_dict
            if metadata is not None:
                 # Preserve essential chunk-specific keys while overriding/adding others
                chunk_specific_keys_to_keep = ['page', 'page_number', 'source']
                preserved_chunk_meta = {k: v for k, v in existing_metadata_dict.items() if k in chunk_specific_keys_to_keep}
                updated_metadata_dict = {**metadata, **preserved_chunk_meta}

            # Populate data for insertion based on schema fields (excluding PK)
            for field_name in insert_data_lists.keys():
                if field_name == CONFIG.milvus.tags_field:
                    insert_data_lists[field_name].append(updated_tags)
                elif field_name == CONFIG.milvus.metadata_field:
                    insert_data_lists[field_name].append(json.dumps(updated_metadata_dict))
                else:
                    # Copy other fields directly from original data
                    insert_data_lists[field_name].append(chunk_data.get(field_name))

        # 2. Delete the original chunks using their primary keys
        if not original_pks:
             logger.warning("No primary keys found for deletion, skipping delete step.")
        else:
            logger.info(f"Deleting {len(original_pks)} original chunks from '{collection_name}' for filename '{filename}' (PKs: {original_pks[:5]}...).")
            delete_expr = f"{CONFIG.milvus.chunk_id_field} in {original_pks}"
            delete_result = collection.delete(expr=delete_expr)
            logger.info(f"Deletion result: {delete_result}") # Log Milvus delete result
            # Check count (delete_result might be MutationResult with delete_count or pk field)
            deleted_count = getattr(delete_result, 'delete_count', len(getattr(delete_result, 'primary_keys', [])))
            if deleted_count != len(original_pks):
                 logger.warning(f"Expected to delete {len(original_pks)} chunks, but Milvus reported deleting {deleted_count}. Proceeding.")
            else:
                 logger.info(f"Deletion successful: {deleted_count} entities deleted.")
            collection.flush() 
            logger.info(f"Collection '{collection_name}' flushed after deleting original chunks.")
            # time.sleep(0.5) # Optional delay

        # 3. Insert the chunks with updated data
        # Convert dict of lists to list of lists in the correct schema order
        schema_field_order = [field.name for field in get_collection_schema().fields if not field.is_primary]
        insert_payload = [insert_data_lists[field_name] for field_name in schema_field_order]
        
        logger.info(f"Inserting {len(insert_payload[0])} chunks with updated metadata into '{collection_name}' for filename '{filename}'...")
        insert_result = collection.insert(insert_payload)
        logger.info(f"Insertion successful: {len(insert_result.primary_keys)} entities inserted with new PKs ({insert_result.primary_keys[:5]}...).")
        collection.flush() # Ensure insertion is committed
        logger.info(f"Collection '{collection_name}' flushed after inserting updated chunks.")
        
        return True

    except Exception as e:
        logger.exception(f"Error updating metadata in collection '{collection_name}' for document '{filename}': {str(e)}")
        return False

def search_by_tags(
    collection_name: str,
    tags: List[str],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for chunks by tags in a specific collection."""
    collection = init_collection(collection_name) # Use specific collection
    # collection.load() # init_collection ensures load
    
    # Build tag search expression using array_contains
    tag_conditions = [f"array_contains({CONFIG.milvus.tags_field}, '{tag}')" for tag in tags]
    expr = " && ".join(tag_conditions)
    
    # Define output fields based on config
    output_fields = [
        CONFIG.milvus.chunk_id_field,
        CONFIG.milvus.doc_id_field,
        CONFIG.milvus.text_field,
        CONFIG.milvus.metadata_field,
        CONFIG.milvus.tags_field,
        CONFIG.milvus.filename_field
    ]
    # Add category if exists
    if 'category' in CONFIG.milvus.field_params:
        output_fields.append('category')
    
    results = collection.query(
        expr=expr,
        output_fields=output_fields,
        limit=limit
    )
    
    chunks = []
    for hit in results:
        try:
            # Milvus SDK v2.3+ typically returns JSON field as dict and ARRAY as list
            metadata = hit.get(CONFIG.milvus.metadata_field, {})
            hit_tags = hit.get(CONFIG.milvus.tags_field, [])
        except Exception as e:
            logger.warning(f"Error parsing metadata/tags for hit {hit.get(CONFIG.milvus.chunk_id_field)}: {e}")
            metadata = {}
            hit_tags = []
        
        chunk_entry = {
            'chunk_id': hit.get(CONFIG.milvus.chunk_id_field),
            'doc_id': hit.get(CONFIG.milvus.doc_id_field),
            'text': hit.get(CONFIG.milvus.text_field),
            'metadata': metadata,
            'tags': hit_tags,
            'filename': hit.get(CONFIG.milvus.filename_field)
        }
        if 'category' in output_fields:
             chunk_entry['category'] = hit.get('category')
        chunks.append(chunk_entry)
    
    logger.info(f"Found {len(chunks)} chunks in '{collection_name}' with tags {tags}")
    return chunks

def search_by_metadata(
    collection_name: str,
    key: str,
    value: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for chunks by metadata field in a specific collection."""
    collection = init_collection(collection_name) # Use specific collection
    # collection.load() # init_collection ensures load
    
    # Search in metadata JSON using JSON field access
    # Ensure value is properly escaped if it contains quotes
    escaped_value = json.dumps(value) # Use json.dumps for proper string escaping
    expr = f'{CONFIG.milvus.metadata_field}["{key}"] == {escaped_value}' 
    
    # Define output fields based on config
    output_fields = [
        CONFIG.milvus.chunk_id_field,
        CONFIG.milvus.doc_id_field,
        CONFIG.milvus.text_field,
        CONFIG.milvus.metadata_field,
        CONFIG.milvus.tags_field,
        CONFIG.milvus.filename_field
    ]
    # Add category if exists
    if 'category' in CONFIG.milvus.field_params:
        output_fields.append('category')
        
    results = collection.query(
        expr=expr,
        output_fields=output_fields,
        limit=limit
    )
    
    chunks = []
    for hit in results:
        try:
            metadata = hit.get(CONFIG.milvus.metadata_field, {})
            tags = hit.get(CONFIG.milvus.tags_field, [])
        except Exception as e:
            logger.warning(f"Error parsing metadata/tags for hit {hit.get(CONFIG.milvus.chunk_id_field)}: {e}")
            metadata = {}
            tags = []
        
        chunk_entry = {
            'chunk_id': hit.get(CONFIG.milvus.chunk_id_field),
            'doc_id': hit.get(CONFIG.milvus.doc_id_field),
            'text': hit.get(CONFIG.milvus.text_field),
            'metadata': metadata,
            'tags': tags,
            'filename': hit.get(CONFIG.milvus.filename_field)
        }
        if 'category' in output_fields:
             chunk_entry['category'] = hit.get('category')
        chunks.append(chunk_entry)
    
    logger.info(f"Found {len(chunks)} chunks in '{collection_name}' with {key}={value}")
    return chunks

def search_collection(
    query_vector: List[float],
    collection_names: List[str],
    expr: str = None,
    limit: int = 10,
    search_params: Optional[Dict[str, Any]] = None,
    output_fields: Optional[List[str]] = None,
    consistency_level: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search one or more Milvus collections.

    Args:
        query_vector: Query vector to search with.
        collection_names: List of collection names to search.
        expr: Optional filter expression (applied to all collections).
        limit: Maximum number of results to return *in total* across collections.
        search_params: Optional search parameters.
        output_fields: Optional list of fields to return in results.
        consistency_level: Optional consistency level override.

    Returns:
        List of search results with distances, sorted by score.
    """
    if not collection_names:
        logger.warning("search_collection called with no collection names.")
        return []
        
    try:
        ensure_connection()
        
        # Set default search params if not provided
        search_params_to_use = search_params or CONFIG.milvus.search_params

        # Set default output fields if not provided
        output_fields_to_use = output_fields or [
            CONFIG.milvus.text_field,
            CONFIG.milvus.metadata_field,
            CONFIG.milvus.tags_field,
            CONFIG.milvus.filename_field # Also include filename by default
        ]
        # Ensure primary key is always included for identification if needed later
        pk_field = CONFIG.milvus.chunk_id_field
        if pk_field not in output_fields_to_use:
             output_fields_to_use.append(pk_field)

        all_results = []
        
        # Milvus multi-collection search is not directly supported in v2.3 for vector search.
        # We need to search each collection individually and merge results.
        # Note: This is less efficient than a native multi-collection search.
        limit_per_collection = limit # Fetch top `limit` from each collection initially
        
        for collection_name in collection_names:
            try:
                collection = init_collection(collection_name) # Ensure exists and load
                logger.info(f"Searching collection: {collection_name}")
                
                search_kwargs = {
                    "data": [query_vector],
                    "anns_field": CONFIG.milvus.embedding_field,
                    "param": search_params_to_use,
                    "limit": limit_per_collection,
                    "expr": expr,
                    "output_fields": output_fields_to_use
                }
                # Add consistency level if provided
                consistency = consistency_level or CONFIG.milvus.consistency_level
                if consistency:
                     search_kwargs["consistency_level"] = consistency
                
                results = collection.search(**search_kwargs)

                # Process results for this collection
                for hit in results[0]:
                    # Get the dictionary containing the actual fields (handles weird nesting)
                    raw_entity_dict = hit.entity.to_dict()
                    entity_data = raw_entity_dict.get('entity', raw_entity_dict) # Fallback to raw dict if no inner 'entity' key
                    
                    # Parse the metadata field, handle errors
                    parsed_metadata = {}
                    metadata_content = entity_data.get(CONFIG.milvus.metadata_field)
                    if isinstance(metadata_content, str):
                        try:
                            parsed_metadata = json.loads(metadata_content)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse metadata JSON string for hit {hit.id}: {metadata_content}")
                            parsed_metadata = {"error": "invalid json", "raw": metadata_content}
                    elif isinstance(metadata_content, dict): # Handle case where it's already a dict
                        parsed_metadata = metadata_content
                    # else: leave parsed_metadata as empty dict {}

                    # Construct the final result dictionary in the expected format
                    structured_result = {
                        "chunk_id": hit.id, # Use Milvus ID as chunk_id
                        "score": hit.distance,
                        "text": entity_data.get(CONFIG.milvus.text_field, ""),
                        "tags": entity_data.get(CONFIG.milvus.tags_field, []),
                        # Ensure metadata is a dict containing filename etc.
                        "metadata": {
                            "source": parsed_metadata.get("source", entity_data.get(CONFIG.milvus.filename_field, "")),
                            "filename": entity_data.get(CONFIG.milvus.filename_field, parsed_metadata.get("filename", "")),
                            "page": parsed_metadata.get("page")
                        }
                    }
                    all_results.append(structured_result) # Append the correctly structured dict

            except Exception as col_search_e:
                 logger.error(f"Error searching collection '{collection_name}': {col_search_e}", exc_info=True)
                 # Continue searching other collections
                 
        # Sort all collected results by score (distance) - lower is better for L2
        # Adjust sort order if using IP (higher is better)
        sort_reverse = (search_params_to_use.get("metric_type", "L2").upper() == "IP")
        all_results.sort(key=lambda x: x["score"], reverse=sort_reverse)
        
        # Return the top N overall results
        final_results = all_results[:limit]
        logger.info(f"Multi-collection search completed. Returning {len(final_results)} top results from {len(collection_names)} collections.")
        return final_results

    except Exception as e:
        logger.exception(f"Multi-collection search failed: {e}")
        return [] # Return empty list on error

async def disconnect_milvus():
    """Disconnect from Milvus server."""
    try:
        # Check if connection exists before trying to disconnect
        if connections.has_connection("default"):
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        else:
            logger.info("No active Milvus connection to disconnect.")
    except Exception as e:
        logger.error(f"Failed to disconnect from Milvus: {str(e)}")
        # Don't raise here, just log