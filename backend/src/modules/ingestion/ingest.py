"""Document ingestion module for parsing various file types and extracting metadata.

This module handles parsing of PDF, HTML, Markdown, and text files using unstructured,
extracts metadata using spaCy and langdetect, and chunks text for RAG applications.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO, TypedDict, Any, cast, Sequence, NotRequired
import spacy
from langdetect import detect
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Title, ElementMetadata, Element
from transformers import GPT2TokenizerFast
from src.modules.embedding.embed import embed_chunks
from src.modules.vector_store.milvus_ops import (
    store_with_metadata, 
    get_user_collection_name, 
    get_admin_collection_name, 
    get_global_collection_name,
    init_collection # Ensure init_collection is imported for potential pre-checks
)
from src.modules.config.config import CONFIG  # Config's in the house! 🏠
from src.modules.auth.database import User # Import User model
from fastapi import HTTPException, UploadFile # Import UploadFile
from pymilvus import Collection, utility # Add Collection, utility
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("Yo, run 'python -m spacy download en_core_web_sm' first!")
    raise

# Initialize tokenizer for chunking
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Constants
# REMOVED DOCS_DIR - Not needed if parsing directly from memory
# DOCS_DIR = Path("storage/documents")
# DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Type definitions—4090's precision-tuned! 🔥
class ExtractedMetadata(TypedDict):
    """Metadata extracted from text—keeping it tight! 💪"""
    language: str
    entities: Dict[str, List[str]]
    keywords: List[str]

class BaseMetadata(TypedDict):
    """Base metadata fields—always present! 🎯"""
    filename: str
    file_type: str

class ChunkMetadata(BaseMetadata, total=False):
    """Metadata for a text chunk—structured AF! 🎯"""
    title: str
    page: int
    tags: List[str]
    language: str
    entities: Dict[str, List[str]]
    keywords: List[str]

class ChunkDict(TypedDict):
    """Text chunk with metadata—precision data! 💪"""
    text: str
    metadata: ChunkMetadata

class UploadResponse(TypedDict):
    """Response from document upload—clean types! 🚀"""
    filename: str
    num_chunks: int
    tags: List[str]
    metadata: Dict[str, str]
    status: str

def upload_document(
    file: UploadFile, # Expect UploadFile directly from FastAPI
    filename: str, 
    user: User, 
    target_collection_type: str = 'user',
    tags: List[str] = None, 
    metadata: Dict = None
) -> Dict:
    """Process and store a document in the appropriate Milvus collection.
    
    Args:
        file: FastAPI UploadFile object
        filename: Original filename
        user: The authenticated user object.
        target_collection_type: 'user' (default) or 'global'. Admins can target global.
        tags: List of tags
        metadata: Dictionary of metadata
        
    Returns:
        Dict: Status of the upload
    """
    logger.info(f"Starting document upload process for {filename} by user {user.username} (ID: {user.id})")
    
    # Determine target collection name
    collection_name: str
    if user.role == 'admin':
        if target_collection_type == 'global':
            collection_name = get_global_collection_name()
            logger.info(f"Admin {user.username} targeting GLOBAL collection: {collection_name}")
        else:
            collection_name = get_admin_collection_name()
            logger.info(f"Admin {user.username} targeting personal collection: {collection_name}")
    else:
        # Regular users always target their own collection
        collection_name = get_user_collection_name(user.id)
        logger.info(f"User {user.username} targeting personal collection: {collection_name}")
        # Ensure target_collection_type isn't misused by non-admins
        if target_collection_type == 'global':
             logger.warning(f"User {user.username} attempted to target global collection, overriding to personal.")
             target_collection_type = 'user' # Force to user collection
             
    # Ensure the target collection exists before proceeding
    try:
        # 1. Ensure connection and collection exist
        ensure_connection() 
        if not utility.has_collection(collection_name):
            logger.error(f"Target collection '{collection_name}' does not exist and upload cannot create it. Use init_collection separately.")
            # Depending on requirements, we might want to auto-create it here using init_rag_collection(collection_name)?
            # For now, raise error if collection must pre-exist.
            raise HTTPException(status_code=500, detail=f"Target collection '{collection_name}' does not exist.")
        else:
             logger.info(f"Target collection '{collection_name}' found.")
             # Load the collection
             collection = Collection(collection_name)
             collection.load()
             logger.info(f"Loaded collection '{collection_name}' for ingestion.")

        # 2. Delete existing chunks for this document to prevent duplicates
        delete_document(collection_name=collection_name, filename=filename)
        logger.info(f"Removed any existing chunks for '{filename}' from '{collection_name}'.")

        # 3. Parse Document (Pass the UploadFile object directly)
        logger.info(f"Parsing document: {filename}")
        elements = parse_document(file, filename)
        logger.info(f"Parsed {len(elements)} elements from {filename}")
        
        # Check if parsing returned any elements
        if not elements:
            logger.warning(f"Document {filename} parsing yielded no elements. Skipping ingestion.")
            # Return a specific status or raise an error?
            return {
                "filename": filename,
                "num_chunks": 0,
                "tags": tags or [],
                "metadata": metadata or {},
                "status": f"Ingestion skipped: No content parsed from document."
            }

        # 4. Extract Metadata (Optional - Skip for now to simplify)
        # extracted_meta = extract_metadata(" ".join(el.text for el in elements))
        # combined_metadata = {**(metadata or {}), **extracted_meta}
        combined_metadata = metadata or {}
        combined_metadata['filename'] = filename # Ensure filename is in metadata
        logger.info(f"Using combined metadata: {combined_metadata}")

        # 5. Chunk Text (Using combined text from elements)
        logger.info("Chunking text...")
        # Combine text, handling potential None values if elements are empty
        text_content = "\n\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])
        chunks_texts = chunk_text(text_content)
        logger.info(f"Created {len(chunks_texts)} chunks.")

        # Prepare chunks with metadata from elements if available
        chunks_for_embedding = []
        chunk_counter = 0
        for text_chunk in chunks_texts:
            chunk_meta = {
                'filename': filename, 
                'chunk_id': f"{filename}_chunk_{chunk_counter}" # Simple chunk ID
            }
            # Attempt to find corresponding element metadata (simplistic mapping)
            # A better approach might involve mapping based on character offsets
            # For now, just add the basic filename and chunk ID
            chunks_for_embedding.append({'text': text_chunk, 'metadata': chunk_meta})
            chunk_counter += 1

        # 6. Embed Chunks
        logger.info("Generating embeddings...")
        embedded_chunks = embed_chunks(chunks_for_embedding)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks.")

        # 7. Store in Milvus
        logger.info(f"Storing chunks in collection: {collection_name}")
        store_with_metadata(
            collection_name=collection_name, 
            chunks=embedded_chunks, 
            tags=tags, 
            metadata=metadata or {}, # Pass the initial metadata provided in request
            filename=filename 
        )
        logger.info(f"Successfully stored chunks for {filename} in {collection_name}.")

        return {
            "filename": filename,
            "num_chunks": len(embedded_chunks),
            "tags": tags or [],
            "metadata": metadata or {},
            "status": f"Successfully ingested into collection '{collection_name}'"
        }

    except Exception as e:
        logger.exception(f"Error during document ingestion pipeline for {filename}: {e}")
        # Attempt to cleanup uploaded file? Or rely on external cleanup?
        raise HTTPException(status_code=500, detail=f"Error processing document {filename}: {str(e)}")

def extract_metadata(text: str) -> ExtractedMetadata:
    """Extract metadata from text using spaCy and langdetect.
    
    Args:
        text: The input text to analyze
        
    Returns:
        Dictionary containing extracted metadata with language, entities, and keywords
    """
    metadata: ExtractedMetadata = {
        'language': 'unknown',
        'entities': {},
        'keywords': []
    }
    
    try:
        # Detect language
        metadata['language'] = detect(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}")
    
    try:
        # Extract entities and keywords using spaCy
        doc = nlp(text)
        
        # Extract named entities
        entities: Dict[str, List[str]] = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        metadata['entities'] = entities
        
        # Extract keywords (nouns and proper nouns)
        keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')]
        metadata['keywords'] = list(set(keywords))  # Remove duplicates
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {str(e)}")
    
    return metadata

def chunk_text(text: str) -> List[str]:
    """Chunk text into smaller pieces with overlap.
    
    Args:
        text: The input text to chunk
        
    Returns:
        List of text chunks, sized by config
    """
    # Hardcoding's dead—config's running the chunk game! 🔥
    tokens = tokenizer.encode(text)
    chunks: List[str] = []
    
    if len(tokens) <= CONFIG.ingestion.chunk_size:
        return [text]
    
    for i in range(0, len(tokens), CONFIG.ingestion.chunk_size - CONFIG.ingestion.overlap):
        chunk_tokens = tokens[i:i + CONFIG.ingestion.chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    logger.info(f"Sliced {len(tokens)} tokens into {len(chunks)} chunks—size {CONFIG.ingestion.chunk_size}, overlap {CONFIG.ingestion.overlap}, smooth as butter! 🔪")
    return chunks

def parse_document(
    # Changed first argument to accept UploadFile or BinaryIO
    file: Union[UploadFile, BinaryIO], 
    filename: str # Keep filename for metadata
) -> List[Element]: # Return unstructured Elements
    """Parse document content using unstructured, handling different file types.
    
    Args:
        file: File-like object (UploadFile or BinaryIO)
        filename: Original filename for context/metadata
        
    Returns:
        List of unstructured Elements
    """
    logger.info(f"Parsing file: {filename}")
    file_content_type = None
    file_object = None

    # Handle UploadFile vs BinaryIO (e.g., from testing)
    if hasattr(file, 'content_type') and callable(getattr(file, 'read')):
        upload_file = cast(UploadFile, file)
        file_content_type = upload_file.content_type
        # Reset pointer and read into memory for unstructured
        upload_file.file.seek(0) 
        # Pass the internal file object to partition
        file_object = upload_file.file 
        logger.debug(f"Parsing UploadFile: {filename}, Content-Type: {file_content_type}")
    elif hasattr(file, 'read'):
        # Assuming BinaryIO or similar file-like object
        file_object = cast(BinaryIO, file)
        # Try to infer content type from filename extension
        ext = Path(filename).suffix.lower()
        if ext == '.pdf':
            file_content_type = 'application/pdf'
        elif ext == '.txt':
            file_content_type = 'text/plain'
        elif ext == '.md':
            file_content_type = 'text/markdown'
        # Add more types as needed
        logger.debug(f"Parsing BinaryIO: {filename}, Inferred Type: {file_content_type}")
    else:
        raise ValueError("Unsupported file input type for parsing.")

    try:
        # Use unstructured.partition directly with the file object
        # Pass filename for potential metadata, and content_type if known
        elements = partition(
            file=file_object, 
            file_filename=filename, 
            content_type=file_content_type,
            # Add strategy="fast" for PDFs if desired, or handle PDF separately
            # strategy="hi_res" for better PDF parsing if needed (requires extra deps)
             pdf_infer_table_structure=True, # Example: enable table extraction
             include_page_breaks=True # Keep page breaks
        )
        logger.info(f"Successfully parsed {len(elements)} elements from {filename}.")
        return elements
    except Exception as e:
        logger.error(f"Failed to parse document {filename} using unstructured: {e}", exc_info=True)
        raise ValueError(f"Could not parse document: {e}")

# Types locked—code's sharp as fuck! 🔥 