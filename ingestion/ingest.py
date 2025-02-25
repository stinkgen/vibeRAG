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
from unstructured.documents.elements import Text, Title, ElementMetadata
from transformers import GPT2TokenizerFast
from embedding.embed import embed_chunks
from vector_store.milvus_ops import store_with_metadata
from config.config import CONFIG  # Config's in the house! 🏠

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
DOCS_DIR = Path("storage/documents")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

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
    file: BinaryIO,
    filename: str,
    tags: List[str] = [],
    metadata: Dict[str, str] = {}
) -> UploadResponse:
    """Upload and process a document with tags and metadata.
    
    Args:
        file: File-like object containing the document
        filename: Name of the file
        tags: Optional list of tags to associate with the document
        metadata: Optional metadata dictionary
        
    Returns:
        Dictionary with document info and status
    """
    try:
        # Create unique path and save file
        doc_path = DOCS_DIR / filename
        with open(doc_path, 'wb') as f:
            shutil.copyfileobj(file, f)
        
        logger.info(f"Doc uploaded to {doc_path}—storage secured! 📁")
        
        # Add file info to metadata
        file_metadata: ChunkMetadata = {
            'filename': filename,
            'file_type': doc_path.suffix.lower()[1:],  # Remove dot
            'tags': tags
        }
        
        # Parse document into chunks
        try:
            chunks = parse_document(doc_path, tags=tags)
            logger.info(f"Doc parsed—{len(chunks)} chunks ready! 📝")
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk_metadata = cast(ChunkMetadata, chunk['metadata'])
                chunk_metadata.update(file_metadata)
            
            # Embed chunks
            embedded_chunks = embed_chunks(cast(List[Dict[str, Any]], chunks))
            logger.info("Chunks embedded—vectors dropping! 🎯")
            
            # Store in Milvus with metadata
            chunk_ids = store_with_metadata(embedded_chunks, tags, cast(Dict[str, str], metadata))
            logger.info(f"Doc stored with {len(chunk_ids)} chunks—metadata locked! 🔒")
            
            return {
                'filename': filename,
                'num_chunks': len(chunks),
                'tags': tags,
                'metadata': metadata,
                'status': 'success'
            }
            
        except Exception as e:
            # If parsing fails, clean up the uploaded file
            doc_path.unlink(missing_ok=True)
            logger.error(f"Processing failed for {filename}: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")
            
    except Exception as e:
        logger.error(f"Upload failed for {filename}: {str(e)}")
        raise Exception(f"Upload failed: {str(e)}")

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
    file_path: Union[str, Path],
    tags: List[str] = []
) -> List[ChunkDict]:
    """Parse a document and extract text with metadata.
    
    Args:
        file_path: Path to the document to parse
        tags: Optional list of tags to add to metadata
        
    Returns:
        List of dictionaries containing text chunks and associated metadata
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Parsing document: {file_path}")
    
    try:
        # Parse document based on type
        if file_path.suffix.lower() == '.pdf':
            # Use partition_pdf for PDFs to get page numbers
            elements = partition_pdf(str(file_path))
            result: List[ChunkDict] = []
            
            # First, group elements by page
            pages: Dict[int, List[str]] = {}
            current_page = 1
            
            for element in elements:
                if not isinstance(element, (Text, Title)):
                    continue
                
                # Try to get page number from metadata
                try:
                    if hasattr(element, 'metadata'):
                        element_metadata = cast(ElementMetadata, element.metadata)
                        if hasattr(element_metadata, 'page_number'):
                            current_page = getattr(element_metadata, 'page_number', 1)
                except Exception as e:
                    logger.debug(f"Could not get page number from metadata: {str(e)}")
                
                # Initialize page text list if needed
                if current_page not in pages:
                    pages[current_page] = []
                
                # Add element text to page
                text = str(element)
                if text.strip():
                    pages[current_page].append(text)
            
            # Now process each page
            for page_num, page_texts in pages.items():
                # Combine all text from the page
                page_text = "\n".join(page_texts)
                if not page_text.strip():
                    continue
                
                # Extract metadata at the page level
                extracted_metadata = extract_metadata(page_text)
                base_metadata: BaseMetadata = {
                    'filename': file_path.name,
                    'file_type': file_path.suffix.lower()[1:]
                }
                
                # Create chunk metadata with optional fields
                chunk_metadata: ChunkMetadata = {
                    **base_metadata,  # Include required base fields
                    'title': file_path.stem,
                    'page': page_num,
                    'tags': tags,
                    'language': extracted_metadata['language'],
                    'entities': extracted_metadata['entities'],
                    'keywords': extracted_metadata['keywords']
                }
                
                # Chunk the page text
                text_chunks = chunk_text(page_text)
                
                # Create chunk entries with page metadata
                for chunk in text_chunks:
                    if chunk.strip():  # Only add non-empty chunks
                        result.append({
                            'text': chunk,
                            'metadata': chunk_metadata
                        })
            
            return result

        else:
            # Handle other file types
            elements = partition(str(file_path))
            text = "\n\n".join([str(element) for element in elements if str(element).strip()])
            
            if not text.strip():
                raise ValueError("No text content found in document")
            
            # Extract rich metadata
            extracted_metadata_other = extract_metadata(text)
            
            # Create base metadata for non-PDF
            base_metadata_other: BaseMetadata = {
                'filename': file_path.name,
                'file_type': file_path.suffix.lower()[1:]
            }
            
            # Create chunk metadata with optional fields for non-PDF
            chunk_metadata_other: ChunkMetadata = {
                **base_metadata_other,  # Include required base fields
                'title': file_path.stem,
                'page': 1,  # Non-PDF documents are treated as single page
                'tags': tags,
                'language': extracted_metadata_other['language'],
                'entities': extracted_metadata_other['entities'],
                'keywords': extracted_metadata_other['keywords']
            }
            
            # Chunk text
            chunks = chunk_text(text)
            result = [{
                'text': chunk,
                'metadata': chunk_metadata_other
            } for chunk in chunks if chunk.strip()]
            
            logger.info(f"Document chunked—{len(result)} pieces ready! 📄")
        
        if not result:
            raise ValueError("No valid chunks extracted from document")
            
        if tags:
            logger.info(f"Tagged {file_path.name} with {tags}—vibe sorted! 🏷️")
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to parse document {file_path}: {str(e)}")
        raise 

# Types locked—code's sharp as fuck! 🔥 