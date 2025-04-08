from dataclasses import dataclass, field
import torch
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SearchConfig:
    default_limit: int = 5
    min_score: float = 0.5

@dataclass
class ResearchConfig:
    chunks_limit: int = int(os.getenv("CHAT_CHUNKS_LIMIT", "10"))
    model: str = os.getenv("CHAT_MODEL", "gpt-4")
    provider: str = os.getenv("CHAT_PROVIDER", "openai")
    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

@dataclass
class WebSearchConfig:
    limit: int = 5

@dataclass
class MilvusConfig:
    """Configuration for Milvus vector store."""
    host: str = os.getenv("MILVUS_HOST", "localhost")  # Milvus server host
    port: int = int(os.getenv("MILVUS_PORT", "19530"))  # Milvus server port
    dim: int = 384  # Embedding dimension
    embedding_dim: int = 384  # Alias for dim to maintain compatibility
    collection_name: str = os.getenv("MILVUS_COLLECTION", "documents")  # Collection name
    index_type: str = "HNSW"  # Index type
    metric_type: str = "L2"  # Distance metric
    default_batch_size: int = 100  # Default batch size for operations
    tags_field: str = "tags"  # Field for document tags
    tags_max_capacity: int = 10  # Maximum number of tags per document
    
    # Field names
    text_field: str = "text"  # Field name for text content
    embedding_field: str = "embedding"  # Field name for embeddings
    metadata_field: str = "metadata"  # Field name for metadata
    doc_id_field: str = "doc_id"  # Field name for document ID
    filename_field: str = "filename"  # Field name for filename
    chunk_id_field: str = "chunk_id"  # Field name for chunk ID
    
    # Field parameters for collection schema
    field_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "doc_id": {"type": "VARCHAR", "max_length": 255},  # Document ID
        "chunk_id": {"type": "VARCHAR", "max_length": 255},  # Chunk ID
        "text": {"type": "VARCHAR", "max_length": 65535},  # Text content
        "filename": {"type": "VARCHAR", "max_length": 512},  # Source filename
        "page": {"type": "INT64"},  # Page number
        "category": {"type": "VARCHAR", "max_length": 64},  # Document category
        "tags": {"type": "ARRAY", "element_type": "VARCHAR", "max_capacity": 10, "max_length": 128},  # Document tags
        "embedding": {"type": "FLOAT_VECTOR", "dim": 384}  # Text embedding
    })

    # Index parameters
    index_params: Dict[str, Any] = field(default_factory=lambda: {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {
            "M": 8,
            "efConstruction": 64
        }
    })

    # Search parameters
    search_params: Dict[str, Any] = field(default_factory=lambda: {
        "metric_type": "L2",
        "params": {"ef": 16}
    })

    # Consistency level for operations
    consistency_level: str = "Strong"

@dataclass
class OllamaConfig:
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3")
    chat_endpoint: str = "/api/chat"
    generate_endpoint: str = "/api/generate"
    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

@dataclass
class ChatConfig:
    model: str = os.getenv("CHAT_MODEL", "llama3")
    provider: str = os.getenv("CHAT_PROVIDER", "ollama")
    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
    chunks_limit: int = int(os.getenv("CHAT_CHUNKS_LIMIT", "5"))  # Maximum number of chunks to retrieve for chat

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim: int = 384  # Dimension of embeddings from all-MiniLM-L6-v2

@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")  # Set via environment variable
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model: str = os.getenv("CHAT_MODEL", "gpt-4")

@dataclass
class IngestionConfig:
    chunk_size: int = 512  # Number of tokens per chunk
    overlap: int = 50  # Number of overlapping tokens between chunks
    batch_size: int = 32  # Batch size for processing documents
    chunk_overlap: int = 50  # Alias for overlap to maintain compatibility

@dataclass
class PresentationConfig:
    """Configuration for presentation generation."""
    chunks_limit: int = int(os.getenv("CHAT_CHUNKS_LIMIT", "10"))  # Maximum number of chunks to use
    max_slides: int = 10  # Maximum number of slides
    model: str = os.getenv("CHAT_MODEL", "gpt-4")  # Model to use for generation
    provider: str = os.getenv("CHAT_PROVIDER", "openai")  # Provider to use (openai or ollama)
    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))  # Temperature for generation

@dataclass
class Config:
    """Global configuration."""
    chat: ChatConfig = field(default_factory=ChatConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    presentation: PresentationConfig = field(default_factory=PresentationConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

CONFIG = Config()