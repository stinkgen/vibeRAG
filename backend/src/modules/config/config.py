from dataclasses import dataclass, field
import torch
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
import secrets # Import secrets for generating JWT secret
from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel

# Load environment variables
load_dotenv()

# --- JWT Configuration ---
# Generate a default secret if not provided in .env
# WARNING: In production, ALWAYS set a strong JWT_SECRET_KEY in your .env file!
DEFAULT_JWT_SECRET = secrets.token_urlsafe(32)

@dataclass
class AuthConfig:
    secret_key: str = os.getenv("JWT_SECRET_KEY", DEFAULT_JWT_SECRET)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24)) # Default 1 day
    secure_cookie: bool = Field(default=False, description="Set True if using HTTPS") # Added for secure cookie flag

@dataclass
class SearchConfig:
    default_limit: int = 5
    min_score: float = 1.5

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
    collection_name: str = os.getenv("MILVUS_COLLECTION", "documents")  # Collection name FOR RAG DOCS
    agent_memory_collection_name: str = os.getenv("MILVUS_AGENT_MEMORY_COLLECTION", "agent_memories") # Collection for agent memories
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
    host: str = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3")
    chat_endpoint: str = "/api/chat"
    generate_endpoint: str = "/api/generate"
    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

@dataclass
class ChatConfig:
    model: str = os.getenv("CHAT_MODEL", "llama3")
    provider: str = os.getenv("CHAT_PROVIDER", "ollama")
    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
    chunks_limit: int = int(os.getenv("CHAT_CHUNKS_LIMIT", "5"))
    history_limit: int = int(os.getenv("CHAT_HISTORY_LIMIT", "10"))
    system_prompt: str = os.getenv(
        "CHAT_SYSTEM_PROMPT", 
        "You are a helpful AI assistant. Use the provided context to answer the user's query accurately. If the context doesn't contain the answer, state that clearly."
    )

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
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", 4096)) # Reverted to 4k as upper bound

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

DEFAULT_SUMMARY_PROMPT = (
    "Based on the following conversation history (Thought/Action/Observation steps), "
    "identify and summarize the key insights, decisions made, or facts learned. "
    "Focus on information that would be valuable context for future tasks related to the overall goal. "
    "Be concise.\n\n"
    "Conversation History:\n"
    "{scratchpad_content}"
    "\n\nSummary:"
)

@dataclass
class AgentConfig:
    max_steps: int = 10
    # Add summarization config
    summarization_enabled: bool = False
    summarization_prompt_template: str = DEFAULT_SUMMARY_PROMPT
    summarization_llm_provider: Optional[str] = None # Uses default chat LLM if None
    summarization_llm_model: Optional[str] = None    # Uses default chat LLM if None
    embed_summaries: bool = True # Whether to generate/store embeddings for summaries
    memory_retrieval_limit: int = 5 # Max memories to retrieve
    # Default Research Agent
    default_research_agent_name: str = "Default Research Agent"
    default_research_agent_persona: str = "You are a thorough and objective research assistant."
    default_research_agent_goals: str = "Analyze information, identify key insights, and generate concise research reports."
    # Default Presentation Agent
    default_presentation_agent_name: str = "Default Presentation Agent"
    default_presentation_agent_persona: str = "You are a creative and engaging presentation designer."
    default_presentation_agent_goals: str = "Transform research findings or user prompts into clear and visually appealing slide presentations."

@dataclass
class CeleryConfig:
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True
    # Add other Celery settings as needed

@dataclass
class AppConfig:
    """Global configuration."""
    auth: AuthConfig = field(default_factory=AuthConfig)
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
    agent: AgentConfig = field(default_factory=AgentConfig)
    celery: CeleryConfig = field(default_factory=CeleryConfig) # Add Celery config

# --- Load Configuration --- 
def load_config() -> AppConfig:
    """Loads configuration from environment variables or defaults."""
    # ... existing config loading ...
    
    agent_config = AgentConfig(
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "10")),
        summarization_enabled=os.getenv("AGENT_SUMMARIZATION_ENABLED", "False").lower() == "true",
        summarization_prompt_template=os.getenv("AGENT_SUMMARIZATION_PROMPT", DEFAULT_SUMMARY_PROMPT),
        summarization_llm_provider=os.getenv("AGENT_SUMMARIZATION_LLM_PROVIDER", None),
        summarization_llm_model=os.getenv("AGENT_SUMMARIZATION_LLM_MODEL", None),
        embed_summaries=os.getenv("AGENT_EMBED_SUMMARIES", "True").lower() == "true",
        memory_retrieval_limit=int(os.getenv("AGENT_MEMORY_RETRIEVAL_LIMIT", "5")),
        default_research_agent_name=os.getenv("DEFAULT_RESEARCH_AGENT_NAME", "Default Research Agent"),
        default_research_agent_persona=os.getenv("DEFAULT_RESEARCH_AGENT_PERSONA", "You are a thorough and objective research assistant."),
        default_research_agent_goals=os.getenv("DEFAULT_RESEARCH_AGENT_GOALS", "Analyze information, identify key insights, and generate concise research reports."),
        default_presentation_agent_name=os.getenv("DEFAULT_PRESENTATION_AGENT_NAME", "Default Presentation Agent"),
        default_presentation_agent_persona=os.getenv("DEFAULT_PRESENTATION_AGENT_PERSONA", "You are a creative and engaging presentation designer."),
        default_presentation_agent_goals=os.getenv("DEFAULT_PRESENTATION_AGENT_GOALS", "Transform research findings or user prompts into clear and visually appealing slide presentations.")
    )
    
    milvus_config = MilvusConfig(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530")),
        dim=int(os.getenv("EMBEDDING_DIM", "384")), # Use EMBEDDING_DIM consistently?
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")), # Or derive from EmbeddingConfig?
        collection_name=os.getenv("MILVUS_COLLECTION", "documents"),
        agent_memory_collection_name=os.getenv("MILVUS_AGENT_MEMORY_COLLECTION", "agent_memories"),
        # ... load other milvus fields ...
    )
    
    celery_config = CeleryConfig(
        broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
        # Load other Celery settings if needed
    )
    
    return AppConfig(
        # ... existing AppConfig fields ...
        agent=agent_config,
        milvus=milvus_config,
        celery=celery_config,
        # ... other fields ...
    )

CONFIG = load_config()