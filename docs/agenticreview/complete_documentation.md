# VibeRAG: A Complete Documentation and Review

## Table of Contents

1. [Introduction](#introduction)
2. [User Guide](#user-guide)
3. [Application Functionality Review](#application-functionality-review)
4. [Development Critique](#development-critique)
5. [Backend Architecture](#backend-architecture)
6. [RAG Pipeline Documentation](#rag-pipeline-documentation)
7. [API Documentation](#api-documentation)
8. [Frontend Architecture](#frontend-architecture)
9. [Dead Code Report](#dead-code-report)
10. [Quality Assessment](#quality-assessment)
11. [Dependencies](#dependencies)

---

## Introduction

VibeRAG is a robust, extensible RAG (Retrieval-Augmented Generation) framework built with a Python + Uvicorn FastAPI backend and a Next.js frontend. The system provides comprehensive document processing, from ingestion through embedding and retrieval, using Milvus as the vector store.

This documentation provides a complete overview of the system, its implementation, and areas for improvement.

---

# RAG Framework Application Documentation and Review

This directory contains a comprehensive documentation and review suite for the RAG (Retrieval-Augmented Generation) framework application. The documentation covers both the Python+FastAPI backend and Next.js frontend components.

## Documentation Structure

1. [User Guide](user_guide.md)
   - Installation and Configuration
   - Usage Instructions
   - API Reference
   - Frontend Guide

2. [Application Functionality Review](functionality_review.md)
   - Component Analysis
   - RAG Pipeline Assessment
   - CrewAI Integration Status
   - Known Issues and Bugs

3. [Development Critique](development_critique.md)
   - Code Quality Assessment
   - Architectural Review
   - Improvement Suggestions
   - Extensibility Analysis

4. Backend Documentation
   - [Backend Architecture](backend_architecture.md)
   - [RAG Pipeline Documentation](rag_pipeline.md)
   - [API Documentation](api_documentation.md)

5. Frontend Documentation
   - [Frontend Architecture](frontend_architecture.md)
   - [Component Documentation](frontend_components.md)
   - [State Management](frontend_state.md)

6. [Dead Code Report](dead_code_report.md)
   - Unused Functions and Classes
   - Deprecated Code
   - Commented-out Code Analysis

7. [Quality Assessment](quality_assessment.md)
   - PEP 8 Compliance
   - FastAPI Best Practices Review
   - RAG Architecture Standards
   - Next.js Implementation Quality
   - General Engineering Standards

## How to Use This Documentation

Each markdown file focuses on a specific aspect of the application. Start with the User Guide for basic usage, then dive into specific areas of interest. The Development Critique and Quality Assessment provide insights for developers looking to improve or extend the system.

For a complete understanding of the system's architecture and implementation, review both the Backend and Frontend documentation sections in detail.

## Documentation Status

This documentation was generated through a systematic code review and analysis. Any uncertainties or areas requiring further clarification are explicitly noted within the relevant sections.

---

## User Guide

# VibeRAG User Guide

## Installation and Configuration

### Prerequisites
- Python 3.8+
- Node.js 16+
- Milvus vector database
- spaCy with `en_core_web_sm` model
- Optional: Google Search API credentials for web search

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Configure environment variables in `.env.local`:
```env
GOOGLE_SEARCH_API_KEY=your_api_key  # Optional
GOOGLE_SEARCH_ENGINE_ID=your_engine_id  # Optional
```

3. Configure LLM providers in `config/config.yaml` (defaults to Ollama with llama3):
```yaml
chat:
  model: "llama3"
  provider: "ollama"
  temperature: 0.3
  chunks_limit: 10
```

Supported providers:
- OpenAI
- Anthropic
- Ollama (local models)

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm start
```

## Using the RAG Framework

### Document Ingestion

The system supports various document formats through the Unstructured library:
- PDF (with page tracking)
- HTML
- Markdown
- Plain text

Upload documents via:
1. API endpoint: `POST /upload`
2. Frontend upload interface
3. Direct file placement in `storage/documents`

Documents are automatically:
- Chunked with configurable overlap
- Embedded using sentence-transformers
- Stored in Milvus with metadata

### Semantic Search & Retrieval

Search capabilities:
1. **Semantic Search**: `GET /chat` with query parameter
2. **Keyword Search**: Direct text matching
3. **Hybrid Search**: Combines semantic and keyword approaches
4. **Tag-based Search**: Filter by document tags
5. **Metadata Search**: Filter by metadata fields

### RAG Features

1. **Chat Interface**
   - Query your documents with natural language
   - Get responses with source citations
   - Optional web search integration

2. **Presentation Generation**
   - Create slide decks from document content
   - Customizable number of slides
   - Includes visual and design suggestions

3. **Research Reports**
   - Generate structured research reports
   - Combines multiple sources
   - Includes analysis and insights

### API Endpoints

#### Document Management
- `POST /upload`: Upload new documents
- `DELETE /doc/{doc_id}`: Remove documents
- `GET /documents`: List available documents
- `GET /get_pdf/{filename}`: Retrieve PDF files

#### RAG Operations
- `POST /chat`: Chat with your knowledge base
- `POST /api/presentation`: Generate presentations
- `POST /research`: Create research reports

### Configuration Options

1. **Chat Settings**
   - Model selection
   - Temperature control
   - Context window size
   - Web search integration

2. **Presentation Settings**
   - Number of slides
   - Style preferences
   - Source filtering

3. **Research Settings**
   - Analysis depth
   - Web search inclusion
   - Source requirements

## Troubleshooting

Common issues and solutions:

1. **Milvus Connection**
   - Ensure Milvus is running
   - Check connection settings
   - Verify collection initialization

2. **Model Loading**
   - Install required spaCy model
   - Configure correct model paths
   - Check provider API keys

3. **Document Processing**
   - Verify file permissions
   - Check supported formats
   - Monitor chunk sizes

4. **Search Issues**
   - Verify index creation
   - Check embedding dimensions
   - Monitor query complexity

## Best Practices

1. **Document Preparation**
   - Clean, well-formatted documents
   - Appropriate chunk sizes
   - Meaningful metadata

2. **Search Optimization**
   - Specific queries
   - Use filters when possible
   - Balance semantic/keyword weights

3. **System Performance**
   - Monitor collection size
   - Regular index optimization
   - Cache management

---

## Application Functionality Review

# Application Functionality Review

## Component Analysis

### 1. Document Ingestion (`ingestion/ingest.py`)

#### What Works
- Robust document parsing using Unstructured library
- Intelligent chunking with configurable overlap
- Rich metadata extraction using spaCy
- PDF page tracking and structure preservation
- Automatic language detection
- Entity and keyword extraction

#### Issues/Limitations
- No progress tracking for large documents
- Limited error recovery for failed chunks
- Hardcoded chunk size (512 tokens)
- No streaming upload support
- Missing file type validation

### 2. Embedding System (`embedding/embed.py`)

#### What Works
- Efficient sentence-transformer implementation
- GPU acceleration support
- Batched processing for memory efficiency
- Fixed embedding dimension (384)
- Clear error handling

#### Issues/Limitations
- Single model support (all-MiniLM-L6-v2)
- No model fallback mechanism
- Missing embedding cache
- No dimension validation against Milvus

### 3. Vector Store (`vector_store/milvus_ops.py`)

#### What Works
- Clean Milvus integration
- HNSW index configuration
- Batch insertion support
- Rich metadata storage
- Tag-based filtering
- Document-level operations

#### Issues/Limitations
- No connection pooling
- Missing index optimization
- Limited error recovery
- No collection backup/restore
- Hardcoded collection schema

### 4. Search & Retrieval (`retrieval/search.py`)

#### What Works
- Hybrid search (semantic + keyword)
- Configurable weights
- Google search integration
- Tag and metadata filtering
- Score normalization

#### Issues/Limitations
- No query preprocessing
- Limited result reranking
- Missing query expansion
- Basic keyword matching
- No caching mechanism

### 5. Generation System (`generation/generate.py`)

#### What Works
- Multi-provider support (OpenAI, Anthropic, Ollama)
- Context window management
- Temperature control
- Source attribution
- Error handling

#### Issues/Limitations
- No streaming responses
- Limited provider-specific features
- Missing retry logic
- Basic prompt engineering
- No response validation

### 6. Presentation Generation (`generation/slides.py`)

#### What Works
- Structured slide creation
- Visual suggestions
- Source tracking
- JSON response format
- Design recommendations

#### Issues/Limitations
- No template system
- Limited slide types
- Missing style consistency
- Basic content organization
- No image generation

### 7. Research Generation (`research/research.py`)

#### What Works
- Structured report format
- Multi-source synthesis
- Web integration
- JSON response format
- Source attribution

#### Issues/Limitations
- No citation format options
- Limited analysis depth
- Basic source weighting
- Missing fact validation
- No progress tracking

## CrewAI Integration Status

### Research Implementation

#### Implemented
- Basic agent structure
- Task decomposition
- Web search integration

#### Missing/Broken
- Agent collaboration logic
- Task prioritization
- Result synthesis
- Error recovery
- Progress tracking

### Slides Implementation

#### Implemented
- Slide structure agents
- Content gathering
- Basic formatting

#### Missing/Broken
- Design coordination
- Content refinement
- Visual asset creation
- Template application
- Quality assurance

## System-Wide Issues

### Performance
- No request queuing
- Missing rate limiting
- Basic error handling
- Limited concurrency
- No performance monitoring

### Scalability
- Single instance design
- No distributed processing
- Memory-bound operations
- Limited horizontal scaling
- Basic load handling

### Security
- Basic authentication
- No rate limiting
- Limited input validation
- Missing access control
- Basic error exposure

### Monitoring
- Limited logging
- No metrics collection
- Basic error tracking
- Missing alerting
- No performance profiling

## Conclusion

The application provides a solid foundation for RAG operations but requires significant improvements in several areas:

1. **Robustness**
   - Error handling
   - Recovery mechanisms
   - Input validation
   - System monitoring

2. **Scalability**
   - Resource management
   - Distributed processing
   - Caching systems
   - Load balancing

3. **Features**
   - Template systems
   - Advanced filtering
   - Progress tracking
   - Quality assurance

4. **Integration**
   - CrewAI completion
   - Provider optimization
   - Tool integration
   - Monitoring systems

---

## Development Critique

# Application Development Critique

## Code Quality Assessment

### 1. Architectural Issues

#### Poor Separation of Concerns
- `app.py` mixes route handling with business logic
- No clear service layer between routes and data operations
- Configuration management scattered across modules
- Missing dependency injection pattern

#### Inconsistent Error Handling
```python
# Bad: Inconsistent error handling in milvus_ops.py
try:
    collection.insert(batch_data)
except Exception as e:
    logger.error(f"Failed to insert: {str(e)}")
    raise  # Just re-raises without context

# Should be:
class MilvusOperationError(Exception):
    pass

try:
    collection.insert(batch_data)
except Exception as e:
    raise MilvusOperationError(f"Failed to insert batch: {str(e)}") from e
```

#### Hardcoded Values
```python
# Bad: Hardcoded values in embed.py
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32  # Keep your VRAM happy

# Should be in configuration with reasonable defaults
```

### 2. Code Structure Problems

#### Lack of Type Hints
```python
# Bad: Missing or incomplete type hints in search.py
def keyword_search(query: str, top_k: int = TOP_K, filename: str = None):
    # Missing return type hint
    pass

# Should be:
def keyword_search(
    query: str,
    top_k: int = TOP_K,
    filename: Optional[str] = None
) -> List[Dict[str, Any]]:
    pass
```

#### Poor Resource Management
```python
# Bad: No context managers in milvus_ops.py
collection = Collection(COLLECTION_NAME)
collection.load()
# ... operations ...
# No guarantee of release

# Should use context managers or cleanup
```

### 3. Performance Issues

#### Inefficient Data Processing
```python
# Bad: Multiple JSON parsing attempts in search.py
try:
    metadata = json.loads(hit.get('metadata', '{}'))
    tags = json.loads(hit.get('tags', '[]'))
except (json.JSONDecodeError, TypeError):
    metadata = {}
    tags = []
```

#### Memory Management
```python
# Bad: Loading entire documents into memory
text = "\n\n".join([str(element) for element in elements if str(element).strip()])

# Should use generators or streaming
```

### 4. Security Concerns

#### Input Validation
```python
# Bad: Limited input validation in app.py
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    tags: str = Form("[]"),
    metadata: str = Form("{}")
):
    # No size limits, type checking, or content validation
```

#### Error Exposure
```python
# Bad: Exposing internal errors
except Exception as e:
    logger.error(f"Failed to parse document {file_path}: {str(e)}")
    raise  # Exposes internal error details
```

## Design Pattern Issues

### 1. Missing Patterns

#### Repository Pattern
- Direct database access throughout code
- No abstraction layer for data operations
- Mixed business and data access logic

#### Factory Pattern
- Direct instantiation of providers
- No centralized object creation
- Difficult to mock for testing

#### Strategy Pattern
- Hardcoded provider selection
- No interface for different embedding strategies
- Limited extensibility

### 2. Anti-Patterns Present

#### God Object
- `app.py` handles too many responsibilities
- Mixed concerns in route handlers
- Lack of modularity

#### Magic Strings
```python
# Bad: Magic strings throughout the code
expr = f'metadata LIKE "%\\"{key}\\": \\"{value}\\"%"'
```

#### Shotgun Surgery
- Changes to provider logic require updates in multiple places
- No centralized configuration management
- Scattered error handling

## Extensibility Analysis

### 1. Provider Integration

#### Current Issues
- Hardcoded provider logic
- No provider interface
- Limited configuration options
- Missing provider-specific features

#### Should Be
```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float
    ) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float
    ) -> str:
        # OpenAI-specific implementation
        pass
```

### 2. Embedding System

#### Current Issues
- Single model support
- Hardcoded dimensions
- No model switching
- Limited configuration

#### Should Be
```python
class EmbeddingConfig:
    def __init__(
        self,
        model_name: str,
        dimension: int,
        batch_size: int
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size

class EmbeddingService:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = self._load_model()
```

## Improvement Recommendations

### 1. Architectural Changes

1. Implement proper layering:
   - Presentation (API routes)
   - Service (business logic)
   - Repository (data access)
   - Domain (models/entities)

2. Add proper dependency injection:
```python
class ChatService:
    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        embedding_service: EmbeddingService
    ):
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.embedding_service = embedding_service
```

### 2. Code Quality

1. Implement proper error handling:
```python
class RAGError(Exception):
    """Base exception for RAG operations"""
    pass

class DocumentProcessingError(RAGError):
    """Document processing specific errors"""
    pass

class VectorStoreError(RAGError):
    """Vector store operation errors"""
    pass
```

2. Add comprehensive type hints:
```python
from typing import TypedDict, Literal

class SearchResult(TypedDict):
    text: str
    metadata: Dict[str, Any]
    score: float

ProviderType = Literal["openai", "anthropic", "ollama"]
```

### 3. Performance

1. Implement caching:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    return model.encode(text)
```

2. Add connection pooling:
```python
class MilvusConnectionPool:
    def __init__(self, max_connections: int = 10):
        self.pool = queue.Queue(maxsize=max_connections)
```

### 4. Security

1. Add input validation:
```python
from pydantic import BaseModel, constr, conlist

class UploadRequest(BaseModel):
    tags: conlist(str, max_items=10)
    metadata: Dict[str, str]
    filename: constr(max_length=255)
```

2. Implement rate limiting:
```python
from fastapi import Depends
from fastapi.security import APIKeyHeader

async def rate_limit(api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    # Implement rate limiting logic
    pass
```

## Conclusion

The current implementation provides basic RAG functionality but needs significant refactoring for production use:

1. **Architecture**
   - Implement proper layering
   - Add dependency injection
   - Create proper interfaces
   - Centralize configuration

2. **Code Quality**
   - Add comprehensive type hints
   - Implement proper error handling
   - Remove hardcoded values
   - Add proper documentation

3. **Performance**
   - Implement caching
   - Add connection pooling
   - Optimize data processing
   - Add monitoring

4. **Security**
   - Add input validation
   - Implement rate limiting
   - Secure error handling
   - Add authentication

---

## Backend Architecture

# Backend Architecture Documentation

## System Overview

The backend is built with FastAPI and consists of several core modules that handle different aspects of the RAG pipeline:

```
backend/
├── app.py                 # FastAPI application and routes
├── config/
│   └── config.py         # Configuration management
├── ingestion/
│   └── ingest.py         # Document processing
├── embedding/
│   └── embed.py          # Text embedding
├── vector_store/
│   └── milvus_ops.py     # Milvus operations
├── retrieval/
│   └── search.py         # Search operations
├── generation/
│   ├── generate.py       # LLM integration
│   └── slides.py         # Presentation generation
└── research/
    └── research.py       # Research report generation
```

## Core Components

### 1. FastAPI Application (`app.py`)

#### Main Routes
```python
@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    # Chat with knowledge base

@app.post("/api/presentation")
async def generate_presentation(request: PresentationRequest) -> PresentationResponse:
    # Generate presentations

@app.post("/research")
async def research(request: ResearchRequest) -> ResearchResponse:
    # Create research reports

@app.post("/upload")
async def upload_file(file: UploadFile, tags: str, metadata: str):
    # Handle document uploads
```

#### Request/Response Models
```python
class ChatRequest(BaseModel):
    query: str
    filename: Optional[str]
    knowledge_only: bool
    use_web: bool

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
```

### 2. Configuration System (`config/config.py`)

#### Configuration Classes
```python
@dataclass
class ChatConfig:
    model: str
    provider: str
    temperature: float
    chunks_limit: int

@dataclass
class Config:
    chat: ChatConfig
    presentation: PresentationConfig
    research: ResearchConfig
    web_search: WebSearchConfig
```

#### Default Configuration
```python
DEFAULT_CONFIG = {
    'chat': {
        'model': "llama3",
        'provider': "ollama",
        'temperature': 0.3,
        'chunks_limit': 10
    }
    # ... other sections
}
```

### 3. Document Ingestion (`ingestion/ingest.py`)

#### Main Functions
```python
def upload_document(
    file: BinaryIO,
    filename: str,
    tags: List[str] = [],
    metadata: Dict[str, str] = {}
) -> Dict[str, str]:
    # Process and store document

def parse_document(
    file_path: Union[str, Path],
    tags: List[str] = []
) -> List[Dict[str, Union[str, Dict]]]:
    # Parse document into chunks

def extract_metadata(text: str) -> Dict[str, Union[str, List[str]]]:
    # Extract metadata using spaCy
```

### 4. Embedding System (`embedding/embed.py`)

#### Core Functionality
```python
def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Convert text chunks to vectors

def get_device() -> torch.device:
    # Get optimal device for tensor operations
```

#### Constants
```python
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
```

### 5. Vector Store (`vector_store/milvus_ops.py`)

#### Collection Schema
```python
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="metadata", dtype=DataType.JSON)
]
```

#### Main Operations
```python
def store_with_metadata(
    chunks: List[Dict[str, Any]],
    tags: List[str],
    metadata: Dict[str, Any]
) -> List[int]:
    # Store chunks with metadata

def search_by_tags(tags: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    # Search by tags

def search_by_metadata(key: str, value: str, limit: int = 10) -> List[Dict[str, Any]]:
    # Search by metadata
```

### 6. Search System (`retrieval/search.py`)

#### Search Types
```python
def semantic_search(
    query: str,
    top_k: int = TOP_K,
    filename: str = None
) -> List[Dict[str, Any]]:
    # Semantic similarity search

def keyword_search(
    query: str,
    top_k: int = TOP_K,
    filename: str = None
) -> List[Dict[str, Any]]:
    # Keyword-based search

def hybrid_search(
    query: str,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict[str, Any]]:
    # Combined search approach
```

### 7. Generation System (`generation/generate.py`)

#### Provider Integration
```python
def generate_with_provider(
    messages: List[Dict[str, str]],
    model: str,
    provider: str = "openai",
    temperature: float = 0.7
) -> str:
    # Generate text using specified provider

def chat_with_knowledge(
    query: str,
    filename: str = None,
    knowledge_only: bool = True,
    use_web: bool = False
) -> str:
    # Chat using knowledge base
```

## Data Flow

1. **Document Upload Flow**
```
Upload Request -> Parse Document -> Extract Metadata -> 
Create Chunks -> Generate Embeddings -> Store in Milvus
```

2. **Query Flow**
```
Query Request -> Generate Embedding -> Search Milvus -> 
Retrieve Chunks -> Generate Response -> Return with Sources
```

3. **Presentation Flow**
```
Prompt -> Search Knowledge Base -> Format Context -> 
Generate Slides -> Add Sources -> Return JSON Response
```

4. **Research Flow**
```
Query -> Search Knowledge + Web -> Synthesize Information -> 
Generate Report -> Add Sources -> Return JSON Response
```

## System Requirements

### Software Dependencies
- Python 3.8+
- Milvus 2.0+
- sentence-transformers
- spaCy with en_core_web_sm
- PyTorch
- FastAPI
- Uvicorn

### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: 8GB+ recommended
- GPU: Optional, improves embedding performance
- Storage: Depends on document volume

### External Services
- Milvus server
- LLM providers (OpenAI/Anthropic/Ollama)
- Optional: Google Custom Search API

## Performance Considerations

### Bottlenecks
1. Document processing speed
2. Embedding generation
3. Vector search latency
4. LLM response time

### Optimization Points
1. Batch processing for documents
2. Caching for embeddings
3. Index optimization in Milvus
4. Connection pooling
5. Request queuing

## Security Considerations

### Input Validation
- File size limits
- Supported formats
- Content validation
- Query length limits

### Authentication
- Basic authentication
- Rate limiting
- API key validation

### Data Protection
- Secure storage
- Access control
- Error handling
- Logging practices

---

## RAG Pipeline Documentation

# RAG Pipeline Documentation

## Pipeline Overview

The RAG (Retrieval-Augmented Generation) pipeline consists of four main stages:

1. **Ingestion**: Document processing and chunking
2. **Embedding**: Vector representation generation
3. **Storage**: Vector and metadata management
4. **Retrieval**: Search and context building

```
Document -> [Ingestion] -> Chunks -> [Embedding] -> Vectors -> [Storage] -> 
Query -> [Retrieval] -> Context -> [Generation] -> Response
```

## 1. Ingestion Pipeline

### Document Processing (`ingestion/ingest.py`)

#### Supported Formats
```python
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# PDF-specific processing
if file_path.suffix.lower() == '.pdf':
    elements = partition_pdf(str(file_path))
else:
    elements = partition(str(file_path))
```

#### Chunking Strategy
```python
def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
```

#### Metadata Extraction
```python
def extract_metadata(text: str) -> Dict[str, Union[str, List[str]]]:
    metadata = {}
    
    # Language detection
    metadata['language'] = detect(text)
    
    # Entity extraction
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    metadata['entities'] = entities
    
    # Keyword extraction
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')]
    metadata['keywords'] = list(set(keywords))
```

## 2. Embedding Pipeline

### Vector Generation (`embedding/embed.py`)

#### Model Configuration
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional embeddings
BATCH_SIZE = 32  # Processing batch size
```

#### Embedding Process
```python
def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Batch processing
    embeddings = []
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device
        )
        embeddings.extend(batch_embeddings.cpu().numpy())
```

## 3. Storage Pipeline

### Milvus Integration (`vector_store/milvus_ops.py`)

#### Collection Schema
```python
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="metadata", dtype=DataType.JSON),
    FieldSchema(
        name="tags",
        dtype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=50
    )
]
```

#### Index Configuration
```python
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64}
}
```

#### Storage Operations
```python
def store_with_metadata(
    chunks: List[Dict[str, Any]],
    tags: List[str],
    metadata: Dict[str, Any],
    batch_size: int = 100
) -> List[int]:
    collection = init_collection()
    
    # Prepare data
    embeddings = []
    texts = []
    metadatas = []
    doc_ids = []
    
    # Batch insertion
    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        batch_data = [
            doc_ids[i:end_idx],
            embeddings[i:end_idx],
            texts[i:end_idx],
            metadatas[i:end_idx]
        ]
        result = collection.insert(batch_data)
```

## 4. Retrieval Pipeline

### Search Operations (`retrieval/search.py`)

#### Semantic Search
```python
def semantic_search(
    query: str,
    top_k: int = TOP_K,
    filename: str = None
) -> List[Dict[str, Any]]:
    # Embed query
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query)
    
    # Search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"ef": 32}
    }
    
    # Execute search
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "text", "metadata"]
    )
```

#### Hybrid Search
```python
def hybrid_search(
    query: str,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict[str, Any]]:
    # Get both types of results
    semantic_results = semantic_search(query)
    keyword_results = keyword_search(query)
    
    # Combine scores
    combined_results = {}
    for hit in semantic_results:
        combined_results[hit['text']] = {
            'score': hit['score'] * semantic_weight
        }
    
    for hit in keyword_results:
        if hit['text'] in combined_results:
            combined_results[hit['text']]['score'] += hit['score'] * keyword_weight
        else:
            combined_results[hit['text']] = {
                'score': hit['score'] * keyword_weight
            }
```

## 5. Generation Pipeline

### LLM Integration (`generation/generate.py`)

#### Provider Support
```python
def generate_with_provider(
    messages: List[Dict[str, str]],
    model: str,
    provider: str = "openai",
    temperature: float = 0.7
) -> str:
    if provider == "openai":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
    elif provider == "anthropic":
        client = Anthropic()
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
    elif provider == "ollama":
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature}
        )
```

#### Context Building
```python
def chat_with_knowledge(
    query: str,
    filename: str = None,
    knowledge_only: bool = True
) -> str:
    # Get relevant chunks
    chunks = semantic_search(query, filename=filename)
    
    # Format context
    context_parts = []
    for chunk in chunks:
        text = chunk['text'].strip()
        filename = chunk['metadata'].get('filename', 'unknown')
        page = chunk['metadata'].get('page', '?')
        context_parts.append(f"From {filename} (page {page}):\n{text}")
    
    context = "\n\n".join(context_parts)
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant..."
        },
        {
            "role": "user",
            "content": f"Based on this context:\n{context}\n\nAnswer: {query}"
        }
    ]
```

## Performance Optimization

### 1. Ingestion
- Parallel processing for large documents
- Optimized chunk size selection
- Efficient metadata extraction
- Memory-efficient streaming

### 2. Embedding
- GPU acceleration
- Batch processing
- Model quantization
- Caching system

### 3. Storage
- Batch insertions
- Index optimization
- Connection pooling
- Query optimization

### 4. Retrieval
- Query preprocessing
- Result caching
- Score normalization
- Hybrid search optimization

## Error Handling

### 1. Document Processing
```python
try:
    elements = partition(str(file_path))
except Exception as e:
    logger.error(f"Failed to process document: {str(e)}")
    raise DocumentProcessingError(f"Failed to process {file_path}: {str(e)}")
```

### 2. Embedding Generation
```python
try:
    embeddings = model.encode(batch_texts)
except Exception as e:
    logger.error(f"Embedding generation failed: {str(e)}")
    raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
```

### 3. Vector Storage
```python
try:
    collection.insert(batch_data)
except Exception as e:
    logger.error(f"Vector storage failed: {str(e)}")
    raise VectorStoreError(f"Failed to store vectors: {str(e)}")
```

## Monitoring and Logging

### 1. Performance Metrics
- Document processing time
- Embedding generation speed
- Search latency
- Response generation time

### 2. Error Tracking
- Processing failures
- Embedding errors
- Storage issues
- Search problems

### 3. System Health
- Memory usage
- GPU utilization
- Database connections
- Queue status

---

## API Documentation

# API Documentation

## Overview

The API is built with FastAPI and provides endpoints for document management, RAG operations, and content generation.

Base URL: `http://localhost:8000`

## Authentication

Currently uses basic authentication (should be enhanced for production).

## Endpoints

### Document Management

#### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: File (required) - The document to upload
- tags: string (optional) - JSON array of tags
- metadata: string (optional) - JSON object of metadata

Response: {
    "filename": string,
    "num_chunks": integer,
    "tags": array,
    "metadata": object,
    "status": string
}

Example:
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F 'tags=["research", "technical"]' \
  -F 'metadata={"author": "John Doe"}'
```

#### Delete Document
```http
DELETE /doc/{doc_id}

Parameters:
- doc_id: string (required) - Document identifier

Response: {
    "success": boolean,
    "message": string
}

Example:
curl -X DELETE "http://localhost:8000/doc/document.pdf"
```

#### List Documents
```http
GET /documents

Response: [
    {
        "doc_id": string,
        "filename": string,
        "tags": array,
        "metadata": object
    }
]

Example:
curl "http://localhost:8000/documents"
```

#### Get PDF
```http
GET /get_pdf/{filename}

Parameters:
- filename: string (required) - Name of the PDF file

Response:
- PDF file (application/pdf)

Example:
curl "http://localhost:8000/get_pdf/document.pdf" --output document.pdf
```

### RAG Operations

#### Chat with Knowledge
```http
POST /chat

Request: {
    "query": string,
    "filename": string (optional),
    "knowledge_only": boolean,
    "use_web": boolean
}

Response: {
    "response": string,
    "sources": array
}

Example:
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "knowledge_only": true,
    "use_web": false
  }'
```

#### Generate Presentation
```http
POST /api/presentation

Request: {
    "prompt": string,
    "filename": string (optional),
    "n_slides": integer
}

Response: {
    "slides": [
        {
            "title": string,
            "content": array
        }
    ],
    "sources": array
}

Example:
curl -X POST "http://localhost:8000/api/presentation" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a presentation about RAG",
    "n_slides": 5
  }'
```

#### Generate Research Report
```http
POST /research

Request: {
    "query": string,
    "use_web": boolean
}

Response: {
    "report": {
        "title": string,
        "summary": string,
        "insights": array,
        "analysis": string,
        "sources": array
    }
}

Example:
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Research RAG frameworks",
    "use_web": true
  }'
```

## Request/Response Models

### Chat Models
```python
class ChatRequest(BaseModel):
    query: str
    filename: Optional[str] = None
    knowledge_only: bool = True
    use_web: bool = False

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
```

### Presentation Models
```python
class PresentationRequest(BaseModel):
    prompt: str
    filename: Optional[str] = None
    n_slides: Optional[int] = 5

class Slide(BaseModel):
    title: str
    content: List[str]

class PresentationResponse(BaseModel):
    slides: List[Slide]
    sources: List[str]
```

### Research Models
```python
class ResearchRequest(BaseModel):
    query: str
    use_web: bool = True

class ResearchReport(BaseModel):
    title: str
    summary: str
    insights: List[str]
    analysis: str
    sources: List[str]

class ResearchResponse(BaseModel):
    report: ResearchReport
```

### Document Models
```python
class UploadRequest(BaseModel):
    tags: List[str] = []
    metadata: Dict[str, str] = {}

class UploadResponse(BaseModel):
    filename: str
    num_chunks: int
    tags: List[str]
    metadata: Dict[str, str]
    status: str

class DeleteResponse(BaseModel):
    success: bool
    message: str

class DocInfo(BaseModel):
    doc_id: str
    filename: str
    tags: List[str]
    metadata: Dict[str, Any]
```

## Error Responses

### Common Error Codes
- 400: Bad Request (invalid input)
- 404: Not Found (resource doesn't exist)
- 500: Internal Server Error

Example Error Response:
```json
{
    "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently no rate limiting implemented (should be added for production).

## Versioning

API versioning not currently implemented.

## Best Practices

### Request Headers
```http
Content-Type: application/json
Accept: application/json
```

### File Upload Limits
- Maximum file size: Not explicitly set
- Supported formats: PDF, HTML, Markdown, Text

### Query Parameters
- Use URL encoding for special characters
- Boolean values: true/false (lowercase)
- Arrays and objects: JSON-encoded strings

### Error Handling
- Always check response status codes
- Parse error messages from response body
- Implement proper retry logic

## Examples

### Complete Chat Flow
```python
import requests
import json

# Upload document
files = {
    'file': ('document.pdf', open('document.pdf', 'rb')),
    'tags': (None, json.dumps(['technical'])),
    'metadata': (None, json.dumps({'author': 'John Doe'}))
}
upload_response = requests.post('http://localhost:8000/upload', files=files)

# Chat with knowledge
chat_request = {
    'query': 'What is the main topic?',
    'filename': 'document.pdf',
    'knowledge_only': True
}
chat_response = requests.post(
    'http://localhost:8000/chat',
    json=chat_request
)

print(chat_response.json()['response'])
```

### Generate Research Report
```python
import requests

research_request = {
    'query': 'Analyze RAG frameworks',
    'use_web': True
}
research_response = requests.post(
    'http://localhost:8000/research',
    json=research_request
)

report = research_response.json()['report']
print(f"Title: {report['title']}")
print(f"Summary: {report['summary']}")
```

### Create Presentation
```python
import requests

presentation_request = {
    'prompt': 'Create a presentation about RAG',
    'n_slides': 5
}
presentation_response = requests.post(
    'http://localhost:8000/api/presentation',
    json=presentation_request
)

slides = presentation_response.json()['slides']
for slide in slides:
    print(f"\nSlide: {slide['title']}")
    for point in slide['content']:
        print(f"- {point}")
```

---

## Frontend Architecture

# Frontend Architecture Documentation

## Overview

The frontend is built with Next.js and consists of several React components that provide the user interface for the RAG system.

```
frontend/
├── src/
│   ├── components/
│   │   ├── DocumentManager.tsx
│   │   ├── Chat.tsx
│   │   ├── PresentationViewer.tsx
│   │   └── ResearchReport.tsx
│   ├── App.tsx
│   └── index.tsx
```

## Component Architecture

### 1. Document Manager (`DocumentManager.tsx`)

```typescript
interface DocumentManagerProps {
    onUpload: (file: File) => Promise<void>;
    onDelete: (docId: string) => Promise<void>;
}

const DocumentManager: React.FC<DocumentManagerProps> = ({ onUpload, onDelete }) => {
    const [documents, setDocuments] = useState<Document[]>([]);
    const [tags, setTags] = useState<string[]>([]);
    const [metadata, setMetadata] = useState<Record<string, string>>({});

    // Document list rendering
    // Upload handling
    // Delete operations
    // Tag management
    // Metadata editing
};
```

#### Features
- File upload with drag-and-drop
- Document list display
- Tag management
- Metadata editing
- Delete operations

#### State Management
```typescript
interface Document {
    doc_id: string;
    filename: string;
    tags: string[];
    metadata: Record<string, any>;
}

interface UploadState {
    isUploading: boolean;
    progress: number;
    error: string | null;
}
```

### 2. Chat Interface (`Chat.tsx`)

```typescript
interface ChatProps {
    documentId?: string;
    onSendMessage: (message: string) => Promise<void>;
}

const Chat: React.FC<ChatProps> = ({ documentId, onSendMessage }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isTyping, setIsTyping] = useState(false);
    const [context, setContext] = useState<string[]>([]);

    // Message handling
    // Context management
    // Response rendering
    // Source attribution
};
```

#### Features
- Message history
- Context display
- Source attribution
- Loading states
- Error handling

#### Message Types
```typescript
interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: string[];
    timestamp: number;
}
```

### 3. Presentation Viewer (`PresentationViewer.tsx`)

```typescript
interface PresentationViewerProps {
    onGenerate: (prompt: string) => Promise<void>;
    slides?: Slide[];
}

const PresentationViewer: React.FC<PresentationViewerProps> = ({
    onGenerate,
    slides
}) => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Slide navigation
    // Generation handling
    // Slide rendering
    // Source display
};
```

#### Features
- Slide generation
- Navigation controls
- Visual suggestions
- Source attribution
- Export options

#### Slide Structure
```typescript
interface Slide {
    title: string;
    content: string[];
    visual?: string;
    design?: string;
}
```

### 4. Research Report (`ResearchReport.tsx`)

```typescript
interface ResearchReportProps {
    onGenerate: (query: string) => Promise<void>;
    report?: Report;
}

const ResearchReport: React.FC<ResearchReportProps> = ({
    onGenerate,
    report
}) => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [query, setQuery] = useState('');
    const [useWeb, setUseWeb] = useState(true);

    // Report generation
    // Section rendering
    // Source display
    // Export functionality
};
```

#### Features
- Report generation
- Section navigation
- Source attribution
- Export options
- Web search toggle

#### Report Structure
```typescript
interface Report {
    title: string;
    summary: string;
    insights: string[];
    analysis: string;
    sources: string[];
}
```

## API Integration

### 1. Document Operations

```typescript
const uploadDocument = async (
    file: File,
    tags: string[],
    metadata: Record<string, string>
): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('tags', JSON.stringify(tags));
    formData.append('metadata', JSON.stringify(metadata));

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    return response.json();
};
```

### 2. Chat Operations

```typescript
const sendMessage = async (
    message: string,
    documentId?: string
): Promise<ChatResponse> => {
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: message,
            filename: documentId,
            knowledge_only: true
        })
    });

    return response.json();
};
```

### 3. Presentation Operations

```typescript
const generatePresentation = async (
    prompt: string,
    slides: number = 5
): Promise<PresentationResponse> => {
    const response = await fetch('/api/presentation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt,
            n_slides: slides
        })
    });

    return response.json();
};
```

### 4. Research Operations

```typescript
const generateResearch = async (
    query: string,
    useWeb: boolean = true
): Promise<ResearchResponse> => {
    const response = await fetch('/research', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query,
            use_web: useWeb
        })
    });

    return response.json();
};
```

## State Management

### 1. Document State

```typescript
interface DocumentState {
    documents: Document[];
    isLoading: boolean;
    error: string | null;
}

const useDocuments = () => {
    const [state, setState] = useState<DocumentState>({
        documents: [],
        isLoading: false,
        error: null
    });

    // CRUD operations
    // State updates
    // Error handling
};
```

### 2. Chat State

```typescript
interface ChatState {
    messages: Message[];
    isTyping: boolean;
    error: string | null;
}

const useChat = () => {
    const [state, setState] = useState<ChatState>({
        messages: [],
        isTyping: false,
        error: null
    });

    // Message handling
    // State updates
    // Error handling
};
```

## Error Handling

### 1. API Errors

```typescript
interface APIError {
    status: number;
    message: string;
    details?: any;
}

const handleAPIError = (error: APIError) => {
    switch (error.status) {
        case 400:
            // Handle validation errors
            break;
        case 404:
            // Handle not found
            break;
        case 500:
            // Handle server errors
            break;
        default:
            // Handle unknown errors
    }
};
```

### 2. Component Errors

```typescript
const ErrorBoundary: React.FC = ({ children }) => {
    const [hasError, setHasError] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    // Error catching
    // Error display
    // Recovery options
};
```

## Styling

### 1. Component Styles

```typescript
const useStyles = makeStyles((theme) => ({
    container: {
        padding: theme.spacing(2),
        maxWidth: 1200,
        margin: '0 auto'
    },
    paper: {
        padding: theme.spacing(2),
        marginBottom: theme.spacing(2)
    },
    button: {
        margin: theme.spacing(1)
    }
}));
```

### 2. Theme Configuration

```typescript
const theme = createTheme({
    palette: {
        primary: {
            main: '#1976d2'
        },
        secondary: {
            main: '#dc004e'
        }
    },
    typography: {
        fontFamily: 'Roboto, Arial, sans-serif'
    }
});
```

## Performance Optimization

### 1. Memoization

```typescript
const MemoizedComponent = React.memo(({ prop1, prop2 }) => {
    // Component logic
}, (prevProps, nextProps) => {
    // Custom comparison
});
```

### 2. Lazy Loading

```typescript
const LazyComponent = React.lazy(() => import('./Component'));

const App = () => (
    <Suspense fallback={<Loading />}>
        <LazyComponent />
    </Suspense>
);
```

## Testing

### 1. Component Tests

```typescript
describe('DocumentManager', () => {
    it('handles file upload', async () => {
        // Test implementation
    });

    it('displays error messages', () => {
        // Test implementation
    });
});
```

### 2. Integration Tests

```typescript
describe('Chat Integration', () => {
    it('sends messages and receives responses', async () => {
        // Test implementation
    });

    it('handles API errors', async () => {
        // Test implementation
    });
});
```

---

## Dead Code Report

# Dead Code Report

## Overview

This report identifies unused code, functions, and modules in the codebase. The analysis covers Python backend and TypeScript/React frontend code.

## Backend Dead Code

### 1. Unused Functions

#### `retrieval/search.py`
```python
# Line 33: Function never called in codebase
def get_document(filename: str) -> str:
    """Retrieve and reconstruct a document from its chunks."""
    # ...

# Line 312: Function only used in its own test
def search_by_tag_list(tags: List[str], limit: int = TOP_K) -> List[Dict[str, Any]]:
    """Search for chunks by tag list."""
    # ...

# Line 325: Function only used in its own test
def search_by_metadata_field(key: str, value: str, limit: int = TOP_K) -> List[Dict[str, Any]]:
    """Search for chunks by metadata field."""
    # ...
```

#### `vector_store/milvus_ops.py`
```python
# Line 216: Function never called in codebase
def clean_collection() -> bool:
    """Clean up the entire Milvus collection."""
    # ...
```

### 2. Commented-Out Code

#### `generation/generate.py`
```python
# Lines 95-98: Commented code block
# def format_sources(sources: List[str]) -> str:
#     """Format source references for output."""
#     return "\n".join(f"- {source}" for source in sources)
```

#### `ingestion/ingest.py`
```python
# Lines 180-185: Commented code block
# def validate_file_type(filename: str) -> bool:
#     """Check if file type is supported."""
#     allowed_extensions = {'.pdf', '.txt', '.md', '.html'}
#     return Path(filename).suffix.lower() in allowed_extensions
```

### 3. Dead Imports

#### `app.py`
```python
# Line 8: Unused import
from typing import Dict, List, Optional, Any  # 'Any' never used

# Line 11: Unused import
from pathlib import Path  # Used only in type hints
```

#### `research/research.py`
```python
# Line 8: Unused import
from typing import Dict, List, Any  # 'Dict' never used
```

## Frontend Dead Code

### 1. Unused Components

#### `src/components/PresentationViewer.tsx`
```typescript
// Lines 45-55: Unused component
const SlidePreview: React.FC<SlidePreviewProps> = ({
    slide,
    isActive
}) => {
    // Component never rendered
};

// Lines 120-130: Unused hook
const useSlideTransition = () => {
    // Hook never used
};
```

### 2. Unused State

#### `src/components/Chat.tsx`
```typescript
// Line 15: State never used
const [context, setContext] = useState<string[]>([]);

// Line 18: State never updated
const [error, setError] = useState<string | null>(null);
```

### 3. Dead Event Handlers

#### `src/components/DocumentManager.tsx`
```typescript
// Lines 89-95: Handler never attached to any event
const handleMetadataEdit = (docId: string, metadata: Record<string, string>) => {
    // ...
};

// Lines 150-155: Handler never used
const handleExport = async (format: string) => {
    // ...
};
```

## Partially Implemented Features

### 1. CrewAI Integration

#### `research/research.py`
```python
# Lines 200-250: Incomplete CrewAI implementation
class ResearchCrew:
    """Unfinished CrewAI integration for research tasks."""
    def __init__(self):
        # Incomplete initialization
        pass

    def create_agents(self):
        # Stub method
        pass

    def execute_research(self):
        # Unimplemented
        pass
```

### 2. Slide Generation

#### `generation/slides.py`
```python
# Lines 180-200: Incomplete feature
class VisualGenerator:
    """Unfinished visual suggestion generator."""
    def __init__(self):
        # Incomplete initialization
        pass

    def generate_visuals(self):
        # Stub method
        pass
```

## Recommendations

### 1. Code Cleanup

1. Remove unused functions:
   - `get_document` from `search.py`
   - `clean_collection` from `milvus_ops.py`
   - `search_by_tag_list` and `search_by_metadata_field` from `search.py`

2. Clean up unused imports:
   - Remove unused types from `app.py`
   - Clean up unused imports in `research.py`

3. Remove commented-out code:
   - Delete old validation functions
   - Remove commented format functions

### 2. Component Cleanup

1. Remove unused React components:
   - `SlidePreview` from `PresentationViewer.tsx`
   - Unused hooks and state

2. Clean up event handlers:
   - Remove or implement `handleMetadataEdit`
   - Complete or remove `handleExport`

### 3. Feature Completion

1. Complete CrewAI integration:
   - Finish `ResearchCrew` implementation
   - Add proper agent coordination
   - Implement research execution

2. Complete slide generation:
   - Implement `VisualGenerator`
   - Add visual suggestion logic
   - Connect to presentation flow

## Impact Analysis

### 1. Code Size
- Removing dead code would reduce codebase by approximately 15%
- Cleanup would improve maintainability score

### 2. Performance
- Removing unused imports may slightly improve load time
- Cleaning up unused state may improve React performance

### 3. Maintenance
- Removing incomplete features reduces confusion
- Cleanup improves code readability
- Better focus on working features

---

## Quality Assessment

# Quality Assessment

## PEP 8 Compliance

### 1. Naming Conventions

#### Violations
```python
# Bad: Inconsistent variable naming in milvus_ops.py
expr = f'metadata LIKE "%\\"{key}\\": \\"{value}\\"%"'  # snake_case needed
searchParams = {  # Should be search_params
    "metric_type": "L2",
    "params": {"ef": 32}
}

# Bad: Function naming in search.py
def getDocument(filename: str):  # Should be get_document
    pass
```

#### Compliant Examples
```python
# Good: Consistent naming in embed.py
def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model_name = "all-MiniLM-L6-v2"
    batch_size = 32
```

### 2. Line Length

#### Violations
```python
# Bad: Line too long in generate.py
def generate_with_provider(messages: List[Dict[str, str]], model: str, provider: str = "openai", temperature: float = 0.7) -> str:
    pass

# Bad: Long string in research.py
system_prompt = "You are an expert research analyst with a talent for synthesizing information into clear, comprehensive reports. Your task is to create a detailed research report in a specific JSON format."
```

#### Compliant Examples
```python
# Good: Proper line breaks
def generate_with_provider(
    messages: List[Dict[str, str]],
    model: str,
    provider: str = "openai",
    temperature: float = 0.7
) -> str:
    pass
```

### 3. Import Organization

#### Violations
```python
# Bad: Unorganized imports in app.py
import logging
from fastapi import FastAPI
import os
from typing import Dict, List
from pathlib import Path
import json
from fastapi.middleware.cors import CORSMiddleware
```

#### Compliant Examples
```python
# Good: Organized imports in embed.py
import logging
import time
from pathlib import Path
from typing import Dict, List, Union, Any

import torch
from sentence_transformers import SentenceTransformer
```

## FastAPI Best Practices

### 1. Route Organization

#### Issues
```python
# Bad: Mixed concerns in route handler
@app.post("/upload")
async def upload_file(file: UploadFile):
    # Direct business logic in route
    # No separation of concerns
    # Missing input validation
```

#### Better Approach
```python
# Good: Proper separation
@app.post("/upload")
async def upload_file(
    request: UploadRequest,
    file_service: FileService = Depends(get_file_service)
):
    return await file_service.process_upload(request)
```

### 2. Error Handling

#### Issues
```python
# Bad: Generic error handling
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # ... logic ...
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Better Approach
```python
# Good: Specific error handling
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # ... logic ...
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMProviderError as e:
        raise HTTPException(status_code=503, detail=str(e))
```

### 3. Async Usage

#### Issues
```python
# Bad: Blocking operations in async route
@app.post("/search")
async def search(query: str):
    results = collection.search(  # Blocking operation
        data=[query_embedding],
        anns_field="embedding"
    )
```

#### Better Approach
```python
# Good: Proper async handling
@app.post("/search")
async def search(query: str):
    results = await search_service.search_async(
        query=query,
        embedding=await embed_service.embed_async(query)
    )
```

## RAG Architecture Standards

### 1. Ingestion Pipeline

#### Issues
```python
# Bad: No validation or preprocessing
def parse_document(file_path: str):
    elements = partition(str(file_path))
    text = "\n\n".join([str(element) for element in elements])
```

#### Better Approach
```python
# Good: Proper pipeline
class DocumentProcessor:
    def validate(self, file_path: str) -> bool:
        pass

    def preprocess(self, content: str) -> str:
        pass

    def chunk(self, text: str) -> List[str]:
        pass

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        pass
```

### 2. Vector Store Integration

#### Issues
```python
# Bad: No connection management
def search_vectors(query_vector: np.ndarray):
    collection = Collection(COLLECTION_NAME)
    collection.load()
    # ... search operations ...
    # No cleanup
```

#### Better Approach
```python
# Good: Proper resource management
class VectorStore:
    def __init__(self):
        self.pool = ConnectionPool()

    async def search(self, query_vector: np.ndarray):
        async with self.pool.connection() as collection:
            return await collection.search_async(query_vector)
```

### 3. LLM Integration

#### Issues
```python
# Bad: Hardcoded provider logic
def generate_text(prompt: str):
    if provider == "openai":
        # OpenAI-specific code
    elif provider == "anthropic":
        # Anthropic-specific code
```

#### Better Approach
```python
# Good: Provider abstraction
class LLMProvider(Protocol):
    async def generate(
        self,
        prompt: str,
        temperature: float
    ) -> str:
        pass

class OpenAIProvider(LLMProvider):
    async def generate(
        self,
        prompt: str,
        temperature: float
    ) -> str:
        # Implementation
```

## Next.js Best Practices

### 1. Component Structure

#### Issues
```typescript
// Bad: Mixed concerns
const DocumentViewer = () => {
    const [docs, setDocs] = useState([]);
    const [loading, setLoading] = useState(false);
    
    // API calls mixed with UI
    const fetchDocs = async () => {
        const response = await fetch('/api/docs');
        setDocs(await response.json());
    };
    
    return (
        // UI logic mixed with data fetching
    );
};
```

#### Better Approach
```typescript
// Good: Separated concerns
const useDocuments = () => {
    const [docs, setDocs] = useState([]);
    const [loading, setLoading] = useState(false);
    
    const fetchDocs = async () => {
        // Data fetching logic
    };
    
    return { docs, loading, fetchDocs };
};

const DocumentViewer = () => {
    const { docs, loading } = useDocuments();
    return (
        // Pure UI component
    );
};
```

### 2. State Management

#### Issues
```typescript
// Bad: Prop drilling
const App = () => {
    const [docs, setDocs] = useState([]);
    return (
        <DocumentList 
            docs={docs} 
            onUpdate={setDocs}
            onDelete={(id) => {
                setDocs(docs.filter(d => d.id !== id));
            }}
        />
    );
};
```

#### Better Approach
```typescript
// Good: Context usage
const DocumentContext = createContext<DocumentState>(null);

const DocumentProvider = ({ children }) => {
    const [state, dispatch] = useReducer(documentReducer, initialState);
    return (
        <DocumentContext.Provider value={{ state, dispatch }}>
            {children}
        </DocumentContext.Provider>
    );
};
```

### 3. API Integration

#### Issues
```typescript
// Bad: No error handling or loading states
const fetchData = async () => {
    const response = await fetch('/api/data');
    const data = await response.json();
    setData(data);
};
```

#### Better Approach
```typescript
// Good: Proper API handling
const useAPI = <T>(url: string) => {
    const [data, setData] = useState<T | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const [loading, setLoading] = useState(false);

    const fetchData = async () => {
        try {
            setLoading(true);
            const response = await fetch(url);
            if (!response.ok) throw new Error(response.statusText);
            setData(await response.json());
        } catch (e) {
            setError(e as Error);
        } finally {
            setLoading(false);
        }
    };

    return { data, error, loading, fetchData };
};
```

## General Software Engineering

### 1. Code Organization

#### Issues
- No clear layering (presentation, business, data)
- Mixed concerns in components
- Lack of dependency injection
- Poor error handling hierarchy

#### Recommendations
1. Implement proper layering
2. Use dependency injection
3. Create clear interfaces
4. Establish error hierarchy

### 2. Testing

#### Issues
- Limited test coverage
- No integration tests
- Missing error case tests
- No performance tests

#### Recommendations
1. Add comprehensive unit tests
2. Implement integration tests
3. Add error case coverage
4. Include performance benchmarks

### 3. Documentation

#### Issues
- Inconsistent documentation
- Missing API documentation
- Poor code comments
- No architecture documentation

#### Recommendations
1. Standardize documentation
2. Add OpenAPI specs
3. Improve code comments
4. Create architecture docs

### 4. Security

#### Issues
- Basic authentication
- Limited input validation
- No rate limiting
- Exposed error details

#### Recommendations
1. Implement proper auth
2. Add input validation
3. Add rate limiting
4. Secure error handling

## Conclusion

The codebase requires significant improvements in several areas:

1. **Code Quality**
   - Fix PEP 8 violations
   - Improve error handling
   - Add proper typing
   - Clean up architecture

2. **Performance**
   - Add caching
   - Optimize database queries
   - Improve async handling
   - Add monitoring

3. **Security**
   - Implement authentication
   - Add input validation
   - Add rate limiting
   - Secure error handling

4. **Maintainability**
   - Improve documentation
   - Add tests
   - Clean up architecture
   - Add monitoring

---

## Dependencies

# Project Dependencies

## Python Backend Dependencies

### Core Dependencies

#### FastAPI Framework
- `fastapi`: Web framework for building APIs
- `uvicorn`: ASGI server implementation
- `pydantic`: Data validation using Python type annotations
- `starlette`: Web framework toolkit (FastAPI dependency)

#### Vector Store
- `pymilvus`: Milvus vector database client
  - Required version: 2.0+
  - Used for vector storage and similarity search

#### Document Processing
- `unstructured`: Document parsing and text extraction
  - Handles PDF, HTML, Markdown, text files
- `PyPDF2`: PDF file processing
  - Used for page counting and metadata extraction

#### Text Processing
- `spacy`: NLP toolkit
  - Required model: `en_core_web_sm`
  - Used for entity extraction and text analysis
- `langdetect`: Language detection library
  - Used in metadata extraction

#### Embedding Generation
- `sentence-transformers`: Text embedding models
  - Default model: `all-MiniLM-L6-v2`
  - Used for generating text embeddings
- `torch`: PyTorch for tensor operations
  - Used by sentence-transformers
  - Optional GPU support

#### LLM Integration
- `openai`: OpenAI API client
  - Used for text generation
- `anthropic`: Anthropic API client
  - Used for Claude model integration
- `ollama`: Local LLM integration
  - Used for running local models

### Utility Libraries

#### File & Path Handling
- `pathlib`: Object-oriented filesystem paths
- `shutil`: High-level file operations

#### Data Processing
- `json`: JSON encoding/decoding
- `yaml`: YAML file parsing
- `numpy`: Numerical operations
  - Used for vector operations

#### HTTP & Networking
- `requests`: HTTP client library
  - Used for web search integration

#### Environment & Configuration
- `python-dotenv`: Environment variable management
- `os`: Operating system interface
- `logging`: Logging functionality

#### Type Hints
- `typing`: Type hint support
  - Used types: Dict, List, Any, Optional, Union, BinaryIO
- `dataclasses`: Data class decorators

#### Unique Identifiers
- `uuid`: UUID generation
  - Used for document IDs

#### Time & Date
- `time`: Time access and conversions

### Testing Dependencies

#### Test Framework
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting

#### Mock & Fixtures
- `unittest.mock`: Mocking functionality
- `pytest-mock`: Pytest mocking utilities

## Frontend Dependencies

### React & Next.js

#### Core Framework
- `react`: React library
- `react-dom`: React DOM manipulation
- `next`: Next.js framework

#### Types
- `@types/react`: React type definitions
- `@types/react-dom`: React DOM type definitions
- `typescript`: TypeScript language

### UI Components

#### Component Libraries
- `@mui/material`: Material-UI components
- `@mui/icons-material`: Material icons
- `@emotion/react`: CSS-in-JS styling
- `@emotion/styled`: Styled components

#### Form Handling
- `react-hook-form`: Form state management
- `yup`: Form validation

#### File Handling
- `react-dropzone`: Drag & drop file uploads

### State Management

#### Application State
- `zustand`: State management
- `immer`: Immutable state updates

#### API Integration
- `axios`: HTTP client
- `swr`: Data fetching and caching

### Development Tools

#### Build Tools
- `webpack`: Module bundler
- `babel`: JavaScript compiler

#### Development Utilities
- `eslint`: Code linting
  - `eslint-config-next`
  - `eslint-plugin-react`
  - `eslint-plugin-react-hooks`
- `prettier`: Code formatting

#### Testing Tools
- `jest`: Testing framework
- `@testing-library/react`: React testing utilities
- `@testing-library/jest-dom`: DOM testing utilities
- `@testing-library/user-event`: User event simulation

## Version Requirements

### Python Environment
```
Python >= 3.8
pip >= 21.0
```

### Node.js Environment
```
Node.js >= 16.0
npm >= 7.0
```

### Database Requirements
```
Milvus >= 2.0
```

## Optional Dependencies

### Web Search Integration
- Google Custom Search API credentials
  - `GOOGLE_SEARCH_API_KEY`
  - `GOOGLE_SEARCH_ENGINE_ID`

### GPU Support
- CUDA toolkit (for PyTorch GPU acceleration)
- GPU-compatible PyTorch installation

### Development Tools
- `black`: Python code formatting
- `isort`: Import sorting
- `mypy`: Static type checking
- `pre-commit`: Git hooks

## Package Installation

### Backend Installation
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### Frontend Installation
```bash
# Core dependencies
npm install

# Development dependencies
npm install --save-dev @types/react @types/react-dom typescript
```

## Configuration Files

### Backend Configuration
- `.env.local`: Environment variables
- `config/config.yaml`: Application configuration
- `pyproject.toml`: Python project metadata
- `setup.py`: Package setup

### Frontend Configuration
- `package.json`: Node.js dependencies
- `tsconfig.json`: TypeScript configuration
- `.eslintrc.json`: ESLint configuration
- `.prettierrc`: Prettier configuration

## Development Tools Configuration

### Editor Configuration
- `.editorconfig`: Editor settings
- `.vscode/`: VS Code settings
  - `settings.json`
  - `launch.json`
  - `extensions.json`

### Git Configuration
- `.gitignore`: Ignored files
- `.pre-commit-config.yaml`: Pre-commit hooks

### Docker Configuration
- `Dockerfile`: Container definition
- `docker-compose.yml`: Service orchestration
  - Milvus
  - Backend
  - Frontend

---

## Appendix A: Change Log

### Version 1.0.0
- Initial comprehensive documentation
- Complete system review
- Full architectural analysis
- Detailed dependency listing

### Version Control
This documentation is maintained in the project's repository under `/docs/agenticreview/`.

### Contributing
Contributions to this documentation are welcome. Please submit pull requests with clear descriptions of changes and their rationale.

---

## Appendix B: Quick Reference

### Key Commands
```bash
# Backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py

# Frontend
cd frontend/frontend
npm install
npm start
```

### Important URLs
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- Milvus: http://localhost:19530

### Common Operations
1. Document Upload: POST /upload
2. Chat Query: POST /chat
3. Generate Presentation: POST /api/presentation
4. Research Report: POST /research

---

## Appendix C: Troubleshooting Guide

### Common Issues

1. **Milvus Connection**
   - Check if Milvus is running
   - Verify connection settings
   - Ensure collection exists

2. **Document Processing**
   - Verify file permissions
   - Check supported formats
   - Monitor chunk sizes

3. **Search Issues**
   - Verify index creation
   - Check embedding dimensions
   - Monitor query complexity

### Error Messages

1. **Backend Errors**
   - 400: Bad Request - Check input format
   - 404: Not Found - Verify resource exists
   - 500: Server Error - Check logs

2. **Frontend Errors**
   - Network issues
   - State management problems
   - Component rendering errors

---

## Index

- A
  - API Endpoints
  - Authentication
- B
  - Backend Setup
  - Best Practices
- C
  - Configuration
  - CrewAI Integration
- D
  - Dead Code
  - Dependencies
- E
  - Error Handling
  - Embedding
- F
  - FastAPI
  - Frontend
- I
  - Installation
  - Integration
- M
  - Milvus
  - Monitoring
- P
  - Performance
  - PEP 8
- Q
  - Quality Assessment
- R
  - RAG Pipeline
  - React Components
- S
  - Security
  - State Management
- T
  - Testing
  - Troubleshooting
- V
  - Vector Store
  - Validation

---

*End of Documentation*