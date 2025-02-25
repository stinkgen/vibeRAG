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