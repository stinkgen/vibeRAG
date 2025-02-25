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