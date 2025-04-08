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