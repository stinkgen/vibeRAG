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