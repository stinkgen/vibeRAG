"""FastAPI backend for VibeRAG - serving up knowledge with style.

This module provides REST endpoints for chat, presentation, and research functionality,
complete with request validation and swagger docs that slap.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from pathlib import Path
import json
import asyncio
import socket
import uvicorn
from contextlib import asynccontextmanager
import aiohttp

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, constr
from dotenv import load_dotenv
from pymilvus import Collection

# from frontend.backend.services.chat_service import ChatService, RAGError, SearchError, GenerationError # Commented out - remove later
from src.modules.generation.slides import create_presentation # Fixed import
from src.modules.research.research import create_research_report # Fixed import
from src.modules.vector_store.milvus_ops import connect_milvus, init_collection, delete_document, ensure_connection # Fixed import
from src.modules.config.config import CONFIG # Fixed import
from src.modules.ingestion.ingest import upload_document # Fixed import
from src.modules.generation.generate import chat_with_knowledge # Import the refactored chat function

# Load env vars from root .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env.local")) # Fixed path

# Configure logging with swagger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections and collections on startup."""
    try:
        # Ensure Milvus connection
        ensure_connection()
        logger.info("Connected to Milvus—waiting for it to be ready...")
        
        # Give Milvus a sec to get its shit together
        await asyncio.sleep(2)
        
        # Initialize collection
        init_collection()
        logger.info("Initialized Milvus connection and collection 🚀")
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {str(e)}")
    
    yield  # App is running here
    
    # Cleanup (if needed)
    logger.info("Shutting down—cleanup time! 🧹")

# Initialize FastAPI with some metadata and lifespan
app = FastAPI(
    title="VibeRAG API",
    description="Streaming chat with your docs—4090's cooking! 🔥",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
STORAGE_DIR = Path("storage/documents") # This path might need adjustment depending on where the app runs vs. where storage/ is. Assuming CWD is project root for now.
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize services
# chat_service = ChatService( # Commented out - remove later
#     model=CONFIG.chat.model,
#     provider=CONFIG.chat.provider,
#     temperature=CONFIG.chat.temperature
# )

# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request with that query heat."""
    query: constr(min_length=1) = Field(..., description="The query to search for—gotta be non-empty!")
    filename: Optional[str] = None
    knowledge_only: bool = True
    use_web: bool = False
    # tags: Optional[List[str]] = Field(None, description="Optional tags to filter chunks by—slice it up! 🔥") # Tags/Metadata filter not implemented in chat_with_knowledge yet
    # metadata: Optional[Dict[str, str]] = Field(None, description="Optional metadata to filter chunks by—filter gang! 💪") # Tags/Metadata filter not implemented in chat_with_knowledge yet
    stream: bool = Field(True, description="Whether to stream the response—real-time vibes! 🎵 Defaulting to True.")
    model: Optional[str] = Field(None, description="The model to use for generation—default uses config value")
    provider: Optional[str] = Field(None, description="The provider to use (ollama or openai)—default uses config value")

# Non-streaming response model (if needed)
class NonStreamChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    """Chat response dropping knowledge."""
    response: str
    sources: List[str]

class PresentationRequest(BaseModel):
    """Request for generating a presentation."""
    prompt: str
    filename: Optional[str] = None
    n_slides: Optional[int] = 5
    model: Optional[str] = None
    provider: Optional[str] = None

class Slide(BaseModel):
    """Single slide with title and content that slaps."""
    title: str
    content: List[str]

class PresentationResponse(BaseModel):
    """Response containing generated slides and sources."""
    slides: List[Slide]
    sources: List[str]

class ResearchRequest(BaseModel):
    """Research request looking for insights."""
    query: str
    use_web: bool = True

class ResearchReport(BaseModel):
    """Structured research report with all the goods."""
    title: str
    summary: str
    insights: List[str]
    analysis: str
    sources: List[str]

class ResearchResponse(BaseModel):
    """Research response packed with knowledge."""
    report: ResearchReport

class UploadRequest(BaseModel):
    """Document upload request with tags and metadata."""
    tags: List[str] = []
    metadata: Dict[str, str] = {}

class UploadResponse(BaseModel):
    """Document upload response with status."""
    filename: str
    num_chunks: int
    tags: List[str]
    metadata: Dict[str, str]
    status: str

class DeleteResponse(BaseModel):
    """Document deletion response."""
    success: bool
    message: str

class DocInfo(BaseModel):
    """Document info with all the metadata vibes."""
    doc_id: str
    filename: str
    tags: List[str]
    metadata: Dict[str, Any]

@app.get("/health")
def health_check():
    """Health check—system's vibing! 🎯"""
    return {"status": "healthy", "message": "4090's ready to shred! 🔥"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests, supporting both streaming and non-streaming modes."""

    # Define the streaming generator function
    async def stream_generator():
        try:
            async for event in chat_with_knowledge(
                query=request.query,
                filename=request.filename,
                knowledge_only=request.knowledge_only,
                use_web=request.use_web,
                model=request.model, # Pass through model/provider if specified
                provider=request.provider,
                temperature=CONFIG.chat.temperature # Use configured temp
            ):
                # Format as Server-Sent Event (SSE)
                yield f"data: {json.dumps(event)}\n\n"
            # Send a final empty message or a specific 'end' event if needed
            # yield "data: {\"type\": \"end\"}\n\n"
        except Exception as e:
            logger.exception(f"Error during chat stream generation: {e}")
            error_event = {"type": "error", "data": f"An internal error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_event)}\n\n"

    # Handle streaming request
    if request.stream:
        logger.info(f"Streaming chat request received: query='{request.query[:50]}...'")
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    # Handle non-streaming request
    else:
        logger.info(f"Non-streaming chat request received: query='{request.query[:50]}...'")
        final_response = ""
        sources_data = []
        try:
            async for event in chat_with_knowledge(
                query=request.query,
                filename=request.filename,
                knowledge_only=request.knowledge_only,
                use_web=request.use_web,
                model=request.model,
                provider=request.provider,
                temperature=CONFIG.chat.temperature
            ):
                if event.get("type") == "sources":
                    sources_data = event.get("data", [])
                elif event.get("type") == "response":
                    final_response += event.get("data", "")
                elif event.get("type") == "error":
                    # If an error occurs mid-stream in non-streaming mode, return it as an HTTP error
                    raise HTTPException(status_code=500, detail=event.get("data", "Unknown generation error"))
            
            return NonStreamChatResponse(response=final_response, sources=sources_data)
        
        except HTTPException as http_exc:
            # Re-raise HTTP exceptions caused by errors during generation
            raise http_exc
        except Exception as e:
            logger.exception(f"Error processing non-streaming chat request: {e}")
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/chat")
async def chat_get(
    query: str = Query(..., description="The query to search for"),
    filename: Optional[str] = Query(None, description="Optional filename to filter chunks"),
    knowledge_only: bool = Query(True, description="If True, only respond based on found knowledge"),
    use_web: bool = Query(False, description="Whether to include web search results"),
    stream: bool = Query(True, description="Whether to stream the response"),
    model: Optional[str] = Query(None, description="The model to use for generation"),
    provider: Optional[str] = Query(None, description="The provider to use (ollama or openai)")
):
    """Streams chat responses via GET—for EventSource compatibility."""
    try:
        # Check if file exists if filename provided
        if filename:
            file_path = STORAGE_DIR / filename
            if not file_path.exists():
                return JSONResponse(
                    status_code=404,
                    content={"error": f"File not found: {filename}"}
                )

        # Get model and provider from request or use defaults
        model_to_use = model or CONFIG.chat.model
        provider_to_use = provider or CONFIG.chat.provider

        # Define a generator that properly formats the responses for SSE - same as in POST endpoint
        async def format_sse_response():
            try:
                # Send an initial message to establish the connection
                yield "data: {\"status\":\"connected\"}\n\n"
                logging.info("SSE connection established, sent initial message")
                
                chat_generator = chat_service.chat_with_knowledge(
                    query=query, 
                    filename=filename,
                    knowledge_only=knowledge_only,
                    use_web=use_web,
                    model=model_to_use,
                    provider=provider_to_use
                )

                async for data in chat_generator:
                    # Make sure data is properly formatted as valid JSON
                    try:
                        # Verify it's valid JSON by parsing and re-serializing
                        json_obj = json.loads(data)
                        clean_json = json.dumps(json_obj)
                        # Format according to SSE standards: "data: {JSON}\n\n"
                        sse_message = f"data: {clean_json}\n\n"
                        logging.info(f"Sending SSE message: {sse_message.strip()[:100]}...")
                        yield sse_message
                    except json.JSONDecodeError as e:
                        logging.error(f"Invalid JSON in stream: {data[:100]}..., Error: {str(e)}")
                        error_message = f"data: {json.dumps({'error': 'Server error: Invalid response format'})}\n\n"
                        logging.info(f"Sending error message: {error_message.strip()}")
                        yield error_message

                # Add a final event to signal the end of the stream
                end_message = "event: end\ndata: {}\n\n"
                logging.info("Sending end event message")
                yield end_message
                logging.info("Stream ended normally")
            except Exception as e:
                logging.error(f"Stream error: {str(e)}")
                error_message = f"data: {json.dumps({'error': str(e)})}\n\n"
                logging.info(f"Sending exception error message: {error_message.strip()}")
                yield error_message
                yield "event: end\ndata: {}\n\n"

        # Ensure proper headers for SSE
        return StreamingResponse(
            format_sse_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # For Nginx
                "Access-Control-Allow-Origin": "*",  # CORS header
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400"  # 24 hours
            }
        )
    except Exception as e:
        logging.error(f"Unexpected error in GET chat: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )

@app.options("/api/presentation")
async def options_presentation():
    """Handle preflight CORS requests for the presentation endpoint."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
        }
    )

@app.post("/api/presentation", response_model=PresentationResponse)
async def generate_presentation(request: PresentationRequest) -> PresentationResponse:
    """Generate a presentation based on the provided prompt."""
    logger.info(f"Presentation request incoming for '{request.prompt}'—slides gonna be lit! 🚀")
    try:
        # Use await here since create_presentation is a coroutine
        result = await create_presentation(
            prompt=request.prompt,
            filename=request.filename,
            n_slides=request.n_slides,
            model=request.model,
            provider=request.provider
        )
        # Return JSON response with CORS headers
        return JSONResponse(
            content=result,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            }
        )
    except Exception as e:
        logger.error(f"Presentation generation failed: {str(e)} 😅")
        # Return error response with CORS headers
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            }
        )

@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Generate a research report based on the provided query."""
    logger.info(f"Research request incoming for '{request.query}'—knowledge synthesis time! 🔬")
    try:
        result = create_research_report(
            query=request.query,
            use_web=request.use_web
        )
        return result
    except Exception as e:
        logger.error(f"Research generation failed: {str(e)} 😅")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_pdf/{filename}")
async def get_pdf(filename: str):
    """Serve a PDF file from the storage directory."""
    try:
        file_path = STORAGE_DIR / filename
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"PDF not found: {filename} 😢"
            )
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error serving PDF {filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error serving PDF: {str(e)}"
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    tags: str = Form("[]"),  # JSON string of tags
    metadata: str = Form("{}")  # JSON string of metadata
):
    """Upload and process a document with tags and metadata."""
    logger.info(f"Upload request incoming for {file.filename}—doc vibes dropping! 📁")
    
    try:
        # Parse JSON strings
        tag_list = json.loads(tags)
        metadata_dict = json.loads(metadata)
        
        # Save file to storage directory for future reference
        temp_file_path = STORAGE_DIR / file.filename
        temp_file_path.parent.mkdir(exist_ok=True)
        
        # Read the file content once
        content = await file.read()
        
        # Save a copy to storage
        with temp_file_path.open("wb") as buffer:
            buffer.write(content)
        
        # Create a BytesIO object from content to pass to upload_document
        from io import BytesIO
        file_bytes = BytesIO(content)
        
        # Process document with the BytesIO object
        result = upload_document(
            file=file_bytes,
            filename=file.filename,
            tags=tag_list,
            metadata=metadata_dict
        )
        
        return UploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.delete("/delete/{filename}", response_model=DeleteResponse)
async def delete_file(filename: str):
    """Delete a document and its chunks from storage and Milvus."""
    logger.info(f"Delete request incoming for {filename}—cleanup time! 🧹")
    try:
        # Delete from Milvus
        delete_document(filename)
        
        # Delete file if it exists
        file_path = STORAGE_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        return DeleteResponse(
            success=True,
            message=f"Successfully deleted {filename}"
        )
    except Exception as e:
        logger.error(f"Delete failed for {filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {str(e)}"
        )

@app.get("/list", response_model=List[DocInfo])
async def list_documents():
    """List all available documents."""
    try:
        collection = Collection(CONFIG.milvus.collection_name)
        collection.load()
        
        # Query for unique filenames and their metadata
        results = collection.query(
            expr="",
            output_fields=["filename", "metadata", "tags"],
            limit=1000
        )
        
        # Group by filename to get unique docs
        docs: Dict[str, DocInfo] = {}
        for hit in results:
            filename = hit.get('filename', '')
            if filename not in docs:
                try:
                    metadata = json.loads(hit.get('metadata', '{}'))
                    tags = json.loads(hit.get('tags', '[]'))
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                    tags = []
                    
                docs[filename] = DocInfo(
                    doc_id=filename,  # Using filename as ID for now
                    filename=filename,
                    tags=tags,
                    metadata=metadata
                )
        
        return list(docs.values())
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@app.get("/api/config")
async def get_config():
    """Get the current configuration."""
    try:
        return {
            "chat": {
                "model": CONFIG.chat.model,
                "provider": CONFIG.chat.provider,
                "temperature": CONFIG.chat.temperature,
                "chunks_limit": CONFIG.chat.chunks_limit
            },
            "openai": {
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "base_url": CONFIG.openai.base_url
            },
            "ollama": {
                "host": CONFIG.ollama.host,
                "model": CONFIG.ollama.model
            },
            "milvus": {
                "host": CONFIG.milvus.host,
                "port": CONFIG.milvus.port,
                "collection_name": CONFIG.milvus.collection_name
            }
        }
    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get config: {str(e)}"
        )

class ConfigUpdate(BaseModel):
    """Configuration update request."""
    chat: Dict[str, Any]
    openai: Dict[str, Any]
    ollama: Dict[str, Any]
    milvus: Dict[str, Any]

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Update the configuration."""
    try:
        # Update runtime configuration
        CONFIG.chat.model = config.chat.get("model", CONFIG.chat.model)
        CONFIG.chat.provider = config.chat.get("provider", CONFIG.chat.provider)
        CONFIG.chat.temperature = config.chat.get("temperature", CONFIG.chat.temperature)
        CONFIG.chat.chunks_limit = config.chat.get("chunks_limit", CONFIG.chat.chunks_limit)
        
        CONFIG.ollama.host = config.ollama.get("host", CONFIG.ollama.host)
        CONFIG.ollama.model = config.ollama.get("model", CONFIG.ollama.model)
        
        CONFIG.milvus.host = config.milvus.get("host", CONFIG.milvus.host)
        CONFIG.milvus.port = config.milvus.get("port", CONFIG.milvus.port)
        CONFIG.milvus.collection_name = config.milvus.get("collection_name", CONFIG.milvus.collection_name)
        
        # Update OpenAI API key in environment
        openai_api_key = config.openai.get("api_key")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            # Also set in config for persistence
            CONFIG.openai.api_key = openai_api_key
            
        CONFIG.openai.base_url = config.openai.get("base_url", CONFIG.openai.base_url)
        
        # Reinitialize services with new config
        global chat_service
        chat_service = ChatService(
            model=CONFIG.chat.model,
            provider=CONFIG.chat.provider,
            temperature=CONFIG.chat.temperature
        )
        
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update config: {str(e)}"
        )

@app.get("/api/providers/ollama/status")
async def get_ollama_status():
    """Check if Ollama is online and list available models."""
    try:
        ollama_host = CONFIG.ollama.host
        
        # Check if Ollama server is online
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{ollama_host}/api/tags") as response:
                    if response.status == 200:
                        # Get list of models
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        return {
                            "online": True,
                            "models": models
                        }
                    else:
                        return {
                            "online": False,
                            "error": f"Ollama server responded with status code {response.status}",
                            "models": []
                        }
            except Exception as e:
                logger.error(f"Failed to connect to Ollama server: {str(e)}")
                return {
                    "online": False,
                    "error": str(e),
                    "models": []
                }
    except Exception as e:
        logger.error(f"Error checking Ollama status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check Ollama status: {str(e)}"
        )

@app.post("/api/providers/ollama/load")
async def load_ollama_model(model_name: str = Form(...)):
    """Load an Ollama model if it's not already loaded."""
    try:
        ollama_host = CONFIG.ollama.host
        
        # Request to pull/load the model
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{ollama_host}/api/pull", 
                    json={"name": model_name}
                ) as response:
                    if response.status == 200:
                        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
                    else:
                        error_text = await response.text()
                        return {"status": "error", "message": f"Failed to load model: {error_text}"}
            except Exception as e:
                logger.error(f"Failed to load Ollama model: {str(e)}")
                return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Error loading Ollama model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load Ollama model: {str(e)}"
        )

@app.get("/api/providers/openai/models")
async def get_openai_models():
    """List available OpenAI models."""
    try:
        # Check if API key is available
        api_key = os.environ.get("OPENAI_API_KEY", CONFIG.openai.api_key)
        if not api_key:
            return {
                "error": "OpenAI API key not configured",
                "models": []
            }
            
        # Get OpenAI client
        from generation.generate import get_openai_client
        try:
            client = get_openai_client()
            models_response = await client.models.list()
            
            # Filter for chat models that work with our application
            compatible_models = []
            for model in models_response.data:
                model_id = model.id
                if (
                    "gpt" in model_id.lower() or  # GPT models
                    "text-embedding" in model_id.lower() or  # Embedding models
                    "claude" in model_id.lower()  # Claude models if available
                ):
                    compatible_models.append(model_id)
                    
            return {
                "models": compatible_models
            }
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {str(e)}")
            return {
                "error": str(e),
                "models": []
            }
    except Exception as e:
        logger.error(f"Error listing OpenAI models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list OpenAI models: {str(e)}"
        )

# Types locked—code's sharp as fuck! 🔥

if __name__ == "__main__":
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0
    
    # Get host and port from environment variables or use defaults
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    default_port = int(os.getenv("BACKEND_PORT", "8000"))
    
    # Check if the configured port is available, otherwise increment
    port = default_port
    if is_port_in_use(port):
        logger.warning(f"Port {port} is already in use. Trying port {port+1}...")
        port = port + 1
        if is_port_in_use(port):
            logger.warning(f"Port {port} is also in use. Trying port {port+1}...")
            port = port + 1
    
    logger.info(f"Starting server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port) 