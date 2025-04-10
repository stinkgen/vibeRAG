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
import starlette.formparsers

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, constr
from dotenv import load_dotenv
from pymilvus import Collection
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

# === In-Memory Chat History Store ===
# WARNING: This is temporary and will be lost on backend restart.
# Structure: { chat_id: List[Dict[str, str]] }
chat_histories_store: Dict[str, List[Dict[str, str]]] = {}
# ====================================

# from frontend.backend.services.chat_service import ChatService, RAGError, SearchError, GenerationError # Commented out - remove later
from src.modules.generation.slides import create_presentation # Fixed import
from src.modules.research.research import create_research_report # Fixed import
from src.modules.vector_store.milvus_ops import connect_milvus, init_collection, delete_document, ensure_connection, update_metadata_in_vector_store # Fixed import
from src.modules.config.config import CONFIG # Fixed import
from src.modules.ingestion.ingest import upload_document # Fixed import
from src.modules.generation.generate import get_openai_client, chat_with_knowledge_ws # Import the refactored chat function and get_openai_client

# Load env vars from root .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env.local")) # Fixed path

# Configure logging with swagger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Increase max file size limit for Starlette's multipart parser
# Set to 100MB (adjust as needed)
# This is a workaround; proper streaming upload is preferred for very large files.
starlette.formparsers.MultiPartParser.max_file_size = 100 * 1024 * 1024
starlette.formparsers.FormParser.max_field_size = 100 * 1024 * 1024 # Also increase field size limit

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections and collections on startup."""
    # --- REMOVING temporary disable block ---
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
    # logger.warning("Milvus initialization temporarily disabled for debugging.") # Ensure this line is commented or removed
    # --- End temporary disable ---
    
    yield  # App is running here
    
    # Cleanup (if needed)
    logger.info("Shutting down—cleanup time! 🧹")

# Initialize FastAPI with some metadata and lifespan
app = FastAPI(
    title="VibeRAG API",
    description="Streaming chat with your docs—4090's cooking! 🔥",
    version="1.0.0",
    lifespan=lifespan,
    # docs_url=None,  # Optional: Disable default /docs
    # redoc_url=None, # Optional: Disable default /redoc
    # openapi_url="/api/v1/openapi.json" # Optional: Set custom OpenAPI URL
)
logger.info("FastAPI app instance created.")

# --- Main API Router ---
# All routes will be defined on this router
api_router = APIRouter()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
STORAGE_DIR = Path("storage/documents") # <-- Uncomment storage dir path
STORAGE_DIR.mkdir(parents=True, exist_ok=True) # <-- Uncomment mkdir

# Initialize services
# chat_service = ChatService( # Commented out - remove later

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

class MetadataUpdateRequest(BaseModel):
    """Request model for updating document metadata."""
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@api_router.get("/health")
def health_check():
    """Health check—system's vibing! 🎯"""
    return {"status": "healthy", "message": "4090's ready to shred! 🔥"}

@api_router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    try:
        # Receive the raw message dictionary first
        message = await websocket.receive()
        # --- DETAILED LOGGING ADDED ---
        logger.info(f"[WS Endpoint] Received initial message. Type: {type(message)}, Content: {message}")
        if isinstance(message, dict):
            logger.info(f"[WS Endpoint] Message keys: {list(message.keys())}")
            if 'type' in message:
                 logger.info(f"[WS Endpoint] Message type field: {message['type']}")
            if 'bytes' in message:
                 logger.info(f"[WS Endpoint] Message bytes field type: {type(message['bytes'])}")
            if 'text' in message:
                 logger.info(f"[WS Endpoint] Message text field type: {type(message['text'])}")
        # --- END DETAILED LOGGING ---
        logger.info(f"Received initial raw message object: {message}")

        raw_initial_data = None
        if isinstance(message, dict) and message.get("type") == "websocket.receive":
            # Check if it has text or bytes
            if "text" in message:
                raw_initial_data = message["text"]
                logger.info(f"Extracted initial text data: {raw_initial_data}")
            elif "bytes" in message:
                # If data comes as bytes, decode assuming utf-8
                raw_initial_data = message["bytes"].decode('utf-8')
                logger.warning(f"Received initial data as bytes, decoded to: {raw_initial_data}")
            else:
                 logger.error("Received message object lacks 'text' or 'bytes' field.")
                 await websocket.close(code=1003, reason="Unsupported initial message format")
                 return
        elif message["type"] == "websocket.disconnect":
             logger.warning(f"Client disconnected immediately after connect: {message.get('code', 'N/A')}")
             return # Exit cleanly
        else:
            logger.error(f"Received unexpected initial message type: {message['type']}")
            await websocket.close(code=1003, reason="Unexpected initial message type")
            return

        if raw_initial_data:
            try:
                # Parse the text as JSON
                params = json.loads(raw_initial_data)
                logger.info(f"Parsed initial parameters: {params}")

                # --- Call the actual handler, passing ONLY expected args ---
                await chat_with_knowledge_ws(
                    websocket=websocket,
                    query=params.get("query"), # Required
                    filename=params.get("filename"), # Optional
                    knowledge_only=params.get("knowledge_only", True), # Optional w/ default
                    use_web=params.get("use_web", False), # Optional w/ default
                    model=params.get("model", CONFIG.chat.model), # Optional w/ default
                    provider=params.get("provider", CONFIG.chat.provider), # Optional w/ default
                    temperature=params.get("temperature", CONFIG.chat.temperature), # Optional w/ default
                    filters=params.get("filters"), # Optional
                    chat_history_id=params.get("chat_history_id") # Pass the ID!
                )
                logger.info("chat_with_knowledge_ws handler finished.")

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse initial message JSON: {json_err}")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1008, reason="Invalid JSON format for initial message")
            except Exception as e:
                logger.exception(f"Error processing initial message or in handler: {e}")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1011, reason="Internal server error during processing")
        else:
             # This case should theoretically not be reached if checks above are correct
             logger.error("raw_initial_data was None after processing receive message.")
             if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.close(code=1011, reason="Internal processing error")


    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (detected by endpoint). ")
    except Exception as e:
        # Catch errors during the initial receive or accept
        logger.exception(f"Error during initial WebSocket setup or receive: {e}")
        # Attempt to close gracefully if possible
        try:
            # Check connection state before attempting final close
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011)
        except RuntimeError as re:
            # Catch the specific error if close is called on an already closed/closing socket
            if "Cannot call \"send\" once a close message has been sent." in str(re):
                logger.warning("Attempted to close WebSocket in final exception handler, but it was already closed.")
            else:
                # Log other RuntimeErrors if they occur during close
                logger.exception(f"RuntimeError during final WebSocket close attempt: {re}")
        except Exception as close_exc:
            # Log other unexpected exceptions during the close attempt
            logger.exception(f"Unexpected exception during final WebSocket close attempt: {close_exc}")
            # Pass, as the primary error 'e' is already logged.
            pass

    # Log endpoint exit
    logger.info("Exiting websocket_chat_endpoint function scope.")

@api_router.options("/presentation")
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

@api_router.post("/presentation", response_model=PresentationResponse)
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

@api_router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Generate a research report based on the provided query."""
    logger.info(f"Research request incoming for '{request.query}'—knowledge synthesis time! 🔬")
    try:
        result = await create_research_report(
            query=request.query,
            use_web=request.use_web
        )
        return result
    except Exception as e:
        logger.error(f"Research generation failed: {str(e)} 😅")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/get_pdf/{filename}")
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

@api_router.post("/upload", response_model=UploadResponse)
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
        
        logger.info(f"File {file.filename} uploaded successfully with {result['num_chunks']} chunks.")
        return UploadResponse(**result)
        
    except Exception as e:
        logger.exception(f"Error processing file {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@api_router.delete("/delete/{filename}", response_model=DeleteResponse)
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
        logger.exception(f"Delete failed for {filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {str(e)}"
        )

@api_router.get("/list", response_model=List[DocInfo])
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
        logger.exception("Error listing documents")
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents"
        )

@api_router.get("/config")
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
        logger.exception("Error fetching configuration")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch configuration"
        )

class ConfigUpdate(BaseModel):
    """Configuration update request."""
    chat: Dict[str, Any]
    openai: Dict[str, Any]
    ollama: Dict[str, Any]
    milvus: Dict[str, Any]

@api_router.post("/config")
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
        
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        logger.exception("Error updating configuration")
        raise HTTPException(
            status_code=500,
            detail="Failed to update configuration"
        )

@api_router.get("/providers/ollama/status")
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

@api_router.post("/providers/ollama/load")
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

@api_router.get("/providers/openai/models")
async def get_openai_models():
    """List available OpenAI models."""
    logger.info("--- ENTERED /providers/openai/models --- ") # Log entry
    try:
        # Check if API key is available
        # Prioritize environment variable, fallback to config object
        api_key_env = os.environ.get("OPENAI_API_KEY")
        api_key_conf = CONFIG.openai.api_key # Assuming config loads it
        api_key = api_key_env or api_key_conf
        
        # Log presence and source (mask the actual key)
        key_status = "Key FOUND" if api_key else "Key NOT FOUND"
        key_source = "(from ENV)" if api_key_env else "(from CONFIG)" if api_key_conf else "(not found)"
        logger.info(f"OpenAI Key Check: Status={key_status}, Source={key_source}, Length={len(api_key) if api_key else 0}")
        
        if not api_key:
            logger.warning("OpenAI API key was not found. Returning error response.")
            return {
                "error": "OpenAI API key not configured",
                "models": []
            }
            
        # Get OpenAI client - Move import here to ensure it's only done if key exists
        from src.modules.generation.generate import get_openai_client 
        try:
            logger.info("Attempting to get OpenAI client...")
            client = get_openai_client() # This might raise ValueError on invalid format
            logger.info(f"OpenAI client obtained. Base URL: {client.base_url}")
            logger.info("Attempting to list OpenAI models via client.models.list()...")
            models_list = await client.models.list()
            logger.info("Successfully listed OpenAI models.")

            # 1. Create the full list of models, sorted by creation date
            all_models_data = sorted(
                [{"id": model.id, "created": model.created} for model in models_list.data],
                key=lambda x: x["created"],
                reverse=True
            )
            logger.info(f"Found {len(all_models_data)} total models from OpenAI.")

            # 2. Filter for likely compatible text generation models
            compatible_models_data = []
            exclusion_terms = [
                "embedding", "image", "audio", "vision", "instruct", 
                "whisper", "tts", "dall-e", "edit", "transcribe",
                "search", "preview" # Add search and preview
            ]
            for model in models_list.data:
                model_id_lower = model.id.lower()
                # Keep if it contains 'gpt' and doesn't contain any exclusion terms
                if "gpt" in model_id_lower and not any(term in model_id_lower for term in exclusion_terms):
                    compatible_models_data.append({"id": model.id, "created": model.created})

            # Sort compatible models by creation date
            compatible_models_data.sort(key=lambda x: x["created"], reverse=True)
            logger.info(f"Found {len(compatible_models_data)} likely compatible text models.")

            # 3. Determine the suggested default model ID
            suggested_default_model_id = CONFIG.openai.default_model # Use configured default as fallback
            if compatible_models_data:
                suggested_default_model_id = compatible_models_data[0]["id"]
                logger.info(f"Suggested default model (newest compatible): {suggested_default_model_id}")
            else:
                logger.warning(f"No compatible text models found based on filtering. Using configured default: {suggested_default_model_id}")

            # 4. Return both lists
            return {
                "all_models": all_models_data,
                "suggested_default": suggested_default_model_id
            }

        except ValueError as ve:
            # Log the specific exception from the OpenAI client init or API call
            logger.error(f"Failed during OpenAI client init or API call: {type(ve).__name__} - {str(ve)}", exc_info=True)
            return {
                "error": f"Failed during OpenAI client init or API call: {str(ve)}",
                "models": []
            }
    except Exception as e:
        # Log errors happening even before checking the key (less likely now)
        logger.error(
            f"Unexpected error in /providers/openai/models handler: {type(e).__name__} - {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing OpenAI models request: {str(e)}"
        )

@api_router.put("/documents/{filename}/metadata", status_code=200)
async def update_document_metadata(filename: str, update_data: MetadataUpdateRequest):
    """Update tags and metadata for a specific document in Milvus."""
    logger.info(f"Metadata update request for {filename}.")
    
    if update_data.tags is None and update_data.metadata is None:
        raise HTTPException(
            status_code=400, 
            detail="No update data provided (tags or metadata must be present)."
        )
    
    try:
        # Call the actual update function
        success = update_metadata_in_vector_store(
            filename=filename, 
            tags=update_data.tags, 
            metadata=update_data.metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update metadata in vector store.")
        
        return {"message": f"Metadata for {filename} updated successfully."}
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.exception(f"Failed to update metadata for {filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred while updating metadata: {str(e)}"
        )

# Types locked—code's sharp as fuck! 🔥

# Include the router with the desired prefix
app.include_router(api_router, prefix="/api/v1")
logger.info("API router included.")

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", 8000))
    logger.info(f"Starting VibeRAG backend server on port {port} 🔥")
    uvicorn.run(app, host="0.0.0.0", port=port) # Removed reload=True for stability 