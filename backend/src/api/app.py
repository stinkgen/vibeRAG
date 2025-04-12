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
from sqlalchemy.orm import Session
from datetime import timedelta
from enum import Enum

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, constr
from dotenv import load_dotenv
from pymilvus import Collection, utility
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import status

# === In-Memory Chat History Store ===
# WARNING: This is temporary and will be lost on backend restart.
# Structure: { chat_id: List[Dict[str, str]] }
chat_histories_store: Dict[str, List[Dict[str, str]]] = {}
# ====================================

# from frontend.backend.services.chat_service import ChatService, RAGError, SearchError, GenerationError # Commented out - remove later
from src.modules.generation.slides import create_presentation # Fixed import
from src.modules.research.research import create_research_report # Fixed import
from src.modules.vector_store.milvus_ops import connect_milvus, init_collection, delete_document, ensure_connection, update_metadata_in_vector_store, get_user_collection_name, get_admin_collection_name, get_global_collection_name, list_all_collections, drop_collection # Fixed import
from src.modules.config.config import CONFIG, ChatConfig, OllamaConfig, OpenAIConfig, MilvusConfig, EmbeddingConfig, IngestionConfig, PresentationConfig, ResearchConfig, WebSearchConfig, SearchConfig, AuthConfig
from src.modules.ingestion.ingest import upload_document # Fixed import
from src.modules.generation.generate import get_openai_client, websocket_endpoint as chat_websocket_handler

# --- Import Auth and DB setup ---
from src.modules.auth.database import (
    get_db, engine, User, UserCreate, UserResponse, SessionLocal, 
    create_db_and_tables, UserUpdate,
    # Add Chat History Pydantic Models
    ChatSessionResponse, ChatSessionListResponse
)
from src.modules.auth.auth import (
    get_current_user, # Use this for active user check
    get_current_active_admin_user, # Use this for admin check
    create_access_token, 
    verify_password, 
    get_password_hash, 
    get_user, 
    create_initial_admin_user, 
    create_user, 
    get_users, 
    get_user_by_username, 
    authenticate_user, 
    TokenData,
    update_user,
    delete_user
)
from sqlalchemy.orm import Session

# --- Import Chat History CRUD --- 
from src.modules.chat.history import (
    create_chat_session,
    get_chat_session,
    get_user_chat_sessions,
    delete_chat_session,
    update_chat_session_title
)

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
    """Initialize connections, collections, and database on startup."""
    logger.info("App starting up...")
    # --- Database Setup ---
    try:
        logger.info("Initializing database...")
        create_db_and_tables() # Create tables if they don't exist (Now imported)
        logger.info("Database tables checked/created.")
        
        # Create initial admin user if none exists
        db = SessionLocal() # Use SessionLocal directly here for lifespan setup
        try:
            await create_initial_admin_user(db)
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to initialize database or create admin user: {str(e)}", exc_info=True)
        # Decide if the app should fail to start here. For now, we log and continue.

    # --- Milvus Setup ---
    # --- REMOVING temporary disable block ---
    try:
        # Ensure Milvus connection
        ensure_connection()
        logger.info("Connected to Milvus—waiting for it to be ready...")
        
        # Give Milvus a sec to get its shit together
        await asyncio.sleep(2)
        
        # Initialize CORE collections (admin and global)
        admin_collection_name = get_admin_collection_name()
        global_collection_name = get_global_collection_name()
        if admin_collection_name:
            init_collection(admin_collection_name)
            logger.info(f"Initialized Admin Milvus collection: {admin_collection_name} 🚀")
        if global_collection_name:
             init_collection(global_collection_name)
             logger.info(f"Initialized Global Milvus collection: {global_collection_name} 🚀")
        # User collections are created dynamically on first upload/search
        
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

# --- Main API Router (Now without global dependency) ---
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
    use_knowledge: bool = True # Add flag to control KB usage
    model: Optional[str] = None # Add optional model
    provider: Optional[str] = None # Add optional provider

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
    doc_id: int
    filename: str
    tags: List[str]
    metadata: Dict[str, Any]

class MetadataUpdateRequest(BaseModel):
    """Request model for updating document metadata."""
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

# --- Authentication Router (Remains unprotected) ---
auth_router = APIRouter()

# --- Pydantic Model for Auth Token Response --- (Defined here)
class Token(BaseModel):
    access_token: str
    token_type: str

@auth_router.post("/auth/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Ensure user is active before creating token
    if not user.is_active:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
         
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role, "id": user.id} # Pass data payload only
    )
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current logged-in user info."""
    return current_user

# --- User Management Routes (Admin Only) ---
@api_router.get("/users", response_model=List[UserResponse], dependencies=[Depends(get_current_active_admin_user)])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve users (Admin only)."""
    users = get_users(db, skip=skip, limit=limit)
    return users

@api_router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_active_admin_user)])
def create_user_route(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user (Admin only) and their Milvus collection."""
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    created_user = create_user(db=db, user=user)
    
    # Create Milvus collection for the new user
    try:
        collection_name = get_user_collection_name(created_user.id)
        init_collection(collection_name) # Creates if not exists
        logger.info(f"Ensured Milvus collection '{collection_name}' exists for user {created_user.username}")
    except Exception as e:
        logger.error(f"Failed to create Milvus collection for {created_user.username}: {e}", exc_info=True)
        # Decide if user creation should fail if collection creation fails?
        # For now, log error but let user creation succeed.
        # Consider raising HTTPException(500, ...) if collection is critical.
        
    return created_user

# --- Endpoint to Update a User (Admin Only) ---
@api_router.put("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(get_current_active_admin_user)])
def update_user_route(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db)):
    """Update user details (Admin only)."""
    updated_user = update_user(db=db, user_id=user_id, user_update=user_update)
    if updated_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user

# --- Endpoint to Delete a User (Admin Only) ---
@api_router.delete("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(get_current_active_admin_user)]) 
def delete_user_route(user_id: int, db: Session = Depends(get_db)):
    """Delete a user (Admin only) and their Milvus collection."""
    deleted_user = delete_user(db=db, user_id=user_id)
    if deleted_user is None:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Drop the user's Milvus collection
    try:
        collection_name = get_user_collection_name(deleted_user.id)
        success = drop_collection(collection_name)
        if success:
            logger.info(f"Successfully dropped Milvus collection '{collection_name}' for deleted user {deleted_user.username}.")
        else:
             logger.warning(f"Could not drop Milvus collection '{collection_name}' for deleted user {deleted_user.username} (may not exist).")
    except Exception as e:
        logger.error(f"Error dropping Milvus collection for deleted user {deleted_user.username}: {e}", exc_info=True)
        # Log error but don't fail the user deletion itself
        
    # TODO: Delete chat history for the user here (Phase 4 cleanup)
    logger.info(f"User ID {user_id} deletion completed.")
    return deleted_user

# --- Chat History Session Endpoints (Protected) ---

@api_router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_user)])
def create_new_chat_session(
    # Optional: Allow setting title on creation
    # session_create: ChatSessionCreate, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Creates a new, empty chat session for the logged-in user."""
    try:
        # Create session with default title for now
        session = create_chat_session(db=db, user_id=current_user.id)
        # Return the session without messages initially
        return ChatSessionResponse.model_validate(session) 
    except Exception as e:
        logger.exception(f"Failed to create chat session for user {current_user.username}")
        raise HTTPException(status_code=500, detail="Failed to create chat session")

@api_router.get("/sessions", response_model=List[ChatSessionListResponse], dependencies=[Depends(get_current_user)])
def list_user_chat_sessions(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Lists all chat sessions for the logged-in user."""
    sessions = get_user_chat_sessions(db=db, user_id=current_user.id, skip=skip, limit=limit)
    return sessions # Response model handles conversion

@api_router.get("/sessions/{session_id}", response_model=ChatSessionResponse, dependencies=[Depends(get_current_user)])
def get_specific_chat_session(
    session_id: int, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Gets a specific chat session including its messages, if it belongs to the user."""
    session = get_chat_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found or access denied")
    # Ensure messages are loaded (SQLAlchemy lazy loading might handle this, but explicit check is fine)
    # session.messages will be populated based on the relationship
    return session # Response model handles conversion including messages

@api_router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(get_current_user)])
def delete_specific_chat_session(
    session_id: int, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Deletes a specific chat session, if it belongs to the user."""
    success = delete_chat_session(db=db, session_id=session_id, user_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found or access denied")
    return Response(status_code=status.HTTP_204_NO_CONTENT) # Return No Content on success

# Optional: Endpoint to update session title
class SessionTitleUpdate(BaseModel):
    title: str = Field(..., min_length=1)
    
@api_router.put("/sessions/{session_id}/title", response_model=ChatSessionListResponse, dependencies=[Depends(get_current_user)])
def update_session_title_route(
    session_id: int, 
    title_update: SessionTitleUpdate,
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Updates the title of a specific chat session."""
    updated_session = update_chat_session_title(db=db, session_id=session_id, user_id=current_user.id, new_title=title_update.title)
    if not updated_session:
         raise HTTPException(status_code=404, detail="Chat session not found or access denied")
    return updated_session

# --- Apply Dependencies Selectively --- 

# Unprotected endpoint
@api_router.get("/health")
def health_check():
    """Health check—system's vibing! 🎯"""
    return {"status": "healthy", "message": "4090's ready to shred! 🔥"}

# Protected endpoints
@api_router.get("/config", dependencies=[Depends(get_current_user)])
async def get_config():
    """Retrieve the current backend configuration."""
    return CONFIG

# --- Pydantic Models for Config Update ---

class AuthConfigUpdate(BaseModel):
    secret_key: Optional[str] = None
    algorithm: Optional[str] = None
    access_token_expire_minutes: Optional[int] = None

class ChatConfigUpdate(BaseModel):
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    chunks_limit: Optional[int] = None

class ResearchConfigUpdate(BaseModel):
    chunks_limit: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None

class PresentationConfigUpdate(BaseModel):
    chunks_limit: Optional[int] = None
    max_slides: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None

class WebSearchConfigUpdate(BaseModel):
    limit: Optional[int] = None

class EmbeddingConfigUpdate(BaseModel):
    model_name: Optional[str] = None
    batch_size: Optional[int] = None
    # device and embedding_dim are usually auto-detected or derived, less likely to be updated via API
    # embedding_dim: Optional[int] = None 

class IngestionConfigUpdate(BaseModel):
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    batch_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class MilvusConfigUpdate(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    # dim/embedding_dim changes require re-indexing, usually not updated on the fly
    # dim: Optional[int] = None 
    # embedding_dim: Optional[int] = None
    collection_name: Optional[str] = None
    index_type: Optional[str] = None
    metric_type: Optional[str] = None
    default_batch_size: Optional[int] = None
    tags_field: Optional[str] = None
    tags_max_capacity: Optional[int] = None
    text_field: Optional[str] = None
    embedding_field: Optional[str] = None
    metadata_field: Optional[str] = None
    doc_id_field: Optional[str] = None
    filename_field: Optional[str] = None
    chunk_id_field: Optional[str] = None
    # field_params, index_params, search_params are complex dicts, less suitable for simple API update
    # field_params: Optional[Dict[str, Dict[str, Any]]] = None
    # index_params: Optional[Dict[str, Any]] = None
    # search_params: Optional[Dict[str, Any]] = None
    consistency_level: Optional[str] = None

class OllamaConfigUpdate(BaseModel):
    host: Optional[str] = None
    model: Optional[str] = None
    chat_endpoint: Optional[str] = None
    generate_endpoint: Optional[str] = None
    temperature: Optional[float] = None

class OpenAIConfigUpdate(BaseModel):
    api_key: Optional[str] = Field(None, description="OpenAI API Key") # Allow updating API key
    base_url: Optional[str] = None
    default_model: Optional[str] = None

class SearchConfigUpdate(BaseModel):
    default_limit: Optional[int] = None
    min_score: Optional[float] = None

# The main model for the POST /config endpoint
class ConfigUpdate(BaseModel):
    auth: Optional[AuthConfigUpdate] = None
    chat: Optional[ChatConfigUpdate] = None
    research: Optional[ResearchConfigUpdate] = None
    presentation: Optional[PresentationConfigUpdate] = None
    web_search: Optional[WebSearchConfigUpdate] = None
    embedding: Optional[EmbeddingConfigUpdate] = None
    ingestion: Optional[IngestionConfigUpdate] = None
    milvus: Optional[MilvusConfigUpdate] = None
    ollama: Optional[OllamaConfigUpdate] = None
    openai: Optional[OpenAIConfigUpdate] = None
    search: Optional[SearchConfigUpdate] = None

@api_router.post("/config", dependencies=[Depends(get_current_active_admin_user)])
async def update_config(config_update: ConfigUpdate, db: Session = Depends(get_db)):
    """Update application configuration - USE WITH CAUTION."""
    # NOTE: This directly modifies the global CONFIG object. Changes are immediate
    # but won't persist across restarts unless .env.local is also updated.
    # Consider a more robust approach (e.g., updating .env.local or using a dedicated config service).
    logger.warning("--- Received request to update configuration --- ")
    updated_sections = []
    try:
        update_data = config_update.dict(exclude_unset=True)
        logger.info(f"Raw update request data: {update_data}")

        for section, updates in update_data.items():
            if hasattr(CONFIG, section):
                config_section = getattr(CONFIG, section)
                logger.info(f"Updating config section: '{section}'")
                for key, value in updates.items():
                    if hasattr(config_section, key):
                        old_value = getattr(config_section, key)
                        setattr(config_section, key, value)
                        logger.info(f"  - Updated '{section}.{key}': from '{old_value}' to '{value}'")
                    else:
                         logger.warning(f"  - Key '{key}' not found in config section '{section}'. Skipping.")
                updated_sections.append(section)
            else:
                 logger.warning(f"Config section '{section}' not found. Skipping.")
        
        # Persist changes? Reload dependent components? (e.g., re-init clients?)
        # This is complex. For now, it only updates the in-memory CONFIG object.
        logger.info(f"Configuration update complete. Updated sections: {updated_sections}")
        return {"message": f"Configuration updated successfully for sections: {updated_sections}", "updated_config": CONFIG.dict()}
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

# Add target_collection parameter, default to 'user'
class UploadTarget(str, Enum):
    USER = "user"
    GLOBAL = "global"
    
@api_router.post("/upload", dependencies=[Depends(get_current_user)])
async def upload_file(
    file: UploadFile = File(...),
    tags: str = Form("[]"),  # JSON string of tags
    metadata: str = Form("{}"),  # JSON string of metadata
    target_collection: UploadTarget = Form(UploadTarget.USER), # New param: user or global
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # Inject current user
):
    """Upload and process a document, targeting user's or global collection."""
    logger.info(f"Upload request for '{file.filename}' by user '{current_user.username}'. Target: {target_collection.value}")
    
    # Admins can target global, users cannot
    if current_user.role != 'admin' and target_collection == UploadTarget.GLOBAL:
         logger.warning(f"User '{current_user.username}' is not admin, cannot target global collection. Overriding to user.")
         target_collection = UploadTarget.USER # Force to user
         
    try:
        tag_list = json.loads(tags)
        metadata_dict = json.loads(metadata)
        
        # Save file to temporary storage (consider user-specific subdirs if needed)
        temp_file_path = STORAGE_DIR / file.filename 
        content = await file.read()
        with temp_file_path.open("wb") as buffer:
            buffer.write(content)
        from io import BytesIO
        file_bytes = BytesIO(content)
        
        # Call updated upload_document with user and target type
        result = upload_document(
            file=file_bytes,
            filename=file.filename,
            user=current_user, # Pass user object
            target_collection_type=target_collection.value, # Pass 'user' or 'global'
            tags=tag_list,
            metadata=metadata_dict
        )
        
        # Clean up temp file
        temp_file_path.unlink(missing_ok=True)
        
        logger.info(f"File '{file.filename}' processed successfully.")
        return UploadResponse(**result)
        
    except Exception as e:
        # Clean up temp file on error too
        if 'temp_file_path' in locals() and temp_file_path.exists():
             temp_file_path.unlink(missing_ok=True)
        logger.exception(f"Error processing upload for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@api_router.get("/list", dependencies=[Depends(get_current_user)])
async def list_documents(user: User = Depends(get_current_user)):
    """List documents accessible to the user (personal + global)."""
    collections_to_query = []
    if user.role == 'admin':
        collections_to_query.append(get_admin_collection_name())
    else:
        collections_to_query.append(get_user_collection_name(user.id))
    collections_to_query.append(get_global_collection_name())
    
    all_docs = {}
    try:
        for collection_name in collections_to_query:
            # Check if collection exists before querying
            if not utility.has_collection(collection_name):
                logger.info(f"Collection '{collection_name}' not found, skipping for list.")
                continue
                
            collection = init_collection(collection_name)
            # Query distinct filenames and one instance of their metadata/tags
            # Using query with limit=1 and group_by might be complex for getting metadata.
            # A simpler approach is to query all, then process in Python.
            results = collection.query(
                expr="", # No filter, get all
                output_fields=[CONFIG.milvus.filename_field, CONFIG.milvus.metadata_field, CONFIG.milvus.tags_field, CONFIG.milvus.chunk_id_field],
                limit=16384 # High limit to get all docs likely
            )
            
            for hit in results:
                 filename = hit.get(CONFIG.milvus.filename_field)
                 if filename and filename not in all_docs:
                     try:
                         metadata = json.loads(hit.get(CONFIG.milvus.metadata_field, '{}'))
                     except:
                         metadata = {}
                     all_docs[filename] = DocInfo(
                         doc_id=hit.get(CONFIG.milvus.chunk_id_field), # Using chunk_id as a proxy doc_id here
                         filename=filename,
                         tags=hit.get(CONFIG.milvus.tags_field, []),
                         metadata=metadata,
                         # Add owner/scope info
                         scope=collection_name # Indicate which collection it came from
                     )
                     
        logger.info(f"Listed {len(all_docs)} unique documents for user '{user.username}'.")
        return list(all_docs.values())

    except Exception as e:
        logger.exception(f"Error listing documents for user '{user.username}': {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")

@api_router.delete("/delete/{filename}", dependencies=[Depends(get_current_user)])
async def delete_file_endpoint(filename: str, user: User = Depends(get_current_user)):
    """Deletes a document from the user's collection or global (if admin)."""
    deleted = False
    message = ""
    
    # Determine target collection
    collection_name_to_delete_from: Optional[str] = None
    
    # 1. Try deleting from user's/admin's personal collection
    if user.role == 'admin':
        user_coll = get_admin_collection_name()
    else:
        user_coll = get_user_collection_name(user.id)
        
    if utility.has_collection(user_coll):
         deleted_from_user = delete_document(collection_name=user_coll, filename=filename)
         if deleted_from_user:
             collection_name_to_delete_from = user_coll
             deleted = True
             message = f"Deleted '{filename}' from personal collection '{user_coll}'."
             logger.info(message)

    # 2. If not deleted from personal AND user is admin, try deleting from global
    if not deleted and user.role == 'admin':
         global_coll = get_global_collection_name()
         if utility.has_collection(global_coll):
             deleted_from_global = delete_document(collection_name=global_coll, filename=filename)
             if deleted_from_global:
                 collection_name_to_delete_from = global_coll
                 deleted = True
                 message = f"Deleted '{filename}' from global collection '{global_coll}'."
                 logger.info(message)
                 
    # 3. If still not deleted, it wasn't found
    if not deleted:
        message = f"Document '{filename}' not found in accessible collections."
        logger.warning(message)
        # Return success=False or raise 404? Raising 404 seems more appropriate.
        raise HTTPException(status_code=404, detail=message)
        
    # 4. Delete the source file from storage (consider user-specific dirs later)
    try:
        file_path = STORAGE_DIR / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted source file from storage: {file_path}")
        else:
            logger.warning(f"Source file not found in storage for deleted doc: {file_path}")
    except Exception as e:
        logger.error(f"Failed to delete source file {filename} from storage: {e}")
        # Don't fail the whole request, but log the error

    return DeleteResponse(success=True, message=message)

@api_router.put("/documents/{filename}/metadata", status_code=200, dependencies=[Depends(get_current_user)])
async def update_document_metadata(filename: str, update_data: MetadataUpdateRequest, user: User = Depends(get_current_user)):
    """Updates metadata for a document in user's or global collection."""
    updated = False
    target_collection = None
    
    # Determine target collection (similar logic to delete)
    # 1. Try user's/admin's personal collection
    if user.role == 'admin':
        user_coll = get_admin_collection_name()
    else:
        user_coll = get_user_collection_name(user.id)
        
    if utility.has_collection(user_coll):
        # Check if doc exists there before attempting update
        coll = init_collection(user_coll)
        res = coll.query(expr=f'{CONFIG.milvus.filename_field} == "{filename}" ', limit=1)
        if res:
             updated = update_metadata_in_vector_store(
                 collection_name=user_coll,
                 filename=filename, 
                 tags=update_data.tags, 
                 metadata=update_data.metadata
             )
             if updated:
                 target_collection = user_coll

    # 2. If not updated in personal AND user is admin, try global
    if not updated and user.role == 'admin':
         global_coll = get_global_collection_name()
         if utility.has_collection(global_coll):
             coll = init_collection(global_coll)
             res = coll.query(expr=f'{CONFIG.milvus.filename_field} == "{filename}" ', limit=1)
             if res:
                 updated = update_metadata_in_vector_store(
                     collection_name=global_coll,
                     filename=filename, 
                     tags=update_data.tags, 
                     metadata=update_data.metadata
                 )
                 if updated:
                      target_collection = global_coll
                      
    if not updated:
         raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in accessible collections or update failed.")

    return {"message": f"Metadata updated successfully for '{filename}' in collection '{target_collection}'."}

@api_router.get("/get_pdf/{filename}", dependencies=[Depends(get_current_user)])
async def get_document(filename: str, user: User = Depends(get_current_user)):
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

@api_router.post("/presentation", dependencies=[Depends(get_current_user)])
async def generate_presentation(request: PresentationRequest, user: User = Depends(get_current_user)):
    """Generate a presentation based on the provided prompt."""
    logger.info(f"Presentation request incoming for '{request.prompt}' by user '{user.username}'—slides gonna be lit! 🚀") # Log user
    try:
        # Use await here since create_presentation is a coroutine
        result = await create_presentation(
            prompt=request.prompt,
            user=user, # Pass the user object
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

@api_router.post("/research", dependencies=[Depends(get_current_user)])
async def generate_research(request: ResearchRequest, user: User = Depends(get_current_user)):
    """Generate a research report based on the provided query."""
    logger.info(f"Research request incoming for '{request.query}' by user '{user.username}'—knowledge synthesis time! 🔬 Use Web: {request.use_web}, Use Knowledge: {request.use_knowledge}") # Log user
    # ---> Log received model/provider <--- 
    logger.info(f"Received model='{request.model}', provider='{request.provider}' in request body.")
    try:
        result = await create_research_report(
            query=request.query,
            user=user, # Pass the user object
            use_web=request.use_web,
            use_knowledge=request.use_knowledge,
            model=request.model, # Pass model
            provider=request.provider # Pass provider
        )
        return result
    except Exception as e:
        logger.error(f"Research generation failed: {str(e)} 😅")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/providers/ollama/status", dependencies=[Depends(get_current_user)])
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

@api_router.post("/providers/ollama/load", dependencies=[Depends(get_current_active_admin_user)])
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

@api_router.get("/providers/openai/models", dependencies=[Depends(get_current_active_admin_user)])
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
                "search" # Removed "preview"
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

            # 4. Return ONLY the compatible models and the suggested default
            return {
                "compatible_models": compatible_models_data, # Return filtered list
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

@api_router.websocket("/ws/chat")
async def websocket_endpoint_route(websocket: WebSocket):
    """Route WebSocket connections to the handler. AUTHENTICATION MUST BE HANDLED WITHIN THE HANDLER."""
    await chat_websocket_handler(websocket) # Call the imported handler

# Types locked—code's sharp as fuck! 🔥

# Include the routers
app.include_router(auth_router, prefix="/api/v1", tags=["Auth"]) # Add tag
app.include_router(api_router, prefix="/api/v1", tags=["API"]) # Add tag
logger.info("API routers included.")

# --- Endpoint to Create Initial Collections (Admin Only, Optional) ---
# Could be called manually or during app startup for admin/global
@api_router.post("/admin/init-collections", status_code=201, dependencies=[Depends(get_current_active_admin_user)])
def init_core_collections():
    """Ensures the global and admin collections exist."""
    try:
        admin_coll = get_admin_collection_name()
        global_coll = get_global_collection_name()
        init_collection(admin_coll)
        init_collection(global_coll)
        logger.info(f"Ensured core collections exist: {admin_coll}, {global_coll}")
        return {"message": f"Core collections '{admin_coll}' and '{global_coll}' initialized."}
    except Exception as e:
        logger.error(f"Failed to initialize core collections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize core Milvus collections.")

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", 8000))
    logger.info(f"Starting VibeRAG backend server on port {port} 🔥")
    uvicorn.run(app, host="0.0.0.0", port=port) # Removed reload=True for stability 