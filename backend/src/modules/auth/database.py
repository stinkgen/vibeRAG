import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from src.modules.config.config import CONFIG

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vibe_auth.db")

# Use check_same_thread=False only for SQLite, required for FastAPI use
engine_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=engine_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# --- SQLAlchemy User Model ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    role: str = Column(String, default="user") # Added role field
    
    # Relationship to ChatSession
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    agents = relationship("Agent", back_populates="owner")

# --- SQLAlchemy ChatSession Model ---
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")

# --- SQLAlchemy ChatMessage Model ---
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    sender = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    session = relationship("ChatSession", back_populates="messages")


# --- Pydantic Models for User Data ---

# Base model for user properties
class UserBase(BaseModel):
    username: str
    role: str = "user"
    is_active: bool = True
    is_admin: bool = False # Keep for potential backward compat or specific checks

# Model for creating a new user (expects password)
class UserCreate(UserBase):
    password: str

# --- Add UserUpdate Model ---
# Model for updating user data (role, active status, optional password reset)
class UserUpdate(BaseModel):
    username: Optional[str] = None # Allow updating username if needed?
    role: Optional[str] = None 
    is_active: Optional[bool] = None
    password: Optional[str] = None # For password resets

# Model for reading user data (excludes password)
class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True # Updated for Pydantic v2


# --- Pydantic Models for Chat History ---

# --- Add Agent Models Here ---

# --- SQLAlchemy Agent Model ---
class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    owner_user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, index=True)
    persona = Column(String)
    goals = Column(String) # Could be JSON/Text if goals are complex
    base_prompt = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    owner = relationship("User", back_populates="agents")
    capabilities = relationship("AgentCapability", back_populates="agent", cascade="all, delete-orphan")
    tasks = relationship("AgentTask", back_populates="agent") # Add relationship to tasks
    # logs = relationship("AgentLog", back_populates="agent", cascade="all, delete-orphan") # Removed - Logs might not need direct Agent backref if task_id is enough

# --- SQLAlchemy AgentCapability Model ---
# Example: If we need explicit mapping of agents to tools
class AgentCapability(Base):
     __tablename__ = "agent_capabilities"
     id = Column(Integer, primary_key=True, index=True)
     agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
     tool_name = Column(String, nullable=False, index=True) # Name of the tool allowed
     
     agent = relationship("Agent", back_populates="capabilities")

# --- SQLAlchemy AgentState Model ---
# Example: For storing simple runtime state if needed in DB
# class AgentState(Base):
#     __tablename__ = "agent_states"
#     id = Column(Integer, primary_key=True, index=True)
#     agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, unique=True, index=True)
#     status = Column(String, default="idle") # e.g., idle, running, error
#     current_task_id = Column(String, nullable=True)
#     last_heartbeat = Column(DateTime, nullable=True)
    
#     agent = relationship("Agent", back_populates="state")


# --- Pydantic Models for Agent Data ---

class AgentBase(BaseModel):
    name: str
    persona: Optional[str] = None
    goals: Optional[str] = None
    base_prompt: Optional[str] = None
    is_active: bool = True
    # Add new optional fields
    # llm_provider: Optional[str] = None 
    # llm_model: Optional[str] = None

class AgentCreate(AgentBase):
    owner_user_id: int # Required on creation

class AgentUpdate(AgentBase):
    name: Optional[str] = None # Allow partial updates
    is_active: Optional[bool] = None
    # Allow updating LLM config
    # llm_provider: Optional[str] = None
    # llm_model: Optional[str] = None

class AgentResponse(AgentBase):
    id: int
    owner_user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# --- SQLAlchemy AgentLog Model ---
class AgentLog(Base):
    __tablename__ = "agent_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    # task_id = Column(String, index=True, nullable=True) # Link logs within a specific task run
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String, index=True, default="INFO") # e.g., INFO, WARN, ERROR, DEBUG, PLAN, ACTION, TOOL_CALL, COMM
    message = Column(Text, nullable=False) # Human-readable log message
    details = Column(Text, nullable=True) # Optional structured details (e.g., JSON string of tool params/results)
    
    # agent = relationship("Agent") # Optional relationship if needed


# --- Pydantic Models for Agent Log Data ---
class AgentLogBase(BaseModel):
    agent_id: int
    # task_id: Optional[str] = None
    level: str
    message: str
    details: Optional[str] = None

class AgentLogResponse(AgentLogBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


# --- SQLAlchemy AgentMemory Model ---
class AgentMemory(Base):
    __tablename__ = "agent_memories"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True) # Cascade delete memories if agent is deleted
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    memory_type = Column(String, index=True, nullable=False) # e.g., 'episodic', 'summary', 'reflection', 'scratchpad_chunk'
    content = Column(Text, nullable=False)
    importance = Column(Float, default=0.5) # Simple importance score (0-1)
    # Optional: Store related memory IDs as JSON string or use association table later
    related_memory_ids = Column(Text, nullable=True) 

    agent = relationship("Agent") # Relationship back to the agent

# --- Pydantic Models for Agent Memory ---
class AgentMemoryBase(BaseModel):
    agent_id: int
    memory_type: str
    content: str
    importance: Optional[float] = 0.5
    related_memory_ids: Optional[List[int]] = None # Parse from/to JSON string in DB

class AgentMemoryCreate(AgentMemoryBase):
    pass # Data provided is sufficient

class AgentMemoryResponse(AgentMemoryBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


# Base model for chat session properties
class ChatSessionBase(BaseModel):
    title: Optional[str] = None

# --- Pydantic Models for Chat History ---

class ChatMessageBase(BaseModel):
    sender: str
    content: str
    timestamp: datetime

class ChatMessageResponse(ChatMessageBase):
    id: int
    session_id: int

    class Config:
        from_attributes = True
        
class ChatSessionCreate(ChatSessionBase):
    pass # Title is enough for creation, user_id comes from context

class ChatSessionResponse(ChatSessionBase):
    id: int
    user_id: int
    created_at: datetime
    last_updated_at: Optional[datetime] # Make optional to handle potential None from DB
    messages: List[ChatMessageResponse] = [] # Include messages when returning a specific session

    class Config:
        from_attributes = True

class ChatSessionListResponse(ChatSessionBase):
    id: int
    user_id: int
    created_at: datetime
    last_updated_at: Optional[datetime] = None # Make optional

    class Config:
        from_attributes = True

# --- SQLAlchemy AgentTask Model ---
class AgentTask(Base):
    __tablename__ = "agent_tasks"

    id = Column(Integer, primary_key=True, index=True) # Use auto-incrementing integer PK
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True) # User who initiated task
    celery_task_id = Column(String, unique=True, index=True, nullable=True) # Store Celery task UUID
    goal = Column(Text, nullable=False) # Original goal/prompt
    input_data = Column(JSON, nullable=True) # Optional input data
    status = Column(String, default="pending", index=True, nullable=False) # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    result_data = Column(JSON, nullable=True) # Store AgentOutput JSON on success
    error_message = Column(Text, nullable=True) # Store error message on failure
    # Add relationship back to Agent/User if needed
    # agent = relationship("Agent")
    # user = relationship("User")


# --- DB Session Dependency ---

def get_db():
    """FastAPI dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- DB Initialization ---
def create_db_and_tables():
    """Creates the database tables."""
    Base.metadata.create_all(bind=engine)

# Optionally add initial admin user creation logic here or in lifespan 