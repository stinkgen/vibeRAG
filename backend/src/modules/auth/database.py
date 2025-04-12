import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

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

class ChatMessageBase(BaseModel):
    sender: str
    content: str
    timestamp: datetime

class ChatMessageResponse(ChatMessageBase):
    id: int
    session_id: int

    class Config:
        from_attributes = True
        
class ChatSessionBase(BaseModel):
    title: str

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