"""CRUD operations for chat history (sessions and messages)."""

import logging
from sqlalchemy.orm import Session
from typing import List, Optional

from src.modules.auth.database import ChatSession, ChatMessage, User, ChatSessionCreate

logger = logging.getLogger(__name__)

# --- Chat Session CRUD ---

def create_chat_session(db: Session, user_id: int, title: Optional[str] = "New Chat") -> ChatSession:
    """Creates a new chat session for a user."""
    session = ChatSession(user_id=user_id, title=title or "New Chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(f"Created new chat session (ID: {session.id}) for user ID: {user_id}")
    return session

def get_chat_session(db: Session, session_id: int, user_id: int) -> Optional[ChatSession]:
    """Gets a specific chat session by ID, ensuring it belongs to the user."""
    return db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()

def get_user_chat_sessions(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[ChatSession]:
    """Gets all chat sessions for a specific user."""
    return db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.last_updated_at.desc()).offset(skip).limit(limit).all()

def update_chat_session_title(db: Session, session_id: int, user_id: int, new_title: str) -> Optional[ChatSession]:
    """Updates the title of a chat session."""
    session = get_chat_session(db, session_id, user_id)
    if session:
        session.title = new_title
        db.commit()
        db.refresh(session)
        logger.info(f"Updated title for session ID: {session_id}")
        return session
    return None

def delete_chat_session(db: Session, session_id: int, user_id: int) -> bool:
    """Deletes a chat session and its messages."""
    session = get_chat_session(db, session_id, user_id)
    if session:
        db.delete(session) # Cascade delete should handle messages
        db.commit()
        logger.info(f"Deleted chat session ID: {session_id}")
        return True
    return False

# --- Chat Message CRUD ---

def add_chat_message(db: Session, session_id: int, sender: str, content: str) -> ChatMessage:
    """Adds a new message to a chat session."""
    # Note: We don't check user_id here directly, assuming session_id validation happened before
    message = ChatMessage(session_id=session_id, sender=sender, content=content)
    db.add(message)
    # Also update the session's last_updated_at timestamp
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.last_updated_at = message.timestamp # Use message's timestamp
    db.commit()
    db.refresh(message)
    # logger.debug(f"Added message to session {session_id}: {sender} - {content[:30]}...") # Maybe too verbose
    return message

def get_session_messages(db: Session, session_id: int, limit: Optional[int] = None) -> List[ChatMessage]:
    """Gets messages for a specific session, optionally limited."""
    query = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc())
    if limit is not None:
        # If limiting, get the *latest* N messages
        query = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.desc()).limit(limit)
        # Need to reverse the order back for chronological display/processing
        return query.all()[::-1] 
    return query.all() 