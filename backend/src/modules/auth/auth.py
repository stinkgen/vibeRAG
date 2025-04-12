import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.modules.config.config import CONFIG
from src.modules.auth.database import User, get_db, SessionLocal, UserCreate, UserUpdate

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login") # Points to the login endpoint

# --- Password Hashing ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# --- JWT Token Handling ---

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None
    is_admin: Optional[bool] = None

def create_access_token(data: Dict[str, Any]) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=CONFIG.auth.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    # Ensure 'is_admin' claim is correctly populated if present in the input data
    # The input data comes from the login function which should include the user's role/admin status
    if "is_admin" not in to_encode:
        logger.warning(f"'is_admin' claim missing from data provided to create_access_token: {data}. Defaulting to False in token.")
        to_encode["is_admin"] = False # Default if missing, but login should provide it.
        
    encoded_jwt = jwt.encode(to_encode, CONFIG.auth.secret_key, algorithm=CONFIG.auth.algorithm)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[TokenData]:
    """Decodes and validates a JWT access token."""
    try:
        payload = jwt.decode(token, CONFIG.auth.secret_key, algorithms=[CONFIG.auth.algorithm])
        # Minimal validation: check expiration is handled by jwt.decode
        token_data = TokenData(
            username=payload.get("sub"), 
            user_id=payload.get("id"), 
            is_admin=payload.get("is_admin", False)
        )
        return token_data
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.PyJWTError as e:
        logger.warning(f"Token decoding error: {e}")
        return None

# --- User Retrieval ---

def get_user(db: Session, username: str) -> Optional[User]:
    """Retrieves a user by username."""
    return db.query(User).filter(User.username == username).first()

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Retrieves a user by ID."""
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Retrieves a user by username (alias for get_user)."""
    return get_user(db, username)

def get_users(db: Session, skip: int = 0, limit: int = 100) -> list[User]:
    """Retrieves a list of users with pagination."""
    return db.query(User).offset(skip).limit(limit).all()

# --- Authentication Function (Placeholder/Example) ---
def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticates a user by username and password."""
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

# --- Authentication Dependency ---

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency to get the current authenticated user from a token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = decode_access_token(token)
    if token_data is None or token_data.user_id is None:
        raise credentials_exception
        
    user = get_user_by_id(db, user_id=token_data.user_id)
    if user is None:
        raise credentials_exception
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
        
    return user

async def get_current_active_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to get the current user, ensuring they are an admin."""
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user

# --- User Creation ---

def create_user(db: Session, user: UserCreate) -> User:
    """Creates a new user in the database."""
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        hashed_password=hashed_password,
        is_admin=user.is_admin, # Use is_admin from UserCreate
        is_active=user.is_active, # Use is_active from UserCreate
        role=user.role # Use role from UserCreate
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User '{user.username}' created successfully.")
    return db_user

# --- Initial Admin User Creation ---

async def create_initial_admin_user(db: Session):
    """Creates the initial admin user if no users exist in the database."""
    existing_user = db.query(User).first()
    if existing_user is None:
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin") # Default password
        hashed_password = get_password_hash(admin_password)
        admin_user = User(
            username=admin_username,
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True,
            role="admin" # Explicitly set role for admin
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        logger.info(f"Initial admin user '{admin_username}' created.")
        if admin_password == "admin":
             logger.warning("Default admin password 'admin' is insecure. Please change it.")
    else:
        logger.info("Users already exist in the database. Skipping initial admin creation.")

# --- Update User --- 
def update_user(db: Session, user_id: int, user_update: UserUpdate) -> Optional[User]:
    """Updates a user's details in the database."""
    db_user = get_user_by_id(db, user_id=user_id)
    if not db_user:
        return None

    update_data = user_update.model_dump(exclude_unset=True)
    
    # Handle password hashing if password is being updated
    if "password" in update_data and update_data["password"]:
        hashed_password = get_password_hash(update_data["password"])
        update_data["password"] = hashed_password
    elif "password" in update_data: # Remove password field if it's empty or None
        del update_data["password"]

    for key, value in update_data.items():
        setattr(db_user, key, value)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User {db_user.username} (ID: {user_id}) updated.")
    return db_user

# --- Delete User --- 
def delete_user(db: Session, user_id: int) -> Optional[User]:
    """Deletes a user from the database."""
    db_user = get_user_by_id(db, user_id=user_id)
    if not db_user:
        return None

    username = db_user.username # Store username for logging before deletion
    db.delete(db_user)
    db.commit()
    logger.info(f"User {username} (ID: {user_id}) deleted.")
    # TODO (Phase 3/4): Add cleanup logic here (delete Milvus collection, chat history) if needed
    return db_user # Return the deleted user object (or just True/None) 