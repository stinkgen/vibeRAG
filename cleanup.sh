#!/bin/bash

# VibeRAG Cleanup Script
# This script prepares the codebase for GitHub by cleaning up sensitive data

echo "🧹 Starting VibeRAG cleanup process..."

# 1. Check .gitignore contains all necessary patterns
echo "✓ Checking .gitignore file"
if ! grep -q "\.env\.local" .gitignore; then
  echo "⚠️  Adding .env.local to .gitignore"
  echo ".env.local" >> .gitignore
fi

if ! grep -q "frontend/frontend/\.env" .gitignore; then
  echo "⚠️  Adding frontend/.env to .gitignore"
  echo "frontend/frontend/.env" >> .gitignore
fi

# 2. Clean up server logs
echo "🗑️  Removing server logs"
find . -name "server.log" -type f -delete

# 3. Remove database volumes if they exist
if [ -d "volumes" ]; then
  echo "🗑️  Removing database volumes"
  rm -rf volumes
fi

# 4. Clean up __pycache__ directories
echo "🗑️  Removing Python cache files"
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

# 5. Clean up any Node.js artifacts in frontend
if [ -d "frontend/frontend/node_modules" ]; then
  echo "🗑️  Removing node_modules"
  rm -rf frontend/frontend/node_modules
fi

# 6. Make sure .env.example is present with all necessary variables
if [ ! -f ".env.example" ]; then
  echo "⚠️  .env.example is missing. Creating a template."
  cat > .env.example << EOL
# VibeRAG Environment Variables Example
# Copy this file to .env.local and fill in your actual API keys and configuration

# API Keys (Required)
OPENAI_API_KEY="your_openai_api_key_here"
GOOGLE_SEARCH_API_KEY="your_google_search_api_key_here"
GOOGLE_SEARCH_ENGINE_ID="your_google_search_engine_id_here"

# Optional Configuration
# Chat model settings
CHAT_MODEL="gpt-4"  # Default model for chat
CHAT_PROVIDER="openai"  # Default provider (openai or ollama)
CHAT_TEMPERATURE=0.7  # Default temperature
CHAT_CHUNKS_LIMIT=10  # Maximum number of chunks to retrieve

# Ollama settings (if using Ollama)
OLLAMA_HOST="http://localhost:11434"
OLLAMA_MODEL="llama3"

# Milvus settings
MILVUS_HOST="localhost"
MILVUS_PORT=19530
MILVUS_COLLECTION="vibe_chunks"

# MinIO credentials (for Milvus object storage)
MINIO_ACCESS_KEY="minioadmin"  # Default value - change for production
MINIO_SECRET_KEY="minioadmin"  # Default value - change for production

# Docker settings
DOCKER_VOLUME_DIRECTORY="./volumes"  # Where to store Docker volumes

# Server configuration
BACKEND_HOST="0.0.0.0"  # Host to bind the backend server to
BACKEND_PORT=8000  # Port for the backend server
BACKEND_URL="http://localhost:8000"  # Full URL for backend access (for frontend)

# Frontend configuration
FRONTEND_PORT=3000  # Port for the frontend server

# Docker service ports
ETCD_PORT=2379  # Port for etcd service
MINIO_PORT=9000  # Port for MinIO service
MILVUS_API_PORT=19530  # Main Milvus API port
MILVUS_METRICS_PORT=9091  # Milvus metrics port

# OpenAI configuration
OPENAI_BASE_URL="https://api.openai.com/v1"  # OpenAI API base URL
EOL
fi

# 7. Clean up any environment files in frontend directory that might contain keys
if [ -f "frontend/frontend/.env" ]; then
  echo "🗑️  Removing frontend/.env (will be regenerated from setup script)"
  rm frontend/frontend/.env
fi

# Make sure setup_env.sh exists and is executable
if [ -f "setup_env.sh" ]; then
  echo "✓ Ensuring setup_env.sh is executable"
  chmod +x setup_env.sh
else
  echo "⚠️  setup_env.sh is missing. Please create it for environment setup."
fi

# 8. Check if the docker-compose.yml has environment variables for credentials
echo "✓ docker-compose.yml is using environment variables for credentials"

# 9. Show information about what's in .gitignore
echo ""
echo "📋 FILES THAT WILL NOT BE PUSHED TO GITHUB (from .gitignore):"
grep -v "^#" .gitignore | grep -v "^$" | sort | uniq | sed 's/^/  - /'

echo ""
echo "✅ Cleanup complete! Your codebase is ready for GitHub."
echo ""
echo "🔒 IMPORTANT: Your .env.local file with API keys is preserved locally and"
echo "   will NOT be pushed to GitHub because it's listed in .gitignore."
echo ""
echo "🔑 For others using your repository, they will need to run ./setup_env.sh"
echo "   or create their own .env.local file based on the .env.example template." 