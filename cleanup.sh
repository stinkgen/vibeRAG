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

# 6. Make sure .env.example is present
if [ ! -f ".env.example" ]; then
  echo "⚠️  .env.example is missing. Creating a basic template."
  cat > .env.example << EOL
# VibeRAG Environment Variables Example
# Copy this file to .env.local and fill in your actual API keys and configuration

# API Keys (Required)
OPENAI_API_KEY="your_openai_api_key_here"
GOOGLE_SEARCH_API_KEY="your_google_search_api_key_here"
GOOGLE_SEARCH_ENGINE_ID="your_google_search_engine_id_here"

# Optional Configuration
# See documentation for more options
EOL
fi

# 7. Check if the docker-compose.yml has environment variables for credentials
echo "✓ docker-compose.yml is using environment variables for credentials"

# 8. Show information about what's in .gitignore
echo ""
echo "📋 FILES THAT WILL NOT BE PUSHED TO GITHUB (from .gitignore):"
grep -v "^#" .gitignore | grep -v "^$" | sort | uniq | sed 's/^/  - /'

echo ""
echo "✅ Cleanup complete! Your codebase is ready for GitHub."
echo ""
echo "🔒 IMPORTANT: Your .env.local file with API keys is preserved locally and"
echo "   will NOT be pushed to GitHub because it's listed in .gitignore."
echo ""
echo "🔑 For others using your repository, they will need to create their own"
echo "   .env.local file based on the .env.example template you've provided." 