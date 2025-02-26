#!/bin/bash

# VibeRAG Environment Setup Script
# This script helps set up the proper environment variables for development and production

echo "🌟 VibeRAG Environment Setup 🌟"
echo "-------------------------------"
echo "This script will help you set up the necessary environment variables."
echo

# Check if .env.local already exists
if [ -f .env.local ]; then
    read -p "An .env.local file already exists. Do you want to overwrite it? (y/n): " overwrite
    if [[ $overwrite != "y" && $overwrite != "Y" ]]; then
        echo "Setup canceled. Your existing .env.local file was not modified."
        exit 0
    fi
fi

# Copy example file as a starting point
cp .env.example .env.local

# Prompt for API keys
echo
echo "API Keys Configuration:"
echo "----------------------"
read -p "OpenAI API Key (leave empty to keep default): " openai_key
read -p "Google Search API Key (leave empty to keep default): " google_key
read -p "Google Search Engine ID (leave empty to keep default): " google_engine_id

# Update API keys if provided
if [ ! -z "$openai_key" ]; then
    sed -i "s|OPENAI_API_KEY=\"your_openai_api_key_here\"|OPENAI_API_KEY=\"$openai_key\"|g" .env.local
fi

if [ ! -z "$google_key" ]; then
    sed -i "s|GOOGLE_SEARCH_API_KEY=\"your_google_search_api_key_here\"|GOOGLE_SEARCH_API_KEY=\"$google_key\"|g" .env.local
fi

if [ ! -z "$google_engine_id" ]; then
    sed -i "s|GOOGLE_SEARCH_ENGINE_ID=\"your_google_search_engine_id_here\"|GOOGLE_SEARCH_ENGINE_ID=\"$google_engine_id\"|g" .env.local
fi

# Server configuration
echo
echo "Server Configuration:"
echo "-------------------"
read -p "Backend Host (default: 0.0.0.0): " backend_host
read -p "Backend Port (default: 8000): " backend_port
read -p "Frontend Port (default: 3000): " frontend_port

# Update server configuration if provided
if [ ! -z "$backend_host" ]; then
    sed -i "s|BACKEND_HOST=\"0.0.0.0\"|BACKEND_HOST=\"$backend_host\"|g" .env.local
fi

if [ ! -z "$backend_port" ]; then
    sed -i "s|BACKEND_PORT=8000|BACKEND_PORT=$backend_port|g" .env.local
    
    # Also update the backend URL to match the port
    current_url=$(grep "BACKEND_URL" .env.local | cut -d'"' -f2)
    new_url=$(echo $current_url | sed "s/:8000/:$backend_port/")
    sed -i "s|BACKEND_URL=\"$current_url\"|BACKEND_URL=\"$new_url\"|g" .env.local
fi

if [ ! -z "$frontend_port" ]; then
    sed -i "s|FRONTEND_PORT=3000|FRONTEND_PORT=$frontend_port|g" .env.local
fi

# Create frontend environment file
mkdir -p frontend/frontend
cp .env.local frontend/frontend/.env
cat > frontend/frontend/.env << EOL
# Frontend Environment Variables

# API Configuration
REACT_APP_API_URL=http://localhost:$(grep "BACKEND_PORT" .env.local | cut -d'=' -f2)

# Other Configuration
REACT_APP_NAME=VibeRAG
REACT_APP_PORT=$(grep "FRONTEND_PORT" .env.local | cut -d'=' -f2)
EOL

# Create production environment file for frontend
cat > frontend/frontend/.env.production << EOL
# Production Environment Variables

# API Configuration - Will be replaced during deployment
REACT_APP_API_URL=\${BACKEND_URL}

# Application Configuration
REACT_APP_NAME=VibeRAG
EOL

echo
echo "✅ Environment setup complete!"
echo "📝 Your configuration has been saved to .env.local"
echo "📝 Frontend environment files have been created in frontend/frontend/"
echo
echo "🚀 Next steps:"
echo "  1. Start your services with 'docker-compose up -d'"
echo "  2. Start the backend with 'cd frontend/backend && python app.py'"
echo "  3. Start the frontend with 'cd frontend/frontend && npm start'"
echo

chmod +x setup_env.sh 