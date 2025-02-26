# VibeRAG

A Retrieval-Augmented Generation (RAG) system with a cyberpunk UI for knowledge management, document analysis, and AI-powered chat.

![VibeRAG](https://img.shields.io/badge/VibeRAG-Cyberpunk%20RAG-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

VibeRAG combines the power of vector databases, LLMs, and semantic search to provide a unified interface for managing documents and interacting with your knowledge base. The system includes:

- Document ingestion and chunking pipeline
- Vector storage with Milvus
- Semantic, keyword, and hybrid search capabilities
- AI chat with sources/citations
- Knowledge filtering by documents, collections, and tags
- Web search integration
- Streaming responses for a responsive UI
- Chat history management
- Presentation generation

## Screenshots

(Add screenshots of your UI here)

## Installation

### Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- Docker and Docker Compose
- OpenAI API key (for GPT models)
- Google Search API key (optional, for web search)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vibeRAG.git
   cd vibeRAG
   ```

2. Run the setup script to configure your environment:
   ```bash
   ./setup_env.sh
   ```
   
   This interactive script will:
   - Create an `.env.local` file from the template
   - Prompt you for API keys and configuration values
   - Set up the frontend environment files
   
   If you prefer manual setup, you can copy the template and edit it yourself:
   ```bash
   cp .env.example .env.local
   ```

3. Start Milvus with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the backend:
   ```bash
   cd frontend/backend
   uvicorn app:app --reload
   ```

6. In another terminal, install frontend dependencies:
   ```bash
   cd frontend/frontend
   npm install
   npm start
   ```

## Usage

### Ingesting Documents

1. Use the UI to upload documents or the command line tool:
   ```bash
   python -m ingest sample.pdf
   ```

### Searching Knowledge Base

1. Navigate to the search page
2. Enter your query
3. Use filters to narrow down results

### AI Chat

1. Go to the chat interface
2. Select "Knowledge Only" to restrict the assistant to your documents
3. Use filters to select specific sources
4. Enable "Web Search" to complement with internet results

### Generating Presentations

1. Enter a topic in the presentation generator
2. Review and export the generated slides

## Configuration

The project now uses environment variables for all configuration. Key files:

- `.env.local`: Main configuration file (created by setup script)
- `.env.example`: Template with all available options
- `frontend/.env`: Frontend environment variables (auto-generated)
- `config/config.yaml`: Additional system settings

Available configuration options include:
- API keys for OpenAI and Google Search
- Server host/port settings
- Database connection parameters
- Model selection and parameters

## Project Structure

```
vibeRAG/
├── config/               # Configuration files
├── embedding/            # Embedding models and utilities
├── frontend/             # UI components and backend API
│   ├── backend/          # FastAPI server
│   └── frontend/         # React frontend
├── generation/           # Text generation utilities
├── ingestion/            # Document processing pipeline
├── research/             # Research report generation
├── retrieval/            # Search functionality
├── storage/              # Document storage
├── tests/                # Test suite
├── vector_store/         # Vector database operations
├── .env.example          # Environment variables template
├── docker-compose.yml    # Docker setup for Milvus
└── requirements.txt      # Python dependencies
```

## Development

### Setting Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Running Tests

```bash
pytest tests/
```

## License

MIT

## Acknowledgments

- This project uses [Milvus](https://milvus.io/) for vector storage
- UI inspired by cyberpunk aesthetics
- Built with FastAPI and React