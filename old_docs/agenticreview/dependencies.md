# Project Dependencies

## Python Backend Dependencies

### Core Dependencies

#### FastAPI Framework
- `fastapi`: Web framework for building APIs
- `uvicorn`: ASGI server implementation
- `pydantic`: Data validation using Python type annotations
- `starlette`: Web framework toolkit (FastAPI dependency)

#### Vector Store
- `pymilvus`: Milvus vector database client
  - Required version: 2.0+
  - Used for vector storage and similarity search

#### Document Processing
- `unstructured`: Document parsing and text extraction
  - Handles PDF, HTML, Markdown, text files
- `PyPDF2`: PDF file processing
  - Used for page counting and metadata extraction

#### Text Processing
- `spacy`: NLP toolkit
  - Required model: `en_core_web_sm`
  - Used for entity extraction and text analysis
- `langdetect`: Language detection library
  - Used in metadata extraction

#### Embedding Generation
- `sentence-transformers`: Text embedding models
  - Default model: `all-MiniLM-L6-v2`
  - Used for generating text embeddings
- `torch`: PyTorch for tensor operations
  - Used by sentence-transformers
  - Optional GPU support

#### LLM Integration
- `openai`: OpenAI API client
  - Used for text generation
- `anthropic`: Anthropic API client
  - Used for Claude model integration
- `ollama`: Local LLM integration
  - Used for running local models

### Utility Libraries

#### File & Path Handling
- `pathlib`: Object-oriented filesystem paths
- `shutil`: High-level file operations

#### Data Processing
- `json`: JSON encoding/decoding
- `yaml`: YAML file parsing
- `numpy`: Numerical operations
  - Used for vector operations

#### HTTP & Networking
- `requests`: HTTP client library
  - Used for web search integration

#### Environment & Configuration
- `python-dotenv`: Environment variable management
- `os`: Operating system interface
- `logging`: Logging functionality

#### Type Hints
- `typing`: Type hint support
  - Used types: Dict, List, Any, Optional, Union, BinaryIO
- `dataclasses`: Data class decorators

#### Unique Identifiers
- `uuid`: UUID generation
  - Used for document IDs

#### Time & Date
- `time`: Time access and conversions

### Testing Dependencies

#### Test Framework
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting

#### Mock & Fixtures
- `unittest.mock`: Mocking functionality
- `pytest-mock`: Pytest mocking utilities

## Frontend Dependencies

### React & Next.js

#### Core Framework
- `react`: React library
- `react-dom`: React DOM manipulation
- `next`: Next.js framework

#### Types
- `@types/react`: React type definitions
- `@types/react-dom`: React DOM type definitions
- `typescript`: TypeScript language

### UI Components

#### Component Libraries
- `@mui/material`: Material-UI components
- `@mui/icons-material`: Material icons
- `@emotion/react`: CSS-in-JS styling
- `@emotion/styled`: Styled components

#### Form Handling
- `react-hook-form`: Form state management
- `yup`: Form validation

#### File Handling
- `react-dropzone`: Drag & drop file uploads

### State Management

#### Application State
- `zustand`: State management
- `immer`: Immutable state updates

#### API Integration
- `axios`: HTTP client
- `swr`: Data fetching and caching

### Development Tools

#### Build Tools
- `webpack`: Module bundler
- `babel`: JavaScript compiler

#### Development Utilities
- `eslint`: Code linting
  - `eslint-config-next`
  - `eslint-plugin-react`
  - `eslint-plugin-react-hooks`
- `prettier`: Code formatting

#### Testing Tools
- `jest`: Testing framework
- `@testing-library/react`: React testing utilities
- `@testing-library/jest-dom`: DOM testing utilities
- `@testing-library/user-event`: User event simulation

## Version Requirements

### Python Environment
```
Python >= 3.8
pip >= 21.0
```

### Node.js Environment
```
Node.js >= 16.0
npm >= 7.0
```

### Database Requirements
```
Milvus >= 2.0
```

## Optional Dependencies

### Web Search Integration
- Google Custom Search API credentials
  - `GOOGLE_SEARCH_API_KEY`
  - `GOOGLE_SEARCH_ENGINE_ID`

### GPU Support
- CUDA toolkit (for PyTorch GPU acceleration)
- GPU-compatible PyTorch installation

### Development Tools
- `black`: Python code formatting
- `isort`: Import sorting
- `mypy`: Static type checking
- `pre-commit`: Git hooks

## Package Installation

### Backend Installation
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### Frontend Installation
```bash
# Core dependencies
npm install

# Development dependencies
npm install --save-dev @types/react @types/react-dom typescript
```

## Configuration Files

### Backend Configuration
- `.env.local`: Environment variables
- `config/config.yaml`: Application configuration
- `pyproject.toml`: Python project metadata
- `setup.py`: Package setup

### Frontend Configuration
- `package.json`: Node.js dependencies
- `tsconfig.json`: TypeScript configuration
- `.eslintrc.json`: ESLint configuration
- `.prettierrc`: Prettier configuration

## Development Tools Configuration

### Editor Configuration
- `.editorconfig`: Editor settings
- `.vscode/`: VS Code settings
  - `settings.json`
  - `launch.json`
  - `extensions.json`

### Git Configuration
- `.gitignore`: Ignored files
- `.pre-commit-config.yaml`: Pre-commit hooks

### Docker Configuration
- `Dockerfile`: Container definition
- `docker-compose.yml`: Service orchestration
  - Milvus
  - Backend
  - Frontend