# Project Structure

```
/workspace/
├── config/
│   ├── __init__.py
│   ├── config.py
│   ├── config.yaml
│   └── crews/
│       ├── presentation_crew.yaml
│       └── research_crew.yaml
├── docs/
│   └── agenticreview/
│       ├── README.md
│       ├── api_documentation.md
│       ├── backend_architecture.md
│       ├── complete_documentation.md
│       ├── cursor_rules.md
│       ├── dead_code_report.md
│       ├── dependencies.md
│       ├── development_critique.md
│       ├── frontend_architecture.md
│       ├── functionality_review.md
│       ├── quality_assessment.md
│       ├── rag_pipeline.md
│       ├── structure.md
│       └── user_guide.md
├── embedding/
│   ├── __init__.py
│   └── embed.py
├── frontend/
│   ├── backend/
│   │   ├── app.py
│   │   ├── docs/
│   │   │   └── whitepaper.pdf
│   │   ├── server.log
│   │   └── storage/
│   │       └── documents/
│   │           ├── Cman_038_3_Thorpe.pdf
│   │           ├── The anarchist cookbook - William Powell (1).pdf
│   │           └── whitepaper.pdf
│   └── frontend/
│       ├── README.md
│       ├── package.json
│       ├── package-lock.json
│       ├── public/
│       │   ├── favicon.ico
│       │   ├── index.html
│       │   ├── logo192.png
│       │   ├── logo512.png
│       │   ├── manifest.json
│       │   ├── robots.txt
│       │   └── sounds/
│       ├── src/
│       │   ├── App.css
│       │   ├── App.module.css
│       │   ├── App.test.tsx
│       │   ├── App.tsx
│       │   ├── components/
│       │   │   ├── Chat.module.css
│       │   │   ├── Chat.tsx
│       │   │   ├── DocumentManager.module.css
│       │   │   ├── DocumentManager.tsx
│       │   │   ├── PresentationViewer.module.css
│       │   │   ├── PresentationViewer.tsx
│       │   │   ├── ResearchReport.module.css
│       │   │   └── ResearchReport.tsx
│       │   ├── index.css
│       │   ├── index.tsx
│       │   ├── logo.svg
│       │   ├── react-app-env.d.ts
│       │   ├── reportWebVitals.ts
│       │   ├── setupTests.ts
│       │   └── styles/
│       │       └── global.css
│       └── tsconfig.json
├── generation/
│   ├── __init__.py
│   ├── generate.py
│   └── slides.py
├── ingestion/
│   ├── __init__.py
│   └── ingest.py
├── research/
│   ├── __init__.py
│   └── research.py
├── retrieval/
│   ├── __init__.py
│   └── search.py
├── storage/
│   └── documents/
├── tests/
│   ├── test_app.py
│   ├── test_crew.py
│   ├── test_embed.py
│   ├── test_generate.py
│   ├── test_ingest.py
│   ├── test_milvus_ops.py
│   ├── test_retrieval.py
│   └── test_search.py
├── vector_store/
│   ├── __init__.py
│   └── milvus_ops.py
├── __init__.py
├── requirements.txt
├── setup.py
├── test_pipeline.py
├── test_script.py
└── whitepaper.pdf

## Key Directories

### Backend Core
- `/config`: Configuration files and crew definitions
- `/embedding`: Text embedding functionality
- `/generation`: Text generation and presentation creation
- `/ingestion`: Document processing and chunking
- `/research`: Research report generation
- `/retrieval`: Search and retrieval operations
- `/vector_store`: Milvus vector database operations

### Frontend
- `/frontend/backend`: FastAPI server
- `/frontend/frontend`: React/Next.js application
  - `/src/components`: React components
  - `/src/styles`: CSS styles

### Testing and Documentation
- `/tests`: Test suite
- `/docs/agenticreview`: Comprehensive documentation

### Storage
- `/storage/documents`: Document storage
- `/frontend/backend/storage/documents`: Frontend document storage

## Notable Files

### Configuration
- `config/config.yaml`: Main configuration
- `config/crews/*.yaml`: CrewAI configurations

### Backend Core
- `embedding/embed.py`: Text embedding
- `generation/generate.py`: Text generation
- `generation/slides.py`: Presentation generation
- `ingestion/ingest.py`: Document processing
- `research/research.py`: Research report generation
- `retrieval/search.py`: Search functionality
- `vector_store/milvus_ops.py`: Vector database operations

### Frontend
- `frontend/backend/app.py`: FastAPI server
- `frontend/frontend/src/App.tsx`: Main React component
- `frontend/frontend/src/components/*.tsx`: React components

### Testing
- `tests/test_*.py`: Test files for each module

### Documentation
- `docs/agenticreview/*.md`: Comprehensive documentation files