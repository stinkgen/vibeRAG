# Frontend Architecture Documentation

## Overview

The frontend is built with Next.js and consists of several React components that provide the user interface for the RAG system.

```
frontend/
├── src/
│   ├── components/
│   │   ├── DocumentManager.tsx
│   │   ├── Chat.tsx
│   │   ├── PresentationViewer.tsx
│   │   └── ResearchReport.tsx
│   ├── App.tsx
│   └── index.tsx
```

## Component Architecture

### 1. Document Manager (`DocumentManager.tsx`)

```typescript
interface DocumentManagerProps {
    onUpload: (file: File) => Promise<void>;
    onDelete: (docId: string) => Promise<void>;
}

const DocumentManager: React.FC<DocumentManagerProps> = ({ onUpload, onDelete }) => {
    const [documents, setDocuments] = useState<Document[]>([]);
    const [tags, setTags] = useState<string[]>([]);
    const [metadata, setMetadata] = useState<Record<string, string>>({});

    // Document list rendering
    // Upload handling
    // Delete operations
    // Tag management
    // Metadata editing
};
```

#### Features
- File upload with drag-and-drop
- Document list display
- Tag management
- Metadata editing
- Delete operations

#### State Management
```typescript
interface Document {
    doc_id: string;
    filename: string;
    tags: string[];
    metadata: Record<string, any>;
}

interface UploadState {
    isUploading: boolean;
    progress: number;
    error: string | null;
}
```

### 2. Chat Interface (`Chat.tsx`)

```typescript
interface ChatProps {
    documentId?: string;
    onSendMessage: (message: string) => Promise<void>;
}

const Chat: React.FC<ChatProps> = ({ documentId, onSendMessage }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isTyping, setIsTyping] = useState(false);
    const [context, setContext] = useState<string[]>([]);

    // Message handling
    // Context management
    // Response rendering
    // Source attribution
};
```

#### Features
- Message history
- Context display
- Source attribution
- Loading states
- Error handling

#### Message Types
```typescript
interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: string[];
    timestamp: number;
}
```

### 3. Presentation Viewer (`PresentationViewer.tsx`)

```typescript
interface PresentationViewerProps {
    onGenerate: (prompt: string) => Promise<void>;
    slides?: Slide[];
}

const PresentationViewer: React.FC<PresentationViewerProps> = ({
    onGenerate,
    slides
}) => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Slide navigation
    // Generation handling
    // Slide rendering
    // Source display
};
```

#### Features
- Slide generation
- Navigation controls
- Visual suggestions
- Source attribution
- Export options

#### Slide Structure
```typescript
interface Slide {
    title: string;
    content: string[];
    visual?: string;
    design?: string;
}
```

### 4. Research Report (`ResearchReport.tsx`)

```typescript
interface ResearchReportProps {
    onGenerate: (query: string) => Promise<void>;
    report?: Report;
}

const ResearchReport: React.FC<ResearchReportProps> = ({
    onGenerate,
    report
}) => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [query, setQuery] = useState('');
    const [useWeb, setUseWeb] = useState(true);

    // Report generation
    // Section rendering
    // Source display
    // Export functionality
};
```

#### Features
- Report generation
- Section navigation
- Source attribution
- Export options
- Web search toggle

#### Report Structure
```typescript
interface Report {
    title: string;
    summary: string;
    insights: string[];
    analysis: string;
    sources: string[];
}
```

## API Integration

### 1. Document Operations

```typescript
const uploadDocument = async (
    file: File,
    tags: string[],
    metadata: Record<string, string>
): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('tags', JSON.stringify(tags));
    formData.append('metadata', JSON.stringify(metadata));

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    return response.json();
};
```

### 2. Chat Operations

```typescript
const sendMessage = async (
    message: string,
    documentId?: string
): Promise<ChatResponse> => {
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: message,
            filename: documentId,
            knowledge_only: true
        })
    });

    return response.json();
};
```

### 3. Presentation Operations

```typescript
const generatePresentation = async (
    prompt: string,
    slides: number = 5
): Promise<PresentationResponse> => {
    const response = await fetch('/api/presentation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt,
            n_slides: slides
        })
    });

    return response.json();
};
```

### 4. Research Operations

```typescript
const generateResearch = async (
    query: string,
    useWeb: boolean = true
): Promise<ResearchResponse> => {
    const response = await fetch('/research', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query,
            use_web: useWeb
        })
    });

    return response.json();
};
```

## State Management

### 1. Document State

```typescript
interface DocumentState {
    documents: Document[];
    isLoading: boolean;
    error: string | null;
}

const useDocuments = () => {
    const [state, setState] = useState<DocumentState>({
        documents: [],
        isLoading: false,
        error: null
    });

    // CRUD operations
    // State updates
    // Error handling
};
```

### 2. Chat State

```typescript
interface ChatState {
    messages: Message[];
    isTyping: boolean;
    error: string | null;
}

const useChat = () => {
    const [state, setState] = useState<ChatState>({
        messages: [],
        isTyping: false,
        error: null
    });

    // Message handling
    // State updates
    // Error handling
};
```

## Error Handling

### 1. API Errors

```typescript
interface APIError {
    status: number;
    message: string;
    details?: any;
}

const handleAPIError = (error: APIError) => {
    switch (error.status) {
        case 400:
            // Handle validation errors
            break;
        case 404:
            // Handle not found
            break;
        case 500:
            // Handle server errors
            break;
        default:
            // Handle unknown errors
    }
};
```

### 2. Component Errors

```typescript
const ErrorBoundary: React.FC = ({ children }) => {
    const [hasError, setHasError] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    // Error catching
    // Error display
    // Recovery options
};
```

## Styling

### 1. Component Styles

```typescript
const useStyles = makeStyles((theme) => ({
    container: {
        padding: theme.spacing(2),
        maxWidth: 1200,
        margin: '0 auto'
    },
    paper: {
        padding: theme.spacing(2),
        marginBottom: theme.spacing(2)
    },
    button: {
        margin: theme.spacing(1)
    }
}));
```

### 2. Theme Configuration

```typescript
const theme = createTheme({
    palette: {
        primary: {
            main: '#1976d2'
        },
        secondary: {
            main: '#dc004e'
        }
    },
    typography: {
        fontFamily: 'Roboto, Arial, sans-serif'
    }
});
```

## Performance Optimization

### 1. Memoization

```typescript
const MemoizedComponent = React.memo(({ prop1, prop2 }) => {
    // Component logic
}, (prevProps, nextProps) => {
    // Custom comparison
});
```

### 2. Lazy Loading

```typescript
const LazyComponent = React.lazy(() => import('./Component'));

const App = () => (
    <Suspense fallback={<Loading />}>
        <LazyComponent />
    </Suspense>
);
```

## Testing

### 1. Component Tests

```typescript
describe('DocumentManager', () => {
    it('handles file upload', async () => {
        // Test implementation
    });

    it('displays error messages', () => {
        // Test implementation
    });
});
```

### 2. Integration Tests

```typescript
describe('Chat Integration', () => {
    it('sends messages and receives responses', async () => {
        // Test implementation
    });

    it('handles API errors', async () => {
        // Test implementation
    });
});
```