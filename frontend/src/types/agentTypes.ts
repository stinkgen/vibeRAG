export interface ScratchpadEntry {
    role: 'user' | 'assistant' | 'system' | 'tool';
    content: string;
    timestamp?: string; // Optional timestamp
}

// Add other shared agent-related types here later if needed 