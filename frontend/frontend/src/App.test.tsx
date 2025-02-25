import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

describe('App Component', () => {
    test('renders app header', () => {
        render(<App />);
        const headerElement = screen.getByRole('heading', { level: 1, name: /VibeRAG/i });
        expect(headerElement).toBeInTheDocument();
    });

    test('switches between tabs', () => {
        render(<App />);
        
        // Check Chat tab (default)
        expect(screen.getByRole('heading', { level: 2, name: /Chat with Knowledge/i })).toBeInTheDocument();
        
        // Switch to Presentation tab
        fireEvent.click(screen.getByRole('button', { name: /Presentations 🎨/i }));
        expect(screen.getByRole('heading', { level: 2, name: /Presentation Generator/i })).toBeInTheDocument();
        
        // Switch to Research tab
        fireEvent.click(screen.getByRole('button', { name: /Research 🔬/i }));
        expect(screen.getByRole('heading', { level: 2, name: /Research Report Generator/i })).toBeInTheDocument();
    });
});

// Mock fetch for component tests
beforeEach(() => {
    global.fetch = jest.fn(() =>
        Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
                response: 'Test response',
                slides: [{ title: 'Test', content: 'Content' }],
                report: 'Test report'
            })
        })
    ) as jest.Mock;
});
