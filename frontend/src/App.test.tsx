import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import App from './App';

// Mock matchMedia for Lucide icons or other components
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // Deprecated
    removeListener: vi.fn(), // Deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock fetch globally
// eslint-disable-next-line @typescript-eslint/no-explicit-any
globalThis.fetch = vi.fn() as any;

describe('pdfchat Frontend App Tests', () => {
  it('renders the branding and main chat area initially', () => {
    render(<App />);
    expect(screen.getByText('pdf')).toBeInTheDocument();
    expect(screen.getByText('chat')).toBeInTheDocument();
    expect(screen.getByText('How can I help you today?')).toBeInTheDocument();
    expect(screen.getByText(/Drop PDF or click to browse/i)).toBeInTheDocument();

    // Chat input should be disabled initially
    const input = screen.getByPlaceholderText('Upload a document to start chatting');
    expect(input).toBeDisabled();
  });

  it('handles PDF dropzone auto-ingestion flow', async () => {
    const user = userEvent.setup();
    let resolveFetch: (value: any) => void = () => {};
    const fetchPromise = new Promise((resolve) => {
      resolveFetch = resolve;
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis.fetch as any).mockReturnValueOnce(fetchPromise);

    render(<App />);
    
    // Create a mock PDF file
    const file = new File(['dummy content'], 'mock_document.pdf', { type: 'application/pdf' });
    const inputEl = document.querySelector('input[type="file"]')!;

    // Trigger upload (do not await fully yet, as it awaits the fetch promise)
    const uploadPromise = user.upload(inputEl as HTMLElement, file);

    // Verify loading state
    await waitFor(() => {
        expect(screen.getByText('Ingesting PDF...')).toBeInTheDocument();
    });

    // Now resolve the fetch mock
    resolveFetch({
      ok: true,
      json: async () => ({ status: 'success' }),
    });

    await uploadPromise;

    // Verify success state
    await waitFor(() => {
        expect(screen.getByText('mock_document.pdf')).toBeInTheDocument();
        expect(screen.getByText(/Ingested successfully/i)).toBeInTheDocument();
        const chatInput = screen.getByPlaceholderText('chat with your pdf');
        expect(chatInput).toBeEnabled();
    });
  });

  it('rejects non-PDF files', async () => {
    const user = userEvent.setup();
    render(<App />);

    const file = new File(['dummy content'], 'mock.txt', { type: 'text/plain' });
    const inputEl = document.querySelector('input[type="file"]')!;

    await user.upload(inputEl as HTMLElement, file);

    // It should not enable the chat since upload failed/rejected
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Upload a document to start chatting')).toBeDisabled();
    });
  });
});
