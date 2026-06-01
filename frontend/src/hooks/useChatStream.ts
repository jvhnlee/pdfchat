import { useState, useRef } from 'react';

export type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export type LoadingState = 'idle' | 'Analyzing...' | 'Generating...';

export function useChatStream(sessionId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  const isStreamingRef = useRef(false);

  const sendMessage = async (question: string) => {
    if (isStreamingRef.current) return;
    isStreamingRef.current = true;
    // Add user message instantly
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    
    // Cycle 1 of loading states
    setLoadingState('Analyzing...');
    
    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId, question }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to fetch response');
      }

      // Cycle 2 of loading states: Request succeeded, waiting for first token
      setLoadingState('Generating...');
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      
      if (!reader) return;

      let isFirstChunk = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        
        if (isFirstChunk) {
            // First token arrived, stop loaders
            setLoadingState('idle'); 
            setMessages((prev) => [...prev, { role: 'assistant', content: chunk }]);
            isFirstChunk = false;
        } else {
            // Stream token into last message
            setMessages((prev) => {
                const newMessages = [...prev];
                const lastIdx = newMessages.length - 1;
                newMessages[lastIdx].content += chunk;
                return newMessages;
            });
        }
      }
    } catch (error: any) {
      console.error(error);
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${error.message}` }]);
      setLoadingState('idle');
    } finally {
      isStreamingRef.current = false;
    }
  };

  const clearChat = async () => {
    try {
      await fetch('http://127.0.0.1:8000/api/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, question: '' }),
      });
      setMessages([]);
    } catch (e) {
      console.error("Failed to clear chat");
    }
  };

  return { messages, sendMessage, loadingState, clearChat };
}
