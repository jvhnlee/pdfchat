import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { useChatStream } from './hooks/useChatStream';
import { ScrollArea } from './components/ui/scroll-area';
import { Trash2, Plus, ArrowRight } from 'lucide-react';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';

export default function App() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [pdfName, setPdfName] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState('');
  const { messages, sendMessage, loadingState, clearChat } = useChatStream(sessionId);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loadingState]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (file.type !== 'application/pdf') {
      toast.error('Only PDF files are supported');
      return;
    }

    setIsUploading(true);
    setUploadSuccess(false);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      setUploadSuccess(true);
      setPdfName(file.name);
      toast.success('PDF successfully ingested!');
    } catch (err: any) {
      setPdfName(null);
      toast.error(`Upload error: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }, [sessionId]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || !uploadSuccess || loadingState !== 'idle') return;
    sendMessage(inputValue);
    setInputValue('');
  };

  const handleClear = () => {
    clearChat();
    toast.info('Chat history cleared');
  };

  const handleDeletePdf = () => {
    setUploadSuccess(false);
    setPdfName(null);
    clearChat();
    toast.info('Document removed');
  };

  return (
    <div className="flex h-screen w-screen bg-white text-zinc-950 overflow-hidden">
      <Toaster position="top-right" />

      {/* Left Column - Branding (Vertical Stacked Title) */}
      <div className="w-[30%] flex flex-col justify-center items-end pr-8 bg-white select-none shrink-0">
        <h1 className="text-8xl font-bold tracking-tight text-green-600 leading-none font-serif text-right">
          pdf
        </h1>
        <h1 className="text-8xl font-bold tracking-tight text-green-600 leading-none font-serif mt-2 text-right">
          chat
        </h1>
      </div>

      {/* Right Column - Chat pane & input line */}
      <div className="flex-1 flex flex-col h-full bg-white pl-8 pr-16 py-12 justify-between relative">
        
        {/* Top bar with status and clear chat option */}
        <div className="flex justify-between items-center mb-6 shrink-0">
          <span className="text-xs text-zinc-400 uppercase tracking-wider">
            {uploadSuccess ? 'Document Active' : 'No Document'}
          </span>
          {messages.length > 0 && (
            <button
              onClick={handleClear}
              className="text-xs text-zinc-400 hover:text-red-500 font-medium transition-colors"
            >
              Clear Chat
            </button>
          )}
        </div>

        {/* Middle Area: Chat messages list or empty state */}
        <div className="flex-1 flex flex-col justify-end overflow-hidden mb-8">
          {messages.length === 0 && loadingState === 'idle' ? (
            <div className="h-full flex flex-col justify-center">
              <h2 className="text-3xl font-bold text-zinc-800 tracking-tight mb-2">How can I help you today?</h2>
              <p className="text-zinc-500 max-w-md text-sm leading-relaxed">
                Ingest a PDF by clicking the plus (+) explorer icon on the input line, and start asking questions about its contents.
              </p>
            </div>
          ) : (
            <ScrollArea className="flex-1 pr-4" ref={scrollRef}>
              <div className="space-y-6 pb-4">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div 
                      className={`max-w-[80%] rounded-2xl p-4 text-sm whitespace-pre-wrap leading-relaxed ${
                        msg.role === 'user' 
                          ? 'bg-zinc-950 text-white rounded-tr-sm' 
                          : 'bg-zinc-50 text-zinc-900 border border-zinc-200/65 rounded-tl-sm'
                      }`}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))}

                {loadingState !== 'idle' && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] bg-zinc-50 border border-zinc-200/65 rounded-2xl rounded-tl-sm p-4 animate-pulse">
                      <div className="text-green-600 font-medium text-xs">
                        {loadingState}
                      </div>
                      <div className="space-y-2 mt-2">
                        <div className="h-1.5 bg-zinc-200 rounded-full w-48 animate-pulse"></div>
                        <div className="h-1.5 bg-zinc-200 rounded-full w-32 animate-pulse"></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          )}
        </div>

        {/* Bottom Area: Input line with plus icon and submit arrow */}
        <div className="flex flex-col shrink-0">
          
          {/* File Name Tag when uploaded */}
          {uploadSuccess && pdfName && (
            <div className="flex items-center gap-2 bg-green-50 border border-green-200/50 text-green-700 text-xs px-3 py-1.5 rounded-full w-fit mb-3 animate-fade-in">
              <span className="font-medium truncate max-w-[200px]" title={pdfName}>{pdfName}</span>
              <span className="text-green-400 font-normal">| Ingested successfully</span>
              <button 
                onClick={handleDeletePdf} 
                type="button"
                className="hover:text-red-500 ml-1 p-0.5 rounded-full hover:bg-green-100 transition-colors"
                title="Remove document"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          )}

          {/* Underlined minimalist input row */}
          <form onSubmit={handleSubmit} className="relative">
            <div className="border-b border-zinc-900 pb-2 flex items-center gap-3">
              
              {/* File Explorer Trigger Button (+) */}
              <div {...getRootProps()} className="cursor-pointer">
                <input {...getInputProps()} />
                <button
                  type="button"
                  className={`h-8 w-8 rounded-full border border-zinc-950 flex items-center justify-center transition-all ${
                    isUploading
                      ? 'bg-zinc-100 border-zinc-300 cursor-not-allowed text-zinc-400'
                      : uploadSuccess
                        ? 'bg-green-600 border-green-600 text-white hover:bg-green-700'
                        : 'bg-transparent text-zinc-950 hover:bg-zinc-100'
                  }`}
                  disabled={isUploading}
                  title="Upload PDF (File Explorer)"
                >
                  {isUploading ? (
                    <span className="text-xs font-bold animate-pulse">...</span>
                  ) : (
                    <Plus className="h-4 w-4" />
                  )}
                </button>
              </div>

              {/* Input text field */}
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder={uploadSuccess ? "chat with your pdf" : "Upload a document to start chatting"}
                disabled={!uploadSuccess || loadingState !== 'idle'}
                className="flex-1 bg-transparent border-none outline-none focus:ring-0 text-zinc-950 text-base placeholder-zinc-400"
              />

              {/* Submit arrow button (→) */}
              <button
                type="submit"
                disabled={!inputValue.trim() || !uploadSuccess || loadingState !== 'idle'}
                className="h-8 w-8 flex items-center justify-center text-zinc-950 hover:text-green-600 disabled:opacity-30 disabled:hover:text-zinc-950 transition-colors"
                title="Send Message"
              >
                <ArrowRight className="h-5 w-5" />
              </button>
            </div>
          </form>

          {/* Helper caption/indicator */}
          {!uploadSuccess && !isUploading && (
            <div className="text-xs text-zinc-400 mt-2">
              {isDragActive ? 'Drop PDF here' : 'Drop PDF or click to browse'}
            </div>
          )}

          {isUploading && (
            <div className="text-xs text-green-600 mt-2 animate-pulse">
              Ingesting PDF...
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
