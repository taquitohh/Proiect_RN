import { useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  inputValue: string;
  onInputChange: (value: string) => void;
}

const quickSuggestions = [
  'cub',
  'sferă',
  'cilindru',
  'con',
  'torus',
  'material',
  'export',
  'șterge',
];

export function ChatInput({ onSend, isLoading, inputValue, onInputChange }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [inputValue]);

  const handleSubmit = () => {
    if (inputValue.trim() && !isLoading) {
      onSend(inputValue.trim());
      onInputChange('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-slate-700 bg-slate-900 p-3 pb-4">
      <div className="max-w-3xl mx-auto">
        {/* Sugestii rapide */}
        <div className="flex flex-wrap gap-1.5 mb-2">
          {quickSuggestions.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => onInputChange(inputValue + (inputValue ? ' ' : '') + suggestion)}
              className="px-2 py-0.5 text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full border border-slate-700 transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>

        {/* Input area */}
        <div className="relative flex items-end gap-2 bg-slate-800 border border-slate-700 rounded-xl p-1.5">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Descrie ce obiect 3D vrei să creezi..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-white placeholder-slate-500 resize-none px-2 py-1.5 focus:outline-none max-h-24"
            disabled={isLoading}
          />
          
          <button
            onClick={handleSubmit}
            disabled={!inputValue.trim() || isLoading}
            className={`flex items-center justify-center w-8 h-8 rounded-lg transition-all ${
              inputValue.trim() && !isLoading
                ? 'bg-gradient-to-r from-pink-600 to-orange-500 hover:from-pink-500 hover:to-orange-400 text-white'
                : 'bg-slate-700 text-slate-500 cursor-not-allowed'
            }`}
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Hint */}
        <p className="text-[10px] text-slate-500 text-center mt-1.5">
          <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">Enter</kbd> trimite · <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">Shift+Enter</kbd> linie nouă
        </p>
      </div>
    </div>
  );
}
