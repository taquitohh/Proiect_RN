import { useRef, useEffect } from 'react';
import { MessageBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator.tsx';
import { Bot, Sparkles } from 'lucide-react';

export interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  interpretation?: string;
  params?: Record<string, unknown>;
  code?: string;
}

interface ChatContainerProps {
  messages: Message[];
  isLoading: boolean;
  onCopyCode: (code: string) => void;
  onDownloadCode: (code: string, filename: string) => void;
}

export function ChatContainer({ messages, isLoading, onCopyCode, onDownloadCode }: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-3">
      {messages.length === 0 ? (
        <div className="h-full flex flex-col items-center justify-center text-center">
          <div className="w-14 h-14 bg-gradient-to-br from-pink-600 to-orange-500 rounded-xl flex items-center justify-center mb-4">
            <Bot className="w-7 h-7 text-white" />
          </div>
          <h2 className="text-xl font-bold text-white mb-1">
            Blender AI
          </h2>
          <p className="text-sm text-slate-400 max-w-sm mb-6">
            Descrie ce obiect 3D vrei să creezi.
          </p>
          
          {/* Exemple */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 max-w-2xl">
            {[
              'Creează un cub de 2 metri',
              'Sferă roșie cu raza 0.5',
              'Material metalic',
            ].map((example, i) => (
              <div
                key={i}
                className="flex items-center gap-2 p-3 bg-slate-800/50 border border-slate-700 rounded-lg text-left hover:bg-slate-800 transition-colors cursor-pointer"
              >
                <Sparkles className="w-4 h-4 text-pink-500 flex-shrink-0" />
                <span className="text-xs text-slate-300">{example}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onCopyCode={onCopyCode}
              onDownloadCode={onDownloadCode}
            />
          ))}
          
          {isLoading && <TypingIndicator />}
          
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
