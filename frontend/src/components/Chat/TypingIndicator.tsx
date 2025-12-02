import { Bot } from 'lucide-react';

export function TypingIndicator() {
  return (
    <div className="flex gap-4">
      {/* Avatar */}
      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-600 to-orange-500 flex items-center justify-center flex-shrink-0">
        <Bot className="w-5 h-5 text-white" />
      </div>

      {/* Indicator */}
      <div className="bg-slate-800 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}
