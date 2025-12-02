import { User, Bot, Copy, Download, Check } from 'lucide-react';
import { useState } from 'react';
import { CodeBlock } from './CodeBlock';
import type { Message } from './ChatContainer';

interface MessageBubbleProps {
  message: Message;
  onCopyCode: (code: string) => void;
  onDownloadCode: (code: string, filename: string) => void;
}

export function MessageBubble({ message, onCopyCode, onDownloadCode }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.type === 'user';

  const handleCopy = () => {
    if (message.code) {
      onCopyCode(message.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('ro-RO', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
          isUser
            ? 'bg-gradient-to-br from-blue-500 to-cyan-500'
            : 'bg-gradient-to-br from-pink-600 to-orange-500'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Conținut */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-tr-sm'
              : 'bg-slate-800 text-slate-100 rounded-tl-sm'
          }`}
        >
          {/* Mesaj principal */}
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Răspuns AI extins */}
        {!isUser && (message.interpretation || message.params || message.code) && (
          <div className="mt-3 space-y-3">
            {/* Interpretare */}
            {message.interpretation && (
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                  Interpretare
                </h4>
                <p className="text-sm text-slate-300">{message.interpretation}</p>
              </div>
            )}

            {/* Parametri detectați */}
            {message.params && Object.keys(message.params).length > 0 && (
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                  Parametri Detectați
                </h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(message.params).map(([key, value]) => (
                    <span
                      key={key}
                      className="inline-flex items-center gap-1 px-2 py-1 bg-slate-700 rounded-lg text-xs"
                    >
                      <span className="text-slate-400">{key}:</span>
                      <span className="text-pink-400 font-mono">{String(value)}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Cod generat */}
            {message.code && (
              <div className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden">
                <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
                  <span className="text-xs font-medium text-slate-400">
                    Python (Blender Script)
                  </span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-1 px-2 py-1 text-xs text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
                    >
                      {copied ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-green-400" />
                          <span className="text-green-400">Copiat!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copiază</span>
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => onDownloadCode(message.code!, 'blender_script.py')}
                      className="flex items-center gap-1 px-2 py-1 text-xs text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Descarcă</span>
                    </button>
                  </div>
                </div>
                <CodeBlock code={message.code} />
              </div>
            )}
          </div>
        )}

        {/* Timestamp */}
        <p className={`text-xs text-slate-500 mt-1 ${isUser ? 'text-right' : ''}`}>
          {formatTime(message.timestamp)}
        </p>
      </div>
    </div>
  );
}
