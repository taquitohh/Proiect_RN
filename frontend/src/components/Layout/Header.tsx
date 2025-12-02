import { Box, Moon, Sun } from 'lucide-react';

interface HeaderProps {
  isConnected: boolean;
  darkMode: boolean;
  onToggleTheme: () => void;
}

export function Header({ isConnected, darkMode, onToggleTheme }: HeaderProps) {
  return (
    <header className="h-14 bg-slate-900 border-b border-slate-700 flex items-center justify-between px-4 fixed top-0 left-0 right-0 z-50">
      {/* Logo și Titlu */}
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-pink-600 rounded-lg flex items-center justify-center">
          <Box className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-white leading-tight">Blender AI</h1>
          <p className="text-[10px] text-slate-400">Text-to-3D</p>
        </div>
      </div>

      {/* Status și Controale */}
      <div className="flex items-center gap-3">
        {/* Status Conexiune */}
        <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-slate-800">
          <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className={`text-xs ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
            {isConnected ? 'Online' : 'Offline'}
          </span>
        </div>

        {/* Toggle Theme */}
        <button
          onClick={onToggleTheme}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
        >
          {darkMode ? (
            <Sun className="w-4 h-4 text-yellow-400" />
          ) : (
            <Moon className="w-4 h-4 text-slate-400" />
          )}
        </button>
      </div>
    </header>
  );
}
