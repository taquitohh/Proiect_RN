import { Plus, MessageSquare, Trash2, Box, Cylinder, Circle, Hexagon, Download, Palette } from 'lucide-react';

interface Conversation {
  id: string;
  title: string;
  date: string;
}

interface SidebarProps {
  conversations: Conversation[];
  activeConversation: string | null;
  onNewConversation: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onTemplateClick: (text: string) => void;
}

const templates = [
  { icon: Box, text: 'Cub', color: 'text-blue-400' },
  { icon: Circle, text: 'Sferă', color: 'text-red-400' },
  { icon: Cylinder, text: 'Cilindru', color: 'text-green-400' },
  { icon: Hexagon, text: 'Material', color: 'text-yellow-400' },
  { icon: Palette, text: 'Sticlă', color: 'text-cyan-400' },
  { icon: Download, text: 'Export', color: 'text-purple-400' },
];

export function Sidebar({
  conversations,
  activeConversation,
  onNewConversation,
  onSelectConversation,
  onDeleteConversation,
  onTemplateClick,
}: SidebarProps) {
  return (
    <aside className="w-52 bg-slate-900 border-r border-slate-700 fixed left-0 top-14 bottom-0 flex flex-col z-40">
      {/* Șabloane Rapide - în partea de sus */}
      <div className="p-3 pb-2">
        <h3 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
          Șabloane
        </h3>
        <div className="grid grid-cols-3 gap-1">
          {templates.map((template, index) => (
            <button
              key={index}
              onClick={() => onTemplateClick(template.text)}
              className="flex flex-col items-center gap-1 px-1 py-1.5 bg-slate-800 hover:bg-slate-700 rounded text-[10px] text-slate-300 hover:text-white transition-colors"
            >
              <template.icon className={`w-3 h-3 ${template.color}`} />
              <span className="truncate text-center w-full">{template.text}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Separator */}
      <div className="border-t border-slate-700 mx-3"></div>

      {/* Istoric Conversații */}
      <div className="flex-1 overflow-y-auto px-2 pt-2">
        <h3 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider px-2 mb-1">
          Istoric
        </h3>
        <div className="space-y-0.5">
          {conversations.length === 0 ? (
            <p className="text-xs text-slate-500 px-2 py-3 text-center">
              Nicio conversație
            </p>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group flex items-center gap-1.5 px-2 py-1.5 rounded cursor-pointer transition-colors ${
                  activeConversation === conv.id
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                }`}
                onClick={() => onSelectConversation(conv.id)}
              >
                <MessageSquare className="w-3 h-3 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs truncate">{conv.title}</p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-red-500/20 rounded transition-all"
                >
                  <Trash2 className="w-3 h-3 text-red-400" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Buton Conversație Nouă - în josul paginii */}
      <div className="border-t border-slate-700 p-3 pb-4 bg-slate-900">
        <button
          onClick={onNewConversation}
          className="w-full flex items-center justify-center gap-1.5 py-2 px-3 bg-gradient-to-r from-pink-600 to-orange-500 hover:from-pink-500 hover:to-orange-400 text-white text-sm font-medium rounded-lg transition-all"
        >
          <Plus className="w-4 h-4" />
          Conversație nouă
        </button>
      </div>
    </aside>
  );
}
