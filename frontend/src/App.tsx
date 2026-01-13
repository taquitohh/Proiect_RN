import { useState, useCallback, useEffect } from 'react';
import { Header } from './components/Layout/Header';
import { Sidebar } from './components/Layout/Sidebar';
import { ChatContainer, ChatInput, Message } from './components/Chat';
import api from './api';

interface Conversation {
  id: string;
  title: string;
  date: string;
  messages: Message[];
}

// Exemplu de cod pentru demo
//const generateExampleCode = (intent: string, params: Record<string, unknown>) => {
//  const templates: Record<string, string> = {
//    create_cube: `import bpy
//
//# Ștergem obiectele existente (opțional)
//# bpy.ops.object.select_all(action='SELECT')
//# bpy.ops.object.delete()
//
//# Creăm un cub
//bpy.ops.mesh.primitive_cube_add(
//    size=${params.size || 2},
//    location=(${params.x || 0}, ${params.y || 0}, ${params.z || 0})
//)
//
//# Redenumim obiectul
//cube = bpy.context.active_object
//cube.name = "Cube_Generated"
//
//print("Cub creat cu succes!")`,
//
//    create_sphere: `import bpy
//
//# Creăm o sferă
//bpy.ops.mesh.primitive_uv_sphere_add(
//    radius=${params.radius || 1},
//    location=(${params.x || 0}, ${params.y || 0}, ${params.z || 0}),
//    segments=32,
//    ring_count=16
//)
//
//# Redenumim obiectul
//sphere = bpy.context.active_object
//sphere.name = "Sphere_Generated"
//
//# Aplicăm smooth shading
//bpy.ops.object.shade_smooth()
//
//print("Sferă creată cu succes!")`,
//
//    create_cylinder: `import bpy
//
//# Creăm un cilindru
//bpy.ops.mesh.primitive_cylinder_add(
//    radius=${params.radius || 1},
//    depth=${params.height || 2},
//    location=(${params.x || 0}, ${params.y || 0}, ${params.z || 0})
//)
//
//# Redenumim obiectul
//cylinder = bpy.context.active_object
//cylinder.name = "Cylinder_Generated"
//
//print("Cilindru creat cu succes!")`,
//
//    apply_material: `import bpy
//
//# Obținem obiectul activ
//obj = bpy.context.active_object
//
//# Creăm un material nou
//mat = bpy.data.materials.new(name="${params.material_name || 'Material_Generated'}")
//mat.use_nodes = True
//
//# Configurăm culoarea
//nodes = mat.node_tree.nodes
//principled = nodes.get("Principled BSDF")
//if principled:
//    principled.inputs["Base Color"].default_value = (${params.r || 0.8}, ${params.g || 0.2}, ${params.b || 0.2}, 1.0)
//    principled.inputs["Metallic"].default_value = ${params.metallic || 0.0}
//    principled.inputs["Roughness"].default_value = ${params.roughness || 0.5}
//
//# Aplicăm materialul
//if obj.data.materials:
//    obj.data.materials[0] = mat
//else:
//    obj.data.materials.append(mat)
//
//print("Material aplicat cu succes!")`,
//  };
//
//  return templates[intent] || templates.create_cube;
//};

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [inputValue, setInputValue] = useState('');
  
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  // Check API connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status');
        setIsConnected(response.ok);
      } catch {
        setIsConnected(false);
      }
    };
    
    checkConnection();
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  const generateId = () => Math.random().toString(36).substring(2, 9);

  const handleNewConversation = useCallback(() => {
    const newConv: Conversation = {
      id: generateId(),
      title: 'Conversație nouă',
      date: new Date().toLocaleDateString('ro-RO'),
      messages: [],
    };
    setConversations(prev => [newConv, ...prev]);
    setActiveConversationId(newConv.id);
    setMessages([]);
  }, []);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveConversationId(id);
    const conv = conversations.find(c => c.id === id);
    setMessages(conv?.messages || []);
  }, [conversations]);

  const handleDeleteConversation = useCallback((id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeConversationId === id) {
      setActiveConversationId(null);
      setMessages([]);
    }
  }, [activeConversationId]);

  const handleTemplateClick = useCallback((text: string) => {
    setInputValue(text);
  }, []);

  const handleSendMessage = useCallback(async (content: string) => {
    // Adaugă mesajul utilizatorului
    const userMessage: Message = {
      id: generateId(),
      type: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    let interpretation = '';
    let params: Record<string, unknown> = {};
    let code = '';
    let intent = '';

    try {
      // Apelăm API-ul backend pentru a genera codul
      const response = await api.generateBlenderCode(content);
      
      if (response.data.success) {
        intent = response.data.intent;
        interpretation = response.data.interpretation;
        params = response.data.params;
        code = response.data.code;
      } else {
        interpretation = 'Eroare la procesarea cererii.';
        code = '# Eroare la generare cod';
      }
    } catch (error) {
      // Fallback la generare locală dacă backend-ul nu e disponibil
      //console.warn('Backend indisponibil, folosesc generare locală:', error);
      //
      //const lowerContent = content.toLowerCase();
      //
      //if (lowerContent.includes('sfer')) {
      //  intent = 'create_sphere';
      //  interpretation = 'Am înțeles că vrei să creezi o sferă. (offline mode)';
      //  params = { radius: 1 };
      //} else if (lowerContent.includes('cilindru')) {
      //  intent = 'create_cylinder';
      //  interpretation = 'Am înțeles că vrei să creezi un cilindru. (offline mode)';
      //  params = { radius: 1, depth: 2 };
      //} else if (lowerContent.includes('cub')) {
      //  intent = 'create_cube';
      //  interpretation = 'Am înțeles că vrei să creezi un cub. (offline mode)';
      //  params = { size: 2 };
      //} else {
      //  intent = 'create_cube';
      //  interpretation = 'Generez un cub implicit. (offline mode)';
      //  params = { size: 2 };
      //}
      
      //code = generateExampleCode(intent, params);
    }

    // Răspunsul AI
    const aiMessage: Message = {
      id: generateId(),
      type: 'ai',
      content: 'Am generat codul Blender pentru cererea ta!',
      timestamp: new Date(),
      interpretation,
      params,
      code,
    };

    setMessages(prev => [...prev, aiMessage]);
    setIsLoading(false);

    // Actualizează conversația
    if (activeConversationId) {
      setConversations(prev =>
        prev.map(c =>
          c.id === activeConversationId
            ? { ...c, messages: [...c.messages, userMessage, aiMessage], title: content.slice(0, 30) + '...' }
            : c
        )
      );
    } else {
      // Creează conversație nouă
      const newConv: Conversation = {
        id: generateId(),
        title: content.slice(0, 30) + '...',
        date: new Date().toLocaleDateString('ro-RO'),
        messages: [userMessage, aiMessage],
      };
      setConversations(prev => [newConv, ...prev]);
      setActiveConversationId(newConv.id);
    }
  }, [activeConversationId]);

  const handleCopyCode = useCallback((code: string) => {
    navigator.clipboard.writeText(code);
  }, []);

  const handleDownloadCode = useCallback((code: string, filename: string) => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  return (
    <div className={`h-screen overflow-hidden ${darkMode ? 'bg-slate-950' : 'bg-gray-100'}`}>
      <Header
        isConnected={isConnected}
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode(!darkMode)}
      />

      <Sidebar
        conversations={conversations}
        activeConversation={activeConversationId}
        onNewConversation={handleNewConversation}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        onTemplateClick={handleTemplateClick}
      />

      {/* Main Content */}
      <main className="ml-52 pt-14 h-full flex flex-col overflow-hidden">
        <ChatContainer
          messages={messages}
          isLoading={isLoading}
          onCopyCode={handleCopyCode}
          onDownloadCode={handleDownloadCode}
        />

        <ChatInput
          onSend={handleSendMessage}
          isLoading={isLoading}
          inputValue={inputValue}
          onInputChange={setInputValue}
        />
      </main>
    </div>
  );
}

export default App;
