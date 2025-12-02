interface CodeBlockProps {
  code: string;
}

export function CodeBlock({ code }: CodeBlockProps) {
  // Syntax highlighting simplu pentru Python
  const highlightCode = (code: string) => {
    const lines = code.split('\n');
    
    return lines.map((line, lineIndex) => {
      let highlighted = line;
      
      // Comentarii
      if (line.trim().startsWith('#')) {
        return (
          <div key={lineIndex} className="text-slate-500 italic">
            {line}
          </div>
        );
      }
      
      // Keywords Python
      const keywords = ['import', 'from', 'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'as', 'with', 'try', 'except', 'finally'];
      keywords.forEach(keyword => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'g');
        highlighted = highlighted.replace(regex, `<span class="text-purple-400">${keyword}</span>`);
      });
      
      // Strings
      highlighted = highlighted.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="text-green-400">$&</span>');
      
      // Numere
      highlighted = highlighted.replace(/\b(\d+\.?\d*)\b/g, '<span class="text-orange-400">$1</span>');
      
      // bpy și ops
      highlighted = highlighted.replace(/\b(bpy)\b/g, '<span class="text-cyan-400">$1</span>');
      highlighted = highlighted.replace(/\.(ops|data|context|types)\b/g, '.<span class="text-yellow-400">$1</span>');
      
      // Funcții
      highlighted = highlighted.replace(/(\w+)\(/g, '<span class="text-blue-400">$1</span>(');
      
      return (
        <div 
          key={lineIndex} 
          dangerouslySetInnerHTML={{ __html: highlighted || '&nbsp;' }} 
        />
      );
    });
  };

  return (
    <pre className="p-4 text-sm font-mono overflow-x-auto">
      <code className="text-slate-300">
        {highlightCode(code)}
      </code>
    </pre>
  );
}
