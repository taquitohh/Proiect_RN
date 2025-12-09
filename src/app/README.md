# 🖥️ Web Service / UI Module

## Descriere
Interfața utilizator pentru sistemul **Text-to-Blender AI** - permite generarea de cod Python Blender din descrieri în limbaj natural.

## Structura completă
```
Proiect_RN/
├── src/
│   ├── app/              # Acest folder (documentație)
│   └── api.py            # Backend Flask API
└── frontend/             # React UI
    ├── src/
    │   ├── App.tsx       # Componenta principală
    │   ├── api/          # Servicii API
    │   └── components/   # Componente UI
    └── package.json
```

## Componente

### Backend (Flask API)
**Fișier:** `src/api.py`  
**Port:** 8000

Endpoints principale:
| Endpoint | Metodă | Descriere |
|----------|--------|-----------|
| `/api/status` | GET | Status server și model |
| `/api/blender/generate` | POST | Generează cod Blender din text |
| `/api/blender/intents` | GET | Lista intenții disponibile |

### Frontend (React + TypeScript)
**Folder:** `frontend/`  
**Port:** 5173

Componente UI:
- **Header** - Bară navigare cu status conexiune
- **Sidebar** - Istoric conversații + template-uri rapide
- **ChatContainer** - Afișare mesaje și cod generat
- **ChatInput** - Input pentru comenzi text
- **CodeBlock** - Afișare cod cu syntax highlighting

## Comenzi de lansare

### Backend
```powershell
cd e:\github\Proiect_RN\src
& ".\.venv\Scripts\python.exe" api.py
```
Server pornește pe: http://localhost:8000

### Frontend
```powershell
cd e:\github\Proiect_RN\frontend
npm run dev
```
UI disponibil pe: http://localhost:5173

## Flow utilizator
```
1. User introduce text: "creează un cub roșu"
          ↓
2. Frontend trimite POST la /api/blender/generate
          ↓
3. Backend: clasificare intenție → extragere parametri → generare cod
          ↓
4. Frontend afișează codul Python generat
          ↓
5. User copiază codul în Blender și îl execută
```

## Screenshot
Vezi: `docs/screenshots/ui_demo.png`

## Tehnologii folosite
- **Backend:** Flask 3.1, Flask-CORS, PyTorch
- **Frontend:** React 18, TypeScript, Vite, Tailwind CSS
- **Comunicare:** REST API (JSON)
