# Raport de Progres Sintetic - Proiect Rețele Neuronale

---

# 📌 ETAPA 1: Infrastructură Inițială (Arhitectură Client-Server)

## 1. Rezumat Executiv
Proiectul a fost inițializat și dezvoltat complet la nivel de cod sursă, urmând o arhitectură modernă Client-Server. Sistemul integrează un backend robust bazat pe Python (FastAPI & PyTorch) cu un frontend reactiv (React & Tailwind CSS).

## 2. Realizări Tehnice

### A. Infrastructură și Mediu
- [x] **Structura Proiectului:** Organizare modulară (`src`, `frontend`, `data`, `config`).
- [x] **Configurare Mediu:**
  - Python 3.12.10 instalat și configurat.
  - Node.js 24.11.1 instalat.
  - Mediu virtual (`venv`) creat și activat.
  - Dependențe Python instalate (`torch`, `fastapi`, `pandas`, etc.).
  - Dependențe Node.js instalate.

### B. Backend (Python & AI)
- [x] **API Server:** Implementat cu **FastAPI** (`src/api.py`) pentru a expune endpoint-uri de antrenament și predicție.
- [x] **Model AI:** Arhitectură de rețea neuronală definită în **PyTorch** (`src/neural_network/model.py`).
- [x] **Pipeline de Date:**
  - Modul de achiziție date (`src/data_acquisition/data_loader.py`).
  - Modul de preprocesare și curățare (`src/preprocessing/preprocessor.py`).

### C. Frontend (React & TypeScript)
- [x] **Framework:** Aplicație creată cu **Vite** pentru performanță optimă.
- [x] **UI/UX:** Stilizare modernă folosind **Tailwind CSS**.
- [x] **Componente Implementate:**
  - `FileUpload.tsx`: Încărcarea seturilor de date.
  - `DataTable.tsx`: Vizualizarea datelor tabulare.
  - `TrainingChart.tsx`: Vizualizarea metricilor de antrenament în timp real.

## 3. Status Etapa 1
Sistemul este complet implementat ("code-complete"). Mediul de dezvoltare VS Code a fost configurat pentru a recunoaște interpretorul Python din mediul virtual, eliminând erorile de analiză statică (Pylance).

---

# 📌 ETAPA 2: Pivot către Text-to-Blender AI

## 4. Motivul Schimbării Direcției

Inițial, proiectul a fost conceput ca un sistem generic de clasificare pe date tabulare (CSV). După analiza cerințelor profesorului și a planului de bătaie, am identificat că proiectul trebuie să rezolve o **problemă concretă din industrie**.

### De ce am pivotat:
1. **Cerința academică:** Proiectul trebuie să demonstreze aplicabilitate într-un **Domeniu Industrial de Interes (DII)**
2. **Unicitate:** Text-to-Blender este un domeniu mai puțin explorat față de clasificarea generică
3. **Reutilizare infrastructură:** Arhitectura FastAPI + React rămâne 100% validă
4. **Valoare practică:** Automatizarea generării de piese 3D are aplicații reale în CAD/CAM

### Ce am păstrat din Etapa 1:
- ✅ Structura de foldere (`src`, `frontend`, `data`, `config`)
- ✅ FastAPI ca backend server
- ✅ React + Vite + Tailwind ca frontend
- ✅ PyTorch pentru rețeaua neuronală
- ✅ Mediul virtual și configurările VS Code

### Ce am modificat/adăugat:
- 🔄 **Input:** CSV tabular → Text în limbaj natural
- 🔄 **Preprocesare:** Normalizare numerică → Tokenizare text (Bag of Words)
- 🔄 **Output:** Clasă numerică → Script Python pentru Blender
- ➕ **Nou:** Generator de scripturi Blender (`src/generators/`)
- ➕ **Nou:** Mock BPY pentru testare (`src/bpy.py`)
- ➕ **Nou:** Dataset NLP cu 175+ exemple

---

## 5. Noua Direcție: Text-to-Blender AI

### 5.1. Obiectivul Sistemului
Dezvoltarea unui sistem AI capabil să:
1. Primească o descriere textuală (ex: "creează un cilindru de 2m înălțime")
2. Interpreteze textul automat folosind NLP
3. Genereze un script Blender Python (bpy)
4. Construiască obiectul 3D cerut în Blender

**Flow:** `Text → Interpretare AI → Parametri → Cod Python → Blender → Piesă 3D`

### 5.2. Domeniul Industrial
**Producție și Design 3D asistat de calculator (CAD)**
- Automatizarea generării de piese 3D
- Reducerea timpului de prototipare cu 80-90%
- Standardizarea pieselor (evitarea erorilor umane)

---

## 6. Realizări Tehnice - Etapa 2

### A. Dataset de Antrenare
- [x] **175+ exemple** în `data/raw/blender_training_dataset.json`
- [x] Format: `{"text": "...", "intent": "...", "params": {...}}`
- [x] Acoperire categorii:
  - Primitive 3D: cuburi, sfere, cilindri, conuri, torusuri, planuri
  - Operații: creare, ștergere, mutare, rotire, scalare, duplicare
  - Materiale: culori (roșu, albastru, verde), texturi (metal, sticlă)
  - Modifiers: Bevel, Mirror, Array, Subdivision
  - Export: FBX, OBJ, STL

### B. Scripturi Blender (`data/raw/blender_scripts/`)
- [x] **46 scripturi Python** funcționale pentru Blender API
- [x] Fiecare script este documentat cu intent și parametri
- [x] Exemple: `create_cube_basic.py`, `apply_material_metal.py`, `add_modifier_bevel.py`

### C. Module Backend Noi

#### TextPreprocessor (`src/preprocessing/preprocessor.py`)
- Transformă text în vectori numerici (Bag of Words)
- Construiește vocabular din datele de antrenament
- Mapează intenții la indici pentru clasificare

#### BlenderScriptGenerator (`src/generators/blender_generator.py`)
- Primește intenția clasificată de AI + parametri extrași
- Completează template-uri cu valorile corespunzătoare
- Generează cod Python valid pentru Blender

#### DataLoader (`src/data_acquisition/data_loader.py`)
- Funcție nouă `load_training_data()` pentru JSON/CSV NLP
- Compatibilitate cu formatul vechi (CSV tabular)

#### Mock BPY (`src/bpy.py`)
- Simulează Blender Python API pentru testare
- Permite rularea scripturilor fără Blender instalat
- Afișează acțiunile executate prin mesaje `[MOCK]`

---

## 7. Arhitectura Sistemului Actualizată

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI        │────▶│  Neural Network │
│   (React)       │     │   Backend        │     │  (PyTorch)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        │                       ▼                        ▼
        │               ┌──────────────────┐     ┌─────────────────┐
        │               │ TextPreprocessor │     │ Intent + Params │
        │               │ (Bag of Words)   │     │ Classification  │
        │               └──────────────────┘     └─────────────────┘
        │                                                │
        │                                                ▼
        │                                        ┌─────────────────┐
        │                                        │ BlenderScript   │
        │                                        │ Generator       │
        │                                        └─────────────────┘
        │                                                │
        ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│ Chat UI         │◀─────────────────────────────│ Script Python   │
│ (Input/Output)  │                              │ pentru Blender  │
└─────────────────┘                              └─────────────────┘
```

---

## 8. Structura Proiectului Completă

```
Proiect_RN/
├── config/                     # Configurări YAML
│   ├── model_config.yaml
│   └── preprocessing_config.yaml
├── data/
│   ├── raw/
│   │   ├── blender_training_dataset.json  # 175+ exemple NLP
│   │   ├── blender_training_data.json     # Format inițial
│   │   └── blender_scripts/               # 46 scripturi Python
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── docs/                       # Documentație
├── frontend/                   # React + TypeScript + Vite
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/
│   │   └── components/
│   └── package.json
├── models/                     # Checkpoint-uri model salvate
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI server
│   ├── bpy.py                  # Mock Blender API (NOU)
│   ├── data_acquisition/
│   │   └── data_loader.py      # Încărcare JSON/CSV
│   ├── generators/             # (NOU)
│   │   └── blender_generator.py
│   ├── neural_network/
│   │   └── model.py            # Rețea neuronală PyTorch
│   └── preprocessing/
│       └── preprocessor.py     # TextPreprocessor (NOU)
├── venv/                       # Mediu virtual Python
├── .vscode/
│   └── settings.json           # Configurare interpretor
├── requirements.txt
├── README.md
└── PROGRES_SINTETIC.md         # Acest fișier
```

---

## 9. Pași Următori

- [ ] Actualizare model neuronal pentru clasificare intenții NLP
- [ ] Implementare endpoint `/api/predict` pentru text-to-script
- [ ] Refacere interfață frontend (Chat UI în loc de upload CSV)
- [ ] Antrenare model pe dataset-ul de 175+ exemple
- [ ] Testare integrată cu Blender
- [ ] Evaluare performanță (accuracy, F1-score)

---

## 10. Resurse și Referințe
- NVIDIA GET3D - A Generative Model of 3D Objects
- OpenAI GPT-4 Technical Report - Multimodal AI for code generation
- Google DreamFusion: Text-to-3D
- Blender Python API Documentation
- Deep Learning for CAD Model Generation (IEEE)

---

## 🔗 Repository GitHub
**[https://github.com/taquitohh/Proiect_RN](https://github.com/taquitohh/Proiect_RN)**
